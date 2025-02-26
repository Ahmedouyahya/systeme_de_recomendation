from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import pandas as pd
import json
import uvicorn
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Chargement des données et nettoyage
with open("movies.json", "r", encoding="utf-8") as f:
    movies_df = pd.read_json(f)
    movies_df = movies_df.dropna(subset=['title', 'description'])
    movies_df['title'] = movies_df['title'].astype(str)

# Conversion des clés de similarité en entiers
with open("cosine_sim.json", "r", encoding="utf-8") as f:
    cosine_sim_content = json.load(f)
    cosine_sim_content = {int(k): v for k, v in cosine_sim_content.items()}

# Simulation d'une matrice utilisateur-film
np.random.seed(42)
num_users = 100
user_item_matrix = pd.DataFrame(
    np.random.randint(1, 6, size=(num_users, len(movies_df))),
    columns=movies_df['title'].tolist()
)

# -------------------------------------------------------------------
# Content-Based (Basé sur le contenu des descriptions)
# -------------------------------------------------------------------
def recommend_content(title: str, top_n: int = 3):
    """Recommandation basée sur la similarité des descriptions"""
    if title not in movies_df['title'].tolist():
        return [] # Return empty list if movie not found for content-based

    idx = movies_df[movies_df['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim_content[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]

    return [(movies_df.iloc[i[0]]['title'], float(i[1])) for i in sim_scores]

# -------------------------------------------------------------------
# Collaborative User-Based (Basé sur les préférences utilisateur)
# -------------------------------------------------------------------
def recommend_collaborative(title: str, top_n: int = 3):
    """Recommandation basée sur les utilisateurs similaires"""
    if title not in user_item_matrix.columns:
        return [] # Return empty list if movie not found for collaborative

    # Calcul de la similarité entre utilisateurs
    user_similarity = cosine_similarity(user_item_matrix)

    # Trouver les utilisateurs qui ont aimé le film
    target_movie_ratings = user_item_matrix[title]
    similar_users = user_similarity.dot(target_movie_ratings)

    # Films les mieux notés par les utilisateurs similaires (weighted average)
    weighted_ratings = user_similarity.T.dot(user_item_matrix)

    # Calculate average weighted rating for each movie and sort
    recommended_movies_series = pd.Series(weighted_ratings.mean(axis=0), index=user_item_matrix.columns)
    recommended_movies_series = recommended_movies_series.sort_values(ascending=False)

    # Exclude the movie itself and get top_n recommendations
    recommended_movie_titles = recommended_movies_series[recommended_movies_series.index != title].head(top_n).index.tolist()
    return recommended_movie_titles


# -------------------------------------------------------------------
# Approche Hybride
# -------------------------------------------------------------------
def recommend_hybrid(title: str, content_weight: float = 0.5, top_n: int = 5):
    """Combine les recommandations content-based et collaborative"""
    content_reco = recommend_content(title, top_n * 2) # Get more content-based to have enough to pick from
    collab_reco = recommend_collaborative(title, top_n * 2) # Get more collaborative-based to have enough to pick from

    if not content_reco and not collab_reco:
        return []

    combined_scores = {}

    # Score content recommendations
    for i, (movie, score) in enumerate(content_reco):
        combined_scores[movie] = combined_scores.get(movie, 0) + (1 - content_weight) * score # Use similarity score directly

    # Score collaborative recommendations
    for i, movie in enumerate(collab_reco):
        combined_scores[movie] = combined_scores.get(movie, 0) + content_weight * (1/(i+1)) # Rank-based score

    # Sort combined recommendations by score
    hybrid_recommendations = sorted(combined_scores.items(), key=lambda item: item[1], reverse=True)[:top_n]
    return [(movie, float(score)) for movie, score in hybrid_recommendations]


# -------------------------------------------------------------------
# Endpoints FastAPI
# -------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "movie_list": movies_df['title'].tolist()}) # Pass movie list to template

@app.post("/recommend", response_class=HTMLResponse)
async def get_recommendations(request: Request, title: str = Form(...), approach: str = Form(...)):
    result = {"searched_title": title}

    try:
        if approach == "content":
            recommendations = recommend_content(title)
            result.update({
                "approach": "Content-Based",
                "recommendations": recommendations
            })
        elif approach == "collaborative":
            recommendations = recommend_collaborative(title)
            result.update({
                "approach": "Collaborative User-Based",
                "recommendations": recommendations
            })
        elif approach == "hybrid":
            recommendations = recommend_hybrid(title)
            result.update({
                "approach": "Hybride",
                "recommendations": recommendations
            })
        else:
            result = {"error": "Approche non reconnue"}

    except Exception as e:
        result = {"error": f"Erreur interne: {str(e)}"}

    return templates.TemplateResponse("results.html", {"request": request, "result": result})

if __name__ == "__main__":
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)