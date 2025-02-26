# Utiliser une image Python officielle
FROM python:3.9

# Définir le dossier de travail
WORKDIR /app

# Copier le fichier de dépendances et installer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier l'ensemble du projet
COPY . .

# Exposer le port utilisé par uvicorn
EXPOSE 8000

# Lancer l'application via le script (uvicorn démarre grâce à __main__)
CMD ["python", "api.py"]
