# Utiliser une image Python officielle comme image parente
FROM python:3.8-slim

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier les fichiers nécessaires dans le conteneur
COPY requirements.txt ./
COPY keras_model.h5 ./
COPY infomodel.py ./
COPY application.py ./
COPY static ./static/
COPY templates ./templates/
COPY labels.txt ./

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Exposer le port sur lequel l'application s'exécutera
EXPOSE 8000

# Commande pour exécuter l'application
CMD ["python", "./application.py"]
