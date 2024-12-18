import pickle 
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
import os
import plotly.express as px  # Pour les graphiques interactifs

# Initialisation de l'application Flask
app = Flask(__name__)

# Chargement du modèle sauvegardé avec pickle
with open('model.pkl', 'rb') as model_file:
    model_data = pickle.load(model_file)
    model = model_data["model"]  # Le modèle RandomForestRegressor
    model_features = model_data["features"]  # Liste des colonnes utilisées pour X

# Fonction pour lire les données depuis le fichier CSV
def get_data():
    data = pd.read_csv("data/car.csv", sep=";")

    return data

# Fonction de prédiction
def predict_price(name, transmission, year, km_driven, engine, max_power, mileage):
    x = np.zeros(len(model_features), dtype='float32')

    x[model_features.index('year')] = year
    x[model_features.index('km_driven')] = km_driven
    x[model_features.index('engine')] = engine
    x[model_features.index('max_power')] = max_power
    x[model_features.index('mileage')] = mileage

    if f"name_{name}" in model_features:
        x[model_features.index(f"name_{name}")] = 1
    if f"transmission_{transmission}" in model_features:
        x[model_features.index(f"transmission_{transmission}")] = 1

    prediction = model.predict([x])[0]
    return float(format(prediction, '.2f'))

# Route pour la page d'accueil
@app.route("/")
def home():
    return render_template("home.html")

# Route pour afficher l'analyse exploratoire
@app.route("/exploratory_analysis")
def exploratory_analysis():
    data = get_data()

    # Prix moyen par marque
    prix_moyen_marque = data.groupby("name")["selling_price"].mean().reset_index()
    fig = px.bar(prix_moyen_marque, x="name", y="selling_price", title="Prix moyen des voitures par marque")

    # Convertir le graphique en HTML
    graph_html = fig.to_html(full_html=False)

    return render_template("analysis.html", graph_html=graph_html)

# Route pour filtrer les données
@app.route("/filter", methods=["POST"])
def filter_data():
    marque = request.form.get("marque")
    annee = request.form.get("annee")

    data = get_data()

    # Filtrer les données selon les paramètres
    if marque:
        data = data[data["name"] == marque]
    if annee:
        data = data[data["year"] == int(annee)]

    # Prix moyen après filtre
    prix_moyen_marque = data.groupby("name")["selling_price"].mean().reset_index()
    fig = px.bar(prix_moyen_marque, x="name", y="selling_price", title="")

    graph_html = fig.to_html(full_html=False)

    return render_template("analysis.html", graph_html=graph_html, marque=marque, annee=annee)

# Page de prédiction
@app.route("/predict", methods=["GET", "POST"])
def predict():
    prediction = None
    error_message = None

    if request.method == "POST":
        try:
            name = request.form["name"]
            transmission = request.form["transmission"]
            year = int(request.form["year"])
            km_driven = float(request.form["km_driven"])
            engine = float(request.form["engine"])
            max_power = float(request.form["max_power"])
            mileage = float(request.form["mileage"])

            prediction = predict_price(name, transmission, year, km_driven, engine, max_power, mileage)
        except Exception as e:
            error_message = f"Erreur lors de la prédiction : {str(e)}"

    return render_template("predict.html", prediction=prediction, error_message=error_message)

# Lancement de l'application Flask
if __name__ == "__main__":
    app.run(debug=True, port=8091)
