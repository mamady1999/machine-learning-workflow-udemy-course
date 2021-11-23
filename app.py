import sklearn
import pandas as pd
from flask import Flask, request
import numpy as np 
import joblib
from fonction_maison import extraire_premiere_lettre


#Load model
pipeline = joblib.load("titanic.model")
print(pipeline)

#Démarrer l'application flask (Cette partie est obligatoire)
app = Flask("__name__")

#Tester l'api
@app.route('/ping', methods = ["GET"])
def ping():
  return ("pong", 200)

#Faire des prédictions 
@app.route('/prediction', methods = ["POST"])
def prediction():
  #Accéder à la réquête
  df = pd.DataFrame(request.json)
  #print(df)
  resultat = pipeline.predict(df)[0]
  return (str(resultat), 201)

#Page d'accueil
@app.route("/")
def index():
  return "<h1>Bienvenue dans notre API. Utiliser /predict en POST pour faire des prédictions sur le titanic</h1>"

if __name__ == "__main__":
  app.run(host = "0.0.0.0")
