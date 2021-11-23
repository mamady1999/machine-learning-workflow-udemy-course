import pandas as pd

def extraire_premiere_lettre(serie):
  #Récupère une série en argument
  #Retourne une DataFrame
  return pd.DataFrame(serie.str.get(0))