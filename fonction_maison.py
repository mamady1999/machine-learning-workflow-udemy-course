import pandas as pd

def extraire_premiere_lettre(serie):
  #R�cup�re une s�rie en argument
  #Retourne une DataFrame
  return pd.DataFrame(serie.str.get(0))