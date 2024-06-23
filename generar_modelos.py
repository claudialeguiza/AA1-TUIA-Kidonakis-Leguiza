# -*- coding: utf-8 -*-
"""generar_modelos.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1mOW7MKodOCy-yQGaTwHjqWuhTcoEnQ6-

<a href="https://colab.research.google.com/github/claudialeguiza/AA1-TUIA-Kidonakis-Leguiza/blob/navegador/generar_modelos.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import RobustScaler, FunctionTransformer
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import Adam
from keras.metrics import Precision
from keras.utils import to_categorical
import joblib
import warnings
warnings.simplefilter('ignore')

datos = pd.read_csv('/content/weatherAUS.csv', delimiter = ",")

df = datos[datos.Location\
                      .isin(( 'Sydney','SydneyAirport','Melbourne', 'MelbourneAirport',\
                             'Canberra','Adelaide', 'MountGambier','Cobar', 'Dartmoor' ))]

def preprocesamiento(data):
    data.info()
    data.isna().sum()

    # Definir columnas con valores nulos
    columnas_con_nulos = ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine',
                          'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm','Humidity9am',
                          'Humidity3pm', 'Pressure9am','Pressure3pm', 'Cloud9am',
                          'Cloud3pm', 'Temp9am', 'Temp3pm', 'RainfallTomorrow']

    # Rellenar valores faltantes en 'RainToday' y 'RainTomorrow'
    data['RainToday'] = data.groupby('Date')['RainToday'].transform(lambda x: x.fillna(x.mode().iloc[0]))
    data['RainTomorrow'] = data.groupby('Date')['RainTomorrow'].transform(lambda x: x.fillna(x.mode().iloc[0]))

    # Rellenar valores faltantes en direcciones del viento
    data['WindGustDir'] = data.groupby('Date')['WindGustDir'].transform(lambda x: x.fillna(x.mode().iloc[0]) if not x.isna().all() else x)
    data['WindDir9am'] = data.groupby('Date')['WindDir9am'].transform(lambda x: x.fillna(x.mode().iloc[0]) if not x.isna().all() else x)
    data['WindDir3pm'] = data.groupby('Date')['WindDir3pm'].transform(lambda x: x.fillna(x.mode().iloc[0]) if not x.isna().all() else x)

    # Rellenar valores faltantes con la media por día para las columnas especificadas
    media_por_dia = data.groupby('Date')[columnas_con_nulos].transform('mean')
    data[columnas_con_nulos] = data[columnas_con_nulos].fillna(media_por_dia)

    data['Date'] = pd.to_datetime(data['Date'])

    return data

def crear_columna_season(data):
   data['season'] = data['Date'].apply(asignar_estacion)
   return data

def asignar_estacion(fecha):
    mes = fecha.month
    if mes in [12, 1, 2]:  # Verano: Diciembre, Enero, Febrero
        return 'Summer'
    elif mes in [3, 4, 5]:  # Otoño: Marzo, Abril, Mayo
        return 'Autumn'
    elif mes in [6, 7, 8]:  # Invierno: Junio, Julio, Agosto
        return 'Winter'
    else:  # Primavera: Septiembre, Octubre, Noviembre
        return 'Spring'

def codificar_variables(data):
    data1 = pd.get_dummies(data, columns=['RainToday', 'RainTomorrow', 'season', 'Location'], drop_first=True)

    # Crear columnas para WindGustDir, WindDir9am, WindDir3pm
    wind_directions = ["SW", "S", 'SSW', 'W', 'SSE', 'E', 'SE', 'NE', 'NNE', 'WSW', 'WNW', 'NW', 'N', 'ESE', 'ENE']
    for var in wind_directions:
        data1[f'WindGustDir_{var}'] = (data['WindGustDir'] == var).astype(int)
        data1[f'WindDir9am_{var}'] = (data['WindDir9am'] == var).astype(int)
        data1[f'WindDir3pm_{var}'] = (data['WindDir3pm'] == var).astype(int)

    return data1.drop(columns=['WindGustDir', 'WindDir9am', 'WindDir3pm'])

def robust_df(data):
  scaler = RobustScaler()
  data_scaled = scaler.fit_transform(data)
  return data_scaled

def truncar_dividir_df(data):
    data = data.sort_values(["Date"])
    fecha_especifica = '2009-01-01'
    data_filtrada = data[data['Date'] >= fecha_especifica]

    data_filtrada.reset_index(drop=True, inplace=True)  # Resetea el índice y no crea uno nuevo
    data_train = data_filtrada.iloc[:21658]
    #data_test = data_filtrada.iloc[21658:]

    return data_train

def eliminar_columnas_estandarizar(data):
    # Separar variables independientes y dependientes
    X_regresion = data.drop(columns =['RainfallTomorrow','Date'])
    X_scaled = robust_df(X_regresion)
    y_regresion = data['RainfallTomorrow']
    y_scaled = robust_df(y_regresion.values.reshape(-1,1))
    return X_scaled, y_scaled

def estandarizar_balancear_clas(data):
    X_clasificacion = data.drop(columns=['RainTomorrow_Yes','Date'])
    X_scaled1 = robust_df(X_clasificacion)
    y_clasificacion = data['RainTomorrow_Yes']
    y_scaled1 =robust_df(y_clasificacion.values.reshape(-1,1))
    smote = SMOTE(random_state=42)
    X_smote_scaled, y_smote_scaled = smote.fit_resample(X_scaled1, y_scaled1)

    return X_smote_scaled, y_smote_scaled

def cargar_modelo_regresion():
    # Cargamos el modelo
      modelo_regresion = load_model('/content/regression_model.h5')
      return modelo_regresion

def cargar_modelo_clasificacion():
      modelo_clasif = load_model('/content/classification_model_optimized.h5')
      return modelo_clasif

pipeline_prepara_datos = Pipeline([
    ('preproceso', FunctionTransformer(preprocesamiento, validate=False)),
    ('season', FunctionTransformer(crear_columna_season, validate=False)),
    ('codificar', FunctionTransformer(codificar_variables, validate=False))
])

# Obtener datos de entrenamiento
df_procesado = pipeline_prepara_datos.fit_transform(df)

pipeline_train_split = Pipeline([
    ('split', FunctionTransformer(truncar_dividir_df, validate=False)),
    ('estandarizar', FunctionTransformer(eliminar_columnas_estandarizar, validate=False)),
    ])

pipeline_modelo_regresion = Pipeline([
     ('modelo', FunctionTransformer(cargar_modelo_regresion, validate=False))
                                      ])
# Obtener datos de entrenamiento
X_train_scaled, y_train_scaled = pipeline_train_split.fit_transform(df_procesado)

 # Entrenar el modelo
pipeline_modelo_regresion.fit(X_train_scaled, y_train_scaled)

joblib.dump(pipeline_modelo_regresion, 'regresion_pipeline.joblib')

pipeline_train_split_clas = Pipeline([
    ('split', FunctionTransformer(truncar_dividir_df, validate=False)),
    ('estandarizar_clas', FunctionTransformer(estandarizar_balancear_clas, validate=False)),
    ])

pipeline_modelo_clasificacion = Pipeline([
     ('modelo_clas', FunctionTransformer(cargar_modelo_clasificacion, validate=False))
                                      ])
# Obtener datos de entrenamiento
X_smote, y_smote = pipeline_train_split_clas.fit_transform(df_procesado)

# Entrenar_modelo
pipeline_modelo_clasificacion.fit(X_smote, y_smote)

joblib.dump(pipeline_modelo_clasificacion, 'clasificacion_pipeline.joblib')