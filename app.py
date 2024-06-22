# -*- coding: utf-8 -*-
"""app.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/18o_QyptAfE2P7AOAwX6vILJajtznJxxw
"""

# !pip install streamlit

import streamlit as st
import pandas as pd
import joblib

# Título de la app
st.title('Pronóstico de lluvia para mañana')

# Cargamos el dataset para obtener el nombre
# de las columnas
df = pd.read_csv('C:\\Users\\solki\\OneDrive\\Documentos\\AA_Final\\AA1-TUIA-Kidonakis-Leguiza\\df_filtrado.csv', index_col=0)

# Cargamos los pipelines
path_regresion = 'models/regresion_pipeline.joblib'
path_clasificacion= 'models/clasificacion_pipeline.joblib'

pipeline_reg  = joblib.load(path_regresion)
pipeline_clas = joblib.load(path_clasificacion)

feature_names = pipeline_reg.named_steps['imputer']\
                            .get_feature_names_out()


# Definimos los nombres de las variables
columnas_numericas = list(df.columns[:-2])

#Creamos un slider para las variables
features = [st.slider(columna,
            df[columna].min(),
            df[columna].max(),
            round(df[columna].mean(), 2))
            for columna in columnas_numericas]

# Mapeamos la opción booleana a un texto
# y la agregamos para la predicción junto a
# las variables númericas
raintoday_option_mapping = {'Sí': 1, 'No': 0}
raintoday_option = st.selectbox('¿Hoy llovió?',
                                list(raintoday_option_mapping.keys()))

all_features = features + [raintoday_option_mapping[raintoday_option]]

data_para_predecir = pd.DataFrame([all_features],
                                  columns=feature_names)

# Hacemos las predicciones con el input del front
pred_reg = pipeline_reg.predict(data_para_predecir)
pred_clas = pipeline_clas.predict(data_para_predecir)

# Mostramos las predicciones en la app

resultado_clas = '**sí** 🌧️' if pred_clas else '**no** 🌞'
respuesta_reg = 'y' if pred_clas else 'pero'
resultado_reg  = round(float(pred_reg[0][0]), 2)

st.markdown(f'Probablemente mañana {resultado_clas} llueva {respuesta_reg} caigan {resultado_reg} mm/h de lluvia.')