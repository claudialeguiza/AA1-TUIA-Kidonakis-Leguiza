# -*- coding: utf-8 -*-
"""app.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1BNy0KUO8PXzKJ_ChcvzfkLw7A3hLPJo7

<a href="https://colab.research.google.com/github/claudialeguiza/AA1-TUIA-Kidonakis-Leguiza/blob/navegador/generar_modelos.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
"""

import streamlit as st
import pandas as pd
import joblib
from datetime import date

def cargar_archivos():
      modelo_clasif = joblib.load('clasificacion_clima.pkl')
      modelo_regresion = joblib.load('regresion_clima.pkl')

      return  modelo_clasif, modelo_regresion

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

def preparar_prediccion(df):

    df[f'RainToday_Yes'] = (df['RainToday']== 'Yes').astype(int)

    df['Date'] = pd.to_datetime(df['Date'])
    df['season'] = df['Date'].apply(asignar_estacion)

    # Crear columnas para season
    season_list = ['Spring', 'Summer','Winter' ]
    for season in season_list:
        df[f'season_{season}'] = (df['season']== season).astype(int)

    #Crear columnas para Location
    Location_list= ['Canberra','Cobar', 'Dartmoor','Melbourne','MelbourneAirport',\
                    'MountGambier','Sydney','SydneyAirport']
    for ciudad in  Location_list:
         df[f'Location_{ciudad}'] = (df['Location']== ciudad).astype(int)

    # Crear columnas para WindGustDir, WindDir9am, WindDir3pm
    wind_directions = ["SW", "S", 'SSW', 'W', 'SSE', 'E', 'SE', 'NE', 'WSW', 'WNW', 'NW', 'N', 'ESE', 'ENE']
    for var in wind_directions:
        df[f'WindGustDir_{var}'] = (df['WindGustDir'] == var).astype(int)
        df[f'WindDir9am_{var}'] = (df['WindDir9am'] == var).astype(int)
        df[f'WindDir3pm_{var}'] = (df['WindDir3pm'] == var).astype(int)

    df = df.drop(columns=['WindGustDir', 'WindDir9am', 'WindDir3pm','Date',\
                          'season', 'Location', 'RainToday'])
    return df

def interactuar_con_usuario():

    armar_prediccion = {}
    fecha = date.today()
    wind_directions = ["SW", "S", 'SSW', 'W', 'SSE', 'E', 'SE', 'NE', 'NNE',\
                       'WSW', 'WNW', 'NW', 'N', 'ESE', 'ENE']
    Location_list= ['Canberra','Cobar','Dartmoor','Melbourne','MelbourneAirport',\
                    'MountGambier','Sydney','SydneyAirport']
    cargar_prdiccion = {
       'Date': fecha,
       'Location': st.selectbox('Location',['Canberra','Cobar','Dartmoor','Melbourne','MelbourneAirport','MountGambier','Sydney','SydneyAirport']),
       'MinTemp' : st.slider ("MinTemp", min_value=-10.0, max_value=35.0, value=15.0, step=0.1 ),
       'MaxTemp': st.slider("MaxTemp", min_value=3.0, max_value=50.0, value= 30.0, step=0.1),
       'Rainfall': st.slider("Rainfall", min_value=0.0, max_value= 121.0,value=60.0, step=0.1 ),
       'Evaporation': st.slider("Evaporation", min_value=0.0, max_value=50.0,value=10.0, step=0.1),
       'Sunshine':st.slider ("Sunshine", min_value=0.0, max_value=15.0, value=6.0, step=0.1),
       'WindGustDir': st.selectbox("WindGustDir", wind_directions),
       'WindGustSpeed': st.slider("WindGustSpeed", min_value=8.0, max_value=123.0, value=30.0, step=0.1),
       'WindDir9am': st.selectbox("WindDir9am", wind_directions, key='WindDir9am'),
       'WindDir3pm': st.selectbox("WindDir3pm",wind_directions,key='WindDir3pm'),
       'WindSpeed9am': st.slider ("WindSpeed9am", min_value=0.0, max_value=70.0, value=60.0, step=0.1),
       'WindSpeed3pm':st.slider("WindSpeed3pm", min_value=0.0, max_value=80.0, value=55.0, step=0.1),
       'Humidity9am': st.slider("Humidity9am", min_value=0.0, max_value=100.0, value=45.0, step=0.1),
       'Humidity3pm': st.slider("Humidity3pm", min_value=0.0, max_value=100.0, value=79.0, step=0.1),
       'Pressure9am': st.slider("Pressure9am", min_value=950.0, max_value=1041.0, value=1020.0, step=0.1),
       'Cloud9am': st.slider("Cloud9am", min_value=0.0, max_value=10.0, value=6.0, step=0.1),
       'Cloud3pm': st.slider("Cloud3pm", min_value=0.0, max_value=10.0, value=7.0, step=0.1),
       'Temp9am': st.slider("Temp9am", min_value=-3.0, max_value=40.0, value=12.0, step=0.1),
       'Temp3pm': st.slider("Temp3pm", min_value= 2.0, max_value=50.0, value=34.0, step=0.1),
       'RainToday' : st.selectbox("RainToday",['Yes', 'No'])}

    df_prediccion =pd.DataFrame([armar_prediccion])
    return df_prediccion

st.title('Pronostico de lluvia para mañana')
st.markdown('Espere mientras cargamos la informacion')

df_prediccion = interactuar_con_usuario()
pipeline_clasificacion, pipeline_regresion = cargar_archivos()
df_prediccion = interactuar_con_usuario()
df_prediccion_filtrada = preparar_prediccion(df_prediccion)
df_prediccion_filtrada['Prediccion_lluvia']= pipeline_clasificacion.predict(df_prediccion_filtrada)[0]
df_prediccion_filtrada['Prediccion_lluvia']= df_prediccion_filtrada['Prediccion_lluvia'].astype(int)

if df_prediccion_filtrada['Prediccion_lluvia'][0] == 1:
       resultado_clas =  '**sí** 🌧️'
       df_prediccion_filtrada['Prediccion_mm']= pipeline_regresion.predict(df_prediccion_filtrada)[0]
       resultado_reg  = round(float(df_prediccion_filtrada['Prediccion_mm'][0]), 2)
else:
      df_prediccion_filtrada['Prediccion_mm'] = 0.0
      resultado_clas = '**no** 🌞'
      resultado_reg = 0

  # Mostramos las predicciones en la app
st.markdown(f'Probablemente mañana {resultado_clas} llueva , precipitaciones: {resultado_reg} mm/h de lluvia.')
st.write('Gracias por usar nuestro servicio')
st.stop()