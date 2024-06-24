import streamlit as st
import pandas as pd
import joblib
from keras.models import load_model


# Definir las funciones de carga de modelos
def cargar_modelo_regresion():
    modelo_regresion = load_model(r'C:\Users\solki\OneDrive\Documentos\AA_FINAL\AA1-TUIA-Kidonakis-Leguiza\regression_model.h5')
    return modelo_regresion

def cargar_modelo_clasificacion():
    modelo_clasif = load_model(r'C:\Users\solki\OneDrive\Documentos\AA_FINAL\AA1-TUIA-Kidonakis-Leguiza\classification_model_optimized.h5')
    return modelo_clasif

# Título de la app
st.title('Pronóstico de lluvia para mañana')

# Cargar el dataset para obtener el nombre de las columnas
df = pd.read_csv(r'C:\Users\solki\OneDrive\Documentos\AA_FINAL\AA1-TUIA-Kidonakis-Leguiza\weatherAUS.csv')

# Seleccionar solo columnas numéricas
columnas_numericas = df.select_dtypes(include=['number']).columns.tolist()

# Cargar los pipelines desde los archivos joblib
path_regresion = r'C:\Users\solki\OneDrive\Documentos\AA_FINAL\AA1-TUIA-Kidonakis-Leguiza\regresion_pipeline.joblib'
path_clasificacion = r'C:\Users\solki\OneDrive\Documentos\AA_FINAL\AA1-TUIA-Kidonakis-Leguiza\\clasificacion_pipeline.joblib'

pipeline_modelo_regresion = joblib.load(path_regresion)
pipeline_modelo_clasificacion = joblib.load(path_clasificacion)

# Crear sliders para las variables numéricas
features = []
for columna in columnas_numericas:
    min_value = float(round(df[columna].min(), 2))  # Convertir min_value a float y redondear a 2 decimales
    max_value = float(round(df[columna].max(), 2))  # Convertir max_value a float y redondear a 2 decimales
    step = (max_value - min_value) / 100.0  # Asegurar que step sea float

    # Ajustar el valor predeterminado usando round después de seleccionar un valor
    default_value = st.slider(columna, min_value, max_value, round(df[columna].mean(), 2), step=step)
    features.append(default_value)

# Mapear la opción booleana a un texto y agregarla para la predicción junto a las variables numéricas
raintoday_option_mapping = {'Sí': 1, 'No': 0}
raintoday_option = st.selectbox('¿Hoy llovió?', list(raintoday_option_mapping.keys()))

all_features = features + [raintoday_option_mapping[raintoday_option]]

# Crear el DataFrame correctamente
data_para_predecir = pd.DataFrame([all_features], columns=columnas_numericas + ['raintoday_option'])

# Hacer predicciones con el input del front
pred_clas = None
try:
    pred_clas = pipeline_modelo_clasificacion.predict(data_para_predecir)
except Exception as e:
    st.error(f"Error al hacer predicciones de clasificación: {e}")

# Mostrar las predicciones en la app
if pred_clas is not None:
    resultado_clas = '**sí** 🌧️' if pred_clas else '**no** 🌞'
    respuesta_reg = 'y' if pred_clas else 'pero'
    resultado_reg = round(float(pred_clas[0]), 2)  # Asegurar que se accede al valor correcto del arreglo de predicciones
    
    st.markdown(f'Probablemente mañana {resultado_clas} llueva {respuesta_reg} caigan {resultado_reg} mm/h de lluvia.')
else:
    st.error("No se pudo realizar la predicción de clasificación.")