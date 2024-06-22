import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, FunctionTransformer
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
import keras
from tensorflow.keras.losses import MeanSquaredError

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

import warnings
warnings.simplefilter('ignore')

# Cargar datos
datos = pd.read_csv(r'C:\Users\solki\OneDrive\Documentos\AA_Final\AA1-TUIA-Kidonakis-Leguiza\weatherAUS.csv', delimiter=",")

# Filtrar datos por ubicaciones específicas
df = datos[datos.Location.isin(('Sydney', 'SydneyAirport', 'Melbourne', 'MelbourneAirport',
                                'Canberra', 'Adelaide', 'MountGambier', 'Cobar', 'Dartmoor'))]

# Definir funciones de preprocesamiento
def preprocesamiento(data):
    data.info()
    data.isna().sum()

    # Definir columnas con valores nulos
    columnas_con_nulos = ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine',
                          'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am',
                          'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am',
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

def estandarizar_df(data):
    scaler = RobustScaler()
    data_scaled = scaler.fit_transform(data)
    return data_scaled

def truncar_dividir_df(data):
    data = data.sort_values(["Date"])
    fecha_especifica = '2009-01-01'
    data_filtrada = data[data['Date'] >= fecha_especifica]

    data_filtrada.reset_index(drop=True, inplace=True)  # Resetea el índice y no crea uno nuevo
    data_train = data_filtrada.iloc[:21658]

    return data_train

pipeline_prepara_datos = Pipeline([
    ('preproceso', FunctionTransformer(preprocesamiento, validate=False)),
    ('season', FunctionTransformer(crear_columna_season, validate=False)),
    ('codificar', FunctionTransformer(codificar_variables, validate=False))
])

# Obtener datos de entrenamiento
df_procesado = pipeline_prepara_datos.fit_transform(df)

pipeline_train_split = Pipeline([
    ('split', FunctionTransformer(truncar_dividir_df, validate=False)),
    ('estandarizar', FunctionTransformer(estandarizar_df, validate=False)),
])

pipeline_train_split_clas = Pipeline([
    ('split', FunctionTransformer(truncar_dividir_df, validate=False)),
    ('estandarizar_clas', FunctionTransformer(estandarizar_df, validate=False)),
])

# Cargar modelos pre-entrenados
ruta_modelo_regresion = r'C:\Users\solki\OneDrive\Documentos\AA_Final\AA1-TUIA-Kidonakis-Leguiza\regression_model.h5'
regression_model = load_model(ruta_modelo_regresion, custom_objects={'MeanSquaredError': MeanSquaredError})
ruta_modelo_classification = r'C:\Users\solki\OneDrive\Documentos\AA_Final\AA1-TUIA-Kidonakis-Leguiza\classification_model_optimized.h5'
classification_model = load_model(ruta_modelo_classification)s

# Preparar datos de regresión y clasificación
X_train_regresion = pipeline_train_split.fit_transform(df_procesado)
X_smote_train_clas = pipeline_train_split_clas.fit_transform(df_procesado)

# Entrenar modelos cargados 
regression_model.fit(X_train_regresion, y_train_regresion, epochs=100, batch_size=32)
classification_model.fit(X_smote_train_clas, y_smote_train_clas, epochs=100, batch_size=32)

# Guardar modelos 
regression_model.save('regression_model.h5')
classification_model.save('classification_model_optimized.h5')
