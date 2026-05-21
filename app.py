import streamlit as st
import joblib
import pandas as pd
import numpy as np

# 1. Configuración de la página
st.set_page_config(page_title="Predictor de Viviendas", page_icon="🏠", layout="centered")

st.title("🏠 Predictor de Precios de Vivienda")
st.write("Introduce los parámetros de la zona y la vivienda para calcular el precio estimado.")

# 2. Cargar el modelo y el scaler de forma segura
@st.cache_resource
def cargar_modelos():
    modelo = joblib.load("modelo_regresion.pkl")
    scaler = joblib.load("feature_scaler.pkl")
    return modelo, scaler

try:
    modelo, scaler = cargar_modelos()
except Exception as e:
    st.error(f"Error al cargar los archivos del modelo: {e}")

# Definir las columnas exactas que espera el scaler
FEATURE_COLUMNS = [
    'longitude', 'latitude', 'housing_median_age',
    'total_rooms', 'total_bedrooms', 'population',
    'households', 'median_income',
    'rooms_per_household', 'bedrooms_per_room', 'population_per_household'
]

# 3. Crear el formulario en la interfaz gráfica
with st.form("formulario_prediccion"):
    st.subheader("Datos de la Vivienda y Zona")
    
    housing_median_age = st.number_input("Edad Media de la Vivienda (años)", min_value=1, max_value=100, value=15)
    total_rooms = st.number_input("Total de Habitaciones en la manzana", min_value=1, value=5600)
    total_bedrooms = st.number_input("Total de Dormitorios en la manzana", min_value=1, value=1000)
    population = st.number_input("Población en la manzana", min_value=1, value=2500)
    households = st.number_input("Total de Hogares / Familias", min_value=1, value=950)
    median_income = st.number_input("Ingreso Medio de la zona (en decenas de miles)", min_value=0.0, step=0.1, value=4.5)
    
    # Botón para enviar el formulario
    boton_predecir = st.form_submit_button("Calcular Predicción")

# 4. Lógica al presionar el botón
if boton_predecir:
    try:
        # Cálculos de las variables combinadas (tal como lo hacías en FastAPI)
        rph = total_rooms / households
        bpr = total_bedrooms / total_rooms
        pph = population / households

        # Valores neutros para longitud y latitud (valores medios fijados)
        entrada = pd.DataFrame([[\
            -119.556526, 35.6177206,
            housing_median_age,
            total_rooms, total_bedrooms, population, households,
            median_income,
            rph, bpr, pph
        ]], columns=FEATURE_COLUMNS)

        # Escalar los datos y predecir
        entrada_scaled = scaler.transform(entrada)
        prediccion = modelo.predict(entrada_scaled)[0]

        # Mostrar el resultado con un diseño llamativo
        st.success(f"### 💰 El precio estimado de la vivienda es: **${prediccion:,.2f}**")

    except ZeroDivisionError:
        st.error("Por favor, asegúrate de que el número de hogares o habitaciones sea mayor a 0.")
    except Exception as e:
        st.error(f"Ocurrió un error al procesar la predicción: {e}")