import streamlit as st
import joblib
import pandas as pd
import numpy as np

# 1. Configuración de la página con el estilo exacto de tu plantilla
st.set_page_config(page_title="Predictor de Valor de Vivienda", page_icon="🏠", layout="centered")

# Inyectamos el CSS personalizado de tu plantilla HTML para cambiar fuentes, tarjetas y colores
st.markdown("""
    <style>
    /* Estilos globales */
    .stApp {
        background-color: #f7f5f0 !important;
        color: #1a1a2e !important;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    /* Títulos e información */
    h1 {
        font-size: 26px !important;
        font-weight: 600 !important;
        color: #1a1a2e !important;
        margin-bottom: 6px !important;
    }
    
    /* Estilo de las Tarjetas (Cards) */
    div[data-testid="stForm"] {
        background: white !important;
        border: 1px solid #e5e7eb !important;
        border-radius: 14px !important;
        padding: 1.5rem !important;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04) !important;
    }
    
    /* Botón Principal estilo Premium */
    div.stButton > button {
        width: 100% !important;
        height: 50px !important;
        background: #1a1a2e !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        font-size: 15px !important;
        font-weight: 600 !important;
        cursor: pointer !important;
        transition: opacity 0.15s !important;
    }
    div.stButton > button:hover {
        opacity: 0.85 !important;
        color: white !important;
    }

    /* Caja de resultado de la predicción */
    .result-box {
        background: white; 
        border: 1px solid #e5e7eb;
        border-radius: 14px; 
        padding: 1.5rem;
        text-align: center; 
        margin-top: 1rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    }
    .result-label {
        font-size: 11px; font-weight: 600;
        letter-spacing: 0.1em; text-transform: uppercase;
        color: #9ca3af; margin-bottom: 8px;
    }
    .result-value {
        font-size: 46px; font-weight: 700;
        color: #1a1a2e; margin-bottom: 4px;
    }
    .result-sub { font-size: 12px; color: #6b7280; margin-bottom: 1rem; }
    
    /* Advertencia */
    .warning-box {
        font-size: 12px; color: #92400e;
        background: #fffbeb; border: 1px solid #fde68a;
        border-radius: 10px; padding: 10px 14px;
        margin-top: 12px; line-height: 1.5; text-align: left;
    }
    </style>
""", unsafe_allow_html=True)

# Encabezado (Hero section)
st.title("🏠 Predictor de valor de vivienda")
st.caption("Ingresa las características de la propiedad para estimar su valor de mercado en dólares (USD).")

# Badges (Etiquetas informativas)
col_b1, col_b2, _ = st.columns([2.5, 2, 4])
with col_b1:
    st.markdown('<span style="background: #dcfce7; color: #166534; border: 1px solid #bbf7d0; font-size: 11px; font-weight: 500; padding: 4px 10px; border-radius: 6px;">📈 Gradient Boosting · R² 0.81</span>', unsafe_allow_html=True)
with col_b2:
    st.markdown('<span style="background: #dbeafe; color: #1e40af; border: 1px solid #bfdbfe; font-size: 11px; font-weight: 500; padding: 4px 10px; border-radius: 6px;">☁️ API Local · Interna</span>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# 2. Cargar el modelo y el scaler de manera local en Streamlit
@st.cache_resource
def cargar_modelos():
    modelo = joblib.load("modelo_regresion.pkl")
    scaler = joblib.load("feature_scaler.pkl")
    return modelo, scaler

try:
    modelo, scaler = cargar_modelos()
except Exception as e:
    st.error(f"Error al cargar los archivos del modelo: {e}")

# Columnas exactas que el modelo necesita
FEATURE_COLUMNS = [
    'longitude', 'latitude', 'housing_median_age',
    'total_rooms', 'total_bedrooms', 'population',
    'households', 'median_income',
    'rooms_per_household', 'bedrooms_per_room', 'population_per_household'
]

# 3. Formulario Único unificando los componentes visuales
with st.form("main_form", clear_on_submit=False):
    
    st.markdown('<div style="font-size: 11px; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: #9ca3af; margin-bottom: 0.5rem;">🏠 Características de la vivienda</div>', unsafe_allow_html=True)
    
    # Grid 1: Habitaciones, Dormitorios y Personas (usando selectores numéricos nativos con pasos de 1)
    g1_c1, g1_c2, g1_c3 = st.columns(3)
    with g1_c1:
        rooms_input = st.number_input("Habitaciones", min_value=1, max_value=10000, value=4, step=1, help="En el bloque")
    with g1_c2:
        bedrooms_input = st.number_input("Dormitorios", min_value=1, max_value=5000, value=2, step=1, help="En el bloque")
    with g1_c3:
        population_input = st.number_input("Personas", min_value=1, max_value=50000, value=3, step=1, help="En el bloque")
        
    st.markdown("<hr style='border-top: 1px solid #f3f4f6; margin: 1rem 0;'>", unsafe_allow_html=True)
    st.markdown('<div style="font-size: 11px; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: #9ca3af; margin-bottom: 0.5rem;">🏘️ Datos del bloque residencial</div>', unsafe_allow_html=True)
    
    # Grid 2: Número de hogares y Antigüedad mediana
    g2_c1, g2_c2 = st.columns(2)
    with g2_c1:
        households_input = st.number_input("Número de hogares", min_value=1, max_value=7000, value=499, step=1)
    with g2_c2:
        age_input = st.number_input("Antigüedad mediana (años)", min_value=1, max_value=52, value=29, step=1)

    st.markdown("<hr style='border-top: 1px solid #f3f4f6; margin: 1rem 0;'>", unsafe_allow_html=True)
    st.markdown('<div style="font-size: 11px; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: #9ca3af; margin-bottom: 0.5rem;">💵 Ingreso del vecindario</div>', unsafe_allow_html=True)
    
    # Slider interactivo para el Ingreso Mediano
    income_input = st.slider("Ingreso mediano del hogar (decenas de miles USD)", min_value=0.5, max_value=15.0, value=3.87, step=0.01)
    equiv_usd = income_input * 10000
    st.caption(f"Equivale a aproximadamente ${equiv_usd:,.0f} USD anuales por hogar")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Botón para ejecutar la predicción
    boton_ejecutar = st.form_submit_button("🧮 Estimar valor de la vivienda")

# 4. Lógica de Predicción y Visualización de Resultados
if boton_ejecutar:
    try:
        # Calcular ratios requeridos por el modelo matemático
        rph = rooms_input / households_input
        bpr = bedrooms_input / rooms_input
        pph = population_input / households_input

        # DataFrame con valores base para Longitud y Latitud neutrales
        entrada = pd.DataFrame([[\
            -119.556526, 35.6177206,
            age_input,
            rooms_input, bedrooms_input, population_input, households_input,
            income_input,
            rph, bpr, pph
        ]], columns=FEATURE_COLUMNS)

        # Escalar datos y ejecutar el modelo de regresión (.pkl)
        entrada_scaled = scaler.transform(entrada)
        prediccion = modelo.predict(entrada_scaled)[0]

        # Caja de Resultados idéntica a tu plantilla
        st.markdown(f"""
            <div class="result-box">
                <div class="result-label">Valor estimado de la vivienda</div>
                <div class="result-value">${prediccion:,.2f}</div>
                <div class="result-sub">USD · Gradient Boosting · Predicción real del modelo</div>
                <hr style="border: none; border-top: 1px solid #f3f4f6; margin: 1rem 0;">
                
                <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 8px; margin: 1rem 0;">
                    <div style="background: #f9fafb; border: 1px solid #f3f4f6; border-radius: 10px; padding: 10px 12px;">
                        <div style="font-size: 11px; color: #9ca3af; margin-bottom: 4px;">Precisión (R²)</div>
                        <div style="font-size: 14px; font-weight: 600; color: #1a1a2e;">0.81 / 1.00</div>
                    </div>
                    <div style="background: #f9fafb; border: 1px solid #f3f4f6; border-radius: 10px; padding: 10px 12px;">
                        <div style="font-size: 11px; color: #9ca3af; margin-bottom: 4px;">Error promedio</div>
                        <div style="font-size: 14px; font-weight: 600; color: #1a1a2e;">$32,991</div>
                    </div>
                    <div style="background: #f9fafb; border: 1px solid #f3f4f6; border-radius: 10px; padding: 10px 12px;">
                        <div style="font-size: 11px; color: #9ca3af; margin-bottom: 4px;">Error cuadrático</div>
                        <div style="font-size: 14px; font-weight: 600; color: #1a1a2e;">$51,166</div>
                    </div>
                </div>
                
                <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 8px; margin-top: 8px;">
                    <div style="background: #f9fafb; border: 1px solid #f3f4f6; border-radius: 10px; padding: 10px 12px;">
                        <div style="font-size: 11px; color: #9ca3af; margin-bottom: 4px;">Hab. por Hogar</div>
                        <div style="font-size: 14px; font-weight: 600; color: #1a1a2e;">{rph:.2f}</div>
                    </div>
                    <div style="background: #f9fafb; border: 1px solid #f3f4f6; border-radius: 10px; padding: 10px 12px;">
                        <div style="font-size: 11px; color: #9ca3af; margin-bottom: 4px;">Dorm. por Hab.</div>
                        <div style="font-size: 14px; font-weight: 600; color: #1a1a2e;">{bpr:.2f}</div>
                    </div>
                    <div style="background: #f9fafb; border: 1px solid #f3f4f6; border-radius: 10px; padding: 10px 12px;">
                        <div style="font-size: 11px; color: #9ca3af; margin-bottom: 4px;">Pers. por Hogar</div>
                        <div style="font-size: 14px; font-weight: 600; color: #1a1a2e;">{pph:.2f}</div>
                    </div>
                </div>
                
                <div class="warning-box">
                    ⚠️ Estimación estadística basada en datos históricos de California. No reemplaza una tasación profesional.
                </div>
            </div>
        """, unsafe_allow_html=True)
        
    except ZeroDivisionError:
        st.error("❌ Por favor, asegúrate de que el número de hogares o habitaciones sea mayor a 0.")
    except Exception as e:
        st.error(f"⚠️ Error inesperado al calcular la predicción: {e}")

# Pie de página
st.markdown("<br><hr style='border-top: 1px solid #e5e7eb;'><div style='text-align: center; font-size: 12px; color: #9ca3af;'>Modelo de Regresión · Dataset California Housing · Desarrollado con FastAPI + Railway</div>", unsafe_allow_html=True)
    
