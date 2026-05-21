import streamlit as st
import joblib
import pandas as pd
import numpy as np

# 1. Configuración de la página
st.set_page_config(page_title="Predictor de Valor de Vivienda", page_icon="🏠", layout="centered")

# Inyección de CSS corregida (Se eliminó el error del texto expuesto)
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
    .custom-card {
        background: white !important;
        border: 1px solid #e5e7eb !important;
        border-radius: 14px !important;
        padding: 1.25rem 1.5rem !important;
        margin-bottom: 1rem !important;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04) !important;
    }
    
    .card-label {
        font-size: 11px !important;
        font-weight: 600 !important;
        letter-spacing: 0.08em !important;
        text-transform: uppercase !important;
        color: #9ca3af !important;
        margin-bottom: 1rem !important;
        display: flex !important;
        align-items: center !important;
        gap: 6px !important;
    }

    /* Diseño del contador estilo plantilla */
    .counter-display {
        background: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        text-align: center;
        font-size: 20px;
        font-weight: 600;
        color: #1a1a2e;
        line-height: 46px;
        height: 46px;
        width: 100%;
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

# Badges
col_b1, col_b2, _ = st.columns([2.5, 2, 4])
with col_b1:
    st.markdown('<span style="background: #dcfce7; color: #166534; border: 1px solid #bbf7d0; font-size: 11px; font-weight: 500; padding: 4px 10px; border-radius: 6px;">📈 Gradient Boosting · R² 0.81</span>', unsafe_allow_html=True)
with col_b2:
    st.markdown('<span style="background: #dbeafe; color: #1e40af; border: 1px solid #bfdbfe; font-size: 11px; font-weight: 500; padding: 4px 10px; border-radius: 6px;">☁️ API en vivo · Railway</span>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# 2. Cargar el modelo y el scaler
@st.cache_resource
def cargar_modelos():
    modelo = joblib.load("modelo_regresion.pkl")
    scaler = joblib.load("feature_scaler.pkl")
    return modelo, scaler

try:
    modelo, scaler = cargar_modelos()
except Exception as e:
    st.error(f"Error al cargar los archivos del modelo: {e}")

FEATURE_COLUMNS = [
    'longitude', 'latitude', 'housing_median_age',
    'total_rooms', 'total_bedrooms', 'population',
    'households', 'median_income',
    'rooms_per_household', 'bedrooms_per_room', 'population_per_household'
]

# Inicializar estados para los botones +/- si no existen
if 'rooms' not in st.session_state: st.session_state.rooms = 4
if 'bedrooms' not in st.session_state: st.session_state.bedrooms = 2
if 'population' not in st.session_state: st.session_state.population = 3

# --- TARJETA 1: CARACTERÍSTICAS DE LA VIVIENDA ---
st.markdown('<div class="custom-card"><div class="card-label">🏠 Características de la vivienda</div>', unsafe_allow_html=True)

g1_c1, g1_c2, g1_c3 = st.columns(3)

with g1_c1:
    st.markdown('<label style="font-size: 13px; color: #4b5563; font-weight: 500;">Habitaciones</label>', unsafe_allow_html=True)
    btn_col1, display_col1, btn_col2 = st.columns([1, 2, 1])
    with btn_col1: 
        if st.button("−", key="btn_r_sub"): st.session_state.rooms = max(1, st.session_state.rooms - 1)
    with display_col1: 
        st.markdown(f'<div class="counter-display">{st.session_state.rooms}</div>', unsafe_allow_html=True)
    with btn_col2: 
        if st.button("+", key="btn_r_add"): st.session_state.rooms += 1
    st.markdown('<div style="font-size: 11px; color: #9ca3af; text-align: center; margin-top:4px;">en el bloque</div>', unsafe_allow_html=True)

with g1_c2:
    st.markdown('<label style="font-size: 13px; color: #4b5563; font-weight: 500;">Dormitorios</label>', unsafe_allow_html=True)
    btn_col3, display_col2, btn_col4 = st.columns([1, 2, 1])
    with btn_col3: 
        if st.button("−", key="btn_b_sub"): st.session_state.bedrooms = max(1, st.session_state.bedrooms - 1)
    with display_col2: 
        st.markdown(f'<div class="counter-display">{st.session_state.bedrooms}</div>', unsafe_allow_html=True)
    with btn_col4: 
        if st.button("+", key="btn_b_add"): st.session_state.bedrooms += 1
    st.markdown('<div style="font-size: 11px; color: #9ca3af; text-align: center; margin-top:4px;">en el bloque</div>', unsafe_allow_html=True)

with g1_c3:
    st.markdown('<label style="font-size: 13px; color: #4b5563; font-weight: 500;">Personas</label>', unsafe_allow_html=True)
    btn_col5, display_col3, btn_col6 = st.columns([1, 2, 1])
    with btn_col5: 
        if st.button("−", key="btn_p_sub"): st.session_state.population = max(1, st.session_state.population - 1)
    with display_col3: 
        st.markdown(f'<div class="counter-display">{st.session_state.population}</div>', unsafe_allow_html=True)
    with btn_col6: 
        if st.button("+", key="btn_p_add"): st.session_state.population += 1
    st.markdown('<div style="font-size: 11px; color: #9ca3af; text-align: center; margin-top:4px;">en el bloque</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)


# --- TARJETA 2: DATOS DEL BLOQUE RESIDENCIAL ---
st.markdown('<div class="custom-card"><div class="card-label">營 Datos del bloque residencial</div>', unsafe_allow_html=True)
g2_c1, g2_c2 = st.columns(2)
with g2_c1:
    households_input = st.number_input("Número de hogares", min_value=1, max_value=7000, value=499, step=1, label_visibility="visible")
with g2_c2:
    age_input = st.number_input("Antigüedad mediana (años)", min_value=1, max_value=52, value=29, step=1, label_visibility="visible")
st.markdown('</div>', unsafe_allow_html=True)


# --- TARJETA 3: INGRESO DEL VECINDARIO ---
st.markdown('<div class="custom-card"><div class="card-label">馃挜 Ingreso del vecindario</div>', unsafe_allow_html=True)
st.markdown('<label style="font-size: 13px; color: #4b5563; font-weight: 500;">Ingreso mediano del hogar <span style="color:#9ca3af;font-weight:400">(decenas de miles USD)</span></label>', unsafe_allow_html=True)
income_input = st.slider("", min_value=0.5, max_value=15.0, value=3.87, step=0.01, label_visibility="collapsed")
equiv_usd = income_input * 10000
st.markdown(f'<p style="font-size: 12px; color: #9ca3af; margin-top:5px;">Equivale a aproximadamente ${equiv_usd:,.0f} USD anuales por hogar</p>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)


# Botón de ejecución fuera de formulario para interactuar con los estados correctamente
boton_ejecutar = st.button("馃 Estimar valor de la vivienda")

if boton_ejecutar:
    try:
        # Recuperar valores de los contadores interactivos
        rooms_input = st.session_state.rooms
        bedrooms_input = st.session_state.bedrooms
        population_input = st.session_state.population

        rph = rooms_input / households_input
        bpr = bedrooms_input / rooms_input
        pph = population_input / households_input

        entrada = pd.DataFrame([[\
            -119.556526, 35.6177206,
            age_input,
            rooms_input, bedrooms_input, population_input, households_input,
            income_input,
            rph, bpr, pph
        ]], columns=FEATURE_COLUMNS)

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
