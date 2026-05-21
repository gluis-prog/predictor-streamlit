import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ── Configuración de página ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Predictor de Valor de Vivienda",
    page_icon="🏠",
    layout="centered",
)

# ── Estilos personalizados ───────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.main { background-color: #f7f5f0; }
.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 780px; }
h1 {
    font-family: 'DM Serif Display', serif !important;
    font-size: 2.4rem !important;
    color: #1a1a2e !important;
    line-height: 1.2 !important;
}
.subtitulo { color: #6b7280; font-size: 1rem; margin-top: -0.5rem; margin-bottom: 2rem; }
.seccion {
    background: white; border-radius: 16px; padding: 1.5rem 2rem;
    margin-bottom: 1.5rem; border: 1px solid #e5e7eb;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}
.seccion-titulo {
    font-size: 0.75rem; font-weight: 600; letter-spacing: 0.1em;
    text-transform: uppercase; color: #9ca3af; margin-bottom: 1rem;
}
.resultado-box {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border-radius: 16px; padding: 2rem; text-align: center; margin-top: 1rem;
}
.resultado-label { color: #9ca3af; font-size: 0.85rem; letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 0.5rem; }
.resultado-valor { color: #ffffff; font-family: 'DM Serif Display', serif; font-size: 3rem; line-height: 1; }
.resultado-sub { color: #6ee7b7; font-size: 0.85rem; margin-top: 0.5rem; }
.advertencia {
    background: #fffbeb; border: 1px solid #fde68a; border-radius: 10px;
    padding: 0.75rem 1rem; font-size: 0.85rem; color: #92400e; margin-top: 1rem;
}
.stButton > button {
    background-color: #1a1a2e !important; color: white !important;
    border: none !important; border-radius: 10px !important;
    padding: 0.6rem 2rem !important; font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important; font-size: 0.95rem !important; width: 100%;
}
label { font-size: 0.88rem !important; font-weight: 500 !important; color: #374151 !important; }
</style>
""", unsafe_allow_html=True)

# ── Encabezado ───────────────────────────────────────────────────────────────
st.markdown("<h1>🏠 Predictor de Valor de Vivienda</h1>", unsafe_allow_html=True)
st.markdown('<p class="subtitulo">Ingresa las características de la propiedad para estimar su valor de mercado en dólares (USD).</p>', unsafe_allow_html=True)

# ── Cargar modelo y escalador ────────────────────────────────────────────────
@st.cache_resource
def cargar_artefactos():
    try:
        modelo = joblib.load("modelo_regresion.pkl")
        scaler = joblib.load("feature_scaler.pkl")
        return modelo, scaler
    except FileNotFoundError:
        return None, None

modelo, scaler = cargar_artefactos()

if modelo is None or scaler is None:
    st.error("⚠️ No se encontraron **modelo_regresion.pkl** y/o **feature_scaler.pkl**. Asegúrate de que ambos estén en la misma carpeta que app.py.")
    st.stop()

# ── Orden exacto de features (debe coincidir con el entrenamiento) ────────────
FEATURE_COLUMNS = [
    'longitude', 'latitude', 'housing_median_age',
    'total_rooms', 'total_bedrooms', 'population',
    'households', 'median_income',
    'rooms_per_household', 'bedrooms_per_room', 'population_per_household'
]

# ── Formulario ───────────────────────────────────────────────────────────────

# Longitud y latitud fijas en su media — no se muestran al usuario
longitude = -119.556526
latitude  =   35.617721

st.markdown('<div class="seccion"><div class="seccion-titulo">🏠 Características de la vivienda</div>', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
with col1:
    total_rooms    = st.number_input("Habitaciones", value=4, min_value=1, max_value=500, step=1,
                                      help="Total de habitaciones en el bloque")
with col2:
    total_bedrooms = st.number_input("Dormitorios", value=2, min_value=1, max_value=300, step=1,
                                      help="Total de dormitorios en el bloque")
with col3:
    population     = st.number_input("Personas", value=3, min_value=1, max_value=2000, step=1,
                                      help="Población del bloque")
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="seccion"><div class="seccion-titulo">🏘️ Datos del bloque residencial</div>', unsafe_allow_html=True)
col4, col5 = st.columns(2)
with col4:
    households         = st.number_input("Número de hogares", value=499, min_value=1, max_value=7000, step=1)
with col5:
    housing_median_age = st.number_input("Antigüedad mediana (años)", value=29, min_value=1, max_value=52, step=1)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="seccion"><div class="seccion-titulo">💵 Ingreso del vecindario</div>', unsafe_allow_html=True)
median_income = st.slider("Ingreso mediano del hogar (en decenas de miles USD)",
                           min_value=0.5, max_value=15.0, value=3.87, step=0.01,
                           help="Ej: 3.87 = ~$38,700 USD anuales")
st.caption(f"Equivale a aproximadamente **${median_income * 10000:,.0f} USD** anuales por hogar")
st.markdown('</div>', unsafe_allow_html=True)

# ── Predicción ───────────────────────────────────────────────────────────────
if st.button("Estimar valor de la vivienda →"):
    if households == 0 or total_rooms == 0:
        st.error("El número de hogares y de habitaciones deben ser mayores a 0.")
    else:
        # Variables derivadas — calculadas igual que en el entrenamiento
        rooms_per_household      = total_rooms    / households
        bedrooms_per_room        = total_bedrooms / total_rooms
        population_per_household = population     / households

        entrada = pd.DataFrame([[
            longitude, latitude, housing_median_age,
            total_rooms, total_bedrooms, population,
            households, median_income,
            rooms_per_household, bedrooms_per_room, population_per_household
        ]], columns=FEATURE_COLUMNS)

        try:
            entrada_scaled = scaler.transform(entrada)
            prediccion = max(0, modelo.predict(entrada_scaled)[0])

            st.markdown(f"""
            <div class="resultado-box">
                <div class="resultado-label">Valor estimado de la vivienda</div>
                <div class="resultado-valor">${prediccion:,.0f}</div>
                <div class="resultado-sub">USD · Gradient Boosting · R² 0.81</div>
            </div>
            """, unsafe_allow_html=True)

            with st.expander("Ver variables calculadas automáticamente"):
                st.write(f"- Habitaciones por hogar: **{rooms_per_household:.2f}**")
                st.write(f"- Dormitorios por habitación: **{bedrooms_per_room:.2f}**")
                st.write(f"- Personas por hogar: **{population_per_household:.2f}**")

            st.markdown("""
            <div class="advertencia">
                ⚠️ Esta predicción es una estimación estadística basada en datos históricos de California.
                No reemplaza una tasación profesional.
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error al realizar la predicción: {e}")

# ── Pie ───────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Gradient Boosting Regressor · Dataset California Housing · Desarrollado con Streamlit")
