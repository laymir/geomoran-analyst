"""
GeoMoran Analyst — Aplicación Web con Streamlit (Diseño Profesional)
"""

import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
from streamlit_folium import st_folium
import plotly.graph_objects as go
import plotly.express as px
import libpysal
from libpysal.weights import Queen, Rook, KNN
from esda.moran import Moran, Moran_Local
import io
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
st.set_page_config(
    page_title="GeoMoran Analyst",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# CSS PROFESIONAL
# =============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ---- BASE ---- */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.stApp {
    background: #0b0f1a;
    color: #e2e8f0;
}

/* ---- SIDEBAR ---- */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1629 0%, #111827 100%);
    border-right: 1px solid #1e2d4a;
}
[data-testid="stSidebar"] * {
    color: #cbd5e1 !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #f1f5f9 !important;
    font-weight: 600;
}

/* ---- HERO HEADER ---- */
.hero-header {
    background: linear-gradient(135deg, #0f2044 0%, #1a3a6b 50%, #0f2044 100%);
    border: 1px solid #1e3a5f;
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.hero-header::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(56,189,248,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-size: 2rem;
    font-weight: 700;
    color: #f8fafc;
    margin: 0 0 0.4rem 0;
    letter-spacing: -0.5px;
}
.hero-title span {
    background: linear-gradient(90deg, #38bdf8, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-subtitle {
    color: #94a3b8;
    font-size: 0.95rem;
    margin: 0;
    font-weight: 400;
}
.hero-badges {
    display: flex;
    gap: 8px;
    margin-top: 1rem;
    flex-wrap: wrap;
}
.badge {
    background: rgba(56,189,248,0.1);
    border: 1px solid rgba(56,189,248,0.25);
    color: #38bdf8;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 500;
    font-family: 'JetBrains Mono', monospace;
}

/* ---- METRIC CARDS ---- */
.metrics-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin: 1.5rem 0;
}
.metric-card {
    background: linear-gradient(135deg, #111827 0%, #1a2235 100%);
    border: 1px solid #1e2d4a;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: #38bdf8; }
.metric-card::after {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 3px; height: 100%;
    border-radius: 3px 0 0 3px;
}
.metric-card.blue::after  { background: #38bdf8; }
.metric-card.purple::after { background: #818cf8; }
.metric-card.green::after  { background: #34d399; }
.metric-card.amber::after  { background: #fbbf24; }
.metric-label {
    font-size: 0.72rem;
    font-weight: 600;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 0.4rem;
}
.metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: #f8fafc;
    font-family: 'JetBrains Mono', monospace;
    line-height: 1;
}
.metric-sub {
    font-size: 0.75rem;
    color: #64748b;
    margin-top: 0.3rem;
}

/* ---- INTERPRETATION BANNER ---- */
.interp-positive {
    background: linear-gradient(135deg, rgba(52,211,153,0.08), rgba(52,211,153,0.03));
    border: 1px solid rgba(52,211,153,0.3);
    border-left: 4px solid #34d399;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin: 1rem 0;
}
.interp-negative {
    background: linear-gradient(135deg, rgba(251,191,36,0.08), rgba(251,191,36,0.03));
    border: 1px solid rgba(251,191,36,0.3);
    border-left: 4px solid #fbbf24;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin: 1rem 0;
}
.interp-random {
    background: linear-gradient(135deg, rgba(100,116,139,0.12), rgba(100,116,139,0.04));
    border: 1px solid rgba(100,116,139,0.3);
    border-left: 4px solid #64748b;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin: 1rem 0;
}
.interp-title {
    font-weight: 600;
    font-size: 0.85rem;
    margin-bottom: 0.3rem;
}
.interp-text {
    font-size: 0.85rem;
    color: #94a3b8;
    line-height: 1.5;
}

/* ---- SECTION HEADER ---- */
.section-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 1.8rem 0 1rem 0;
    padding-bottom: 0.6rem;
    border-bottom: 1px solid #1e2d4a;
}
.section-header h3 {
    font-size: 1rem;
    font-weight: 600;
    color: #e2e8f0;
    margin: 0;
}
.section-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #38bdf8;
    flex-shrink: 0;
}

/* ---- CLUSTER LEGEND ---- */
.cluster-legend {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin: 1rem 0;
}
.cluster-badge {
    display: flex;
    align-items: center;
    gap: 6px;
    background: #111827;
    border: 1px solid #1e2d4a;
    border-radius: 8px;
    padding: 6px 12px;
    font-size: 0.78rem;
    font-weight: 500;
    color: #cbd5e1;
}
.cluster-dot {
    width: 10px; height: 10px;
    border-radius: 3px;
    flex-shrink: 0;
}

/* ---- DATA TABLE ---- */
.stDataFrame {
    border: 1px solid #1e2d4a !important;
    border-radius: 10px !important;
    overflow: hidden;
}

/* ---- TABS ---- */
.stTabs [data-baseweb="tab-list"] {
    background: #111827;
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
    border: 1px solid #1e2d4a;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: #64748b;
    font-size: 0.82rem;
    font-weight: 500;
    padding: 6px 16px;
}
.stTabs [aria-selected="true"] {
    background: #1e3a5f !important;
    color: #38bdf8 !important;
}

/* ---- BUTTONS ---- */
.stButton > button {
    background: linear-gradient(135deg, #1d4ed8, #1e3a8a) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    padding: 0.6rem 1.2rem !important;
    transition: all 0.2s !important;
    width: 100%;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 15px rgba(37,99,235,0.3) !important;
}
.stDownloadButton > button {
    background: linear-gradient(135deg, #065f46, #064e3b) !important;
    color: #6ee7b7 !important;
    border: 1px solid #065f46 !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    width: 100%;
}

/* ---- FILE UPLOADER ---- */
[data-testid="stFileUploader"] {
    background: #111827;
    border: 1px dashed #1e3a5f;
    border-radius: 10px;
    padding: 0.5rem;
}

/* ---- SELECT / SLIDER ---- */
[data-testid="stSelectbox"] > div,
[data-testid="stSelectSlider"] > div {
    background: #111827 !important;
    border-color: #1e2d4a !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
}
.stSlider [data-testid="stTickBar"] { color: #64748b; }

/* ---- INFO BOX ---- */
.info-card {
    background: #111827;
    border: 1px solid #1e2d4a;
    border-radius: 12px;
    padding: 1.4rem;
    margin: 1rem 0;
}
.info-card h4 {
    color: #f1f5f9;
    font-size: 0.9rem;
    font-weight: 600;
    margin: 0 0 0.8rem 0;
}
.moran-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.82rem;
}
.moran-table th {
    background: #1e2d4a;
    color: #94a3b8;
    padding: 8px 12px;
    text-align: left;
    font-weight: 600;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.moran-table td {
    padding: 8px 12px;
    color: #cbd5e1;
    border-bottom: 1px solid #1e2d4a;
}
.moran-table tr:hover td { background: rgba(56,189,248,0.04); }

/* ---- SIDEBAR LOGO AREA ---- */
.sidebar-brand {
    background: linear-gradient(135deg, rgba(56,189,248,0.08), rgba(129,140,248,0.05));
    border: 1px solid rgba(56,189,248,0.15);
    border-radius: 12px;
    padding: 1rem;
    margin-bottom: 1.5rem;
    text-align: center;
}
.sidebar-brand-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #f8fafc;
    margin: 0.3rem 0 0.1rem 0;
}
.sidebar-brand-sub {
    font-size: 0.7rem;
    color: #64748b;
    margin: 0;
}
.sidebar-section {
    font-size: 0.7rem;
    font-weight: 700;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin: 1.2rem 0 0.5rem 0;
    padding-left: 2px;
}

/* ---- WELCOME SCREEN ---- */
.welcome-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin: 1.5rem 0;
}
.welcome-card {
    background: #111827;
    border: 1px solid #1e2d4a;
    border-radius: 12px;
    padding: 1.2rem;
}
.welcome-card-icon { font-size: 1.5rem; margin-bottom: 0.6rem; }
.welcome-card h4 { color: #e2e8f0; font-size: 0.88rem; font-weight: 600; margin: 0 0 0.4rem 0; }
.welcome-card p { color: #64748b; font-size: 0.78rem; line-height: 1.5; margin: 0; }

.steps-list { list-style: none; padding: 0; margin: 1rem 0; }
.steps-list li {
    display: flex; align-items: flex-start; gap: 10px;
    padding: 8px 0;
    border-bottom: 1px solid #1e2d4a;
    font-size: 0.82rem; color: #94a3b8;
}
.step-num {
    background: #1e3a5f;
    color: #38bdf8;
    width: 20px; height: 20px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.7rem; font-weight: 700;
    flex-shrink: 0; margin-top: 1px;
}

/* ---- SCROLLBAR ---- */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0b0f1a; }
::-webkit-scrollbar-thumb { background: #1e2d4a; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #2d4a6e; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# CONSTANTES
# =============================================================================
CLUSTER_COLORS = {
    "HH": "#ef4444",
    "LL": "#3b82f6",
    "HL": "#f97316",
    "LH": "#22c55e",
    "NS": "#334155",
}
CLUSTER_LABELS = {
    "HH": "Alto–Alto",
    "LL": "Bajo–Bajo",
    "HL": "Alto–Bajo (outlier)",
    "LH": "Bajo–Alto (outlier)",
    "NS": "No significativo",
}

# =============================================================================
# FUNCIONES
# =============================================================================
@st.cache_data
def cargar_geodataframe(uploaded_file):
    nombre = uploaded_file.name.lower()
    if nombre.endswith(".geojson") or nombre.endswith(".json"):
        gdf = gpd.read_file(io.BytesIO(uploaded_file.getvalue()))
    elif nombre.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
        lat_col = next((c for c in df.columns if c.lower() in ("lat","latitude","latitud")), None)
        lon_col = next((c for c in df.columns if c.lower() in ("lon","lng","longitude","longitud")), None)
        if not lat_col or not lon_col:
            st.error("CSV necesita columnas latitude/longitude")
            return None
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[lon_col], df[lat_col]), crs="EPSG:4326")
    elif nombre.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
        lat_col = next((c for c in df.columns if c.lower() in ("lat","latitude","latitud")), None)
        lon_col = next((c for c in df.columns if c.lower() in ("lon","lng","longitude","longitud")), None)
        if lat_col and lon_col:
            gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[lon_col], df[lat_col]), crs="EPSG:4326")
        else:
            st.error("Excel necesita columnas latitude/longitude")
            return None
    else:
        st.error("Formato no soportado. Use: GeoJSON, CSV, XLSX")
        return None
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    gdf = gdf.to_crs("EPSG:4326").dropna(subset=["geometry"])
    gdf["geometry"] = gdf["geometry"].buffer(0)
    return gdf.reset_index(drop=True)

def construir_pesos(gdf, metodo, k=5):
    if metodo == "Queen":
        w = Queen.from_dataframe(gdf, silence_warnings=True)
    elif metodo == "Rook":
        w = Rook.from_dataframe(gdf, silence_warnings=True)
    else:
        w = KNN.from_dataframe(gdf, k=k, silence_warnings=True)
    w.transform = "R"
    return w

def calcular_moran_global(y, w, perm=999):
    return Moran(y, w, permutations=perm)

def calcular_moran_local(y, w, perm=999, sig=0.05):
    ml = Moran_Local(y, w, permutations=perm)
    z = (y - y.mean()) / y.std()
    lag = libpysal.weights.lag_spatial(w, y)
    lag_z = (lag - lag.mean()) / lag.std()
    sig_mask = ml.p_sim < sig
    cluster = np.where(sig_mask,
        np.where(z > 0, np.where(lag_z > 0, "HH", "HL"), np.where(lag_z > 0, "LH", "LL")),
        "NS")
    return ml, z, lag_z, cluster

def exportar_excel(gdf, global_res):
    output = io.BytesIO()
    cols = [c for c in gdf.columns if c != "geometry"]
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        pd.DataFrame([global_res]).to_excel(writer, sheet_name="Moran Global", index=False)
        gdf[cols].to_excel(writer, sheet_name="LISA por unidad", index=False)
        if "LISA_cluster" in gdf.columns:
            resumen = gdf.groupby("LISA_cluster").size().reset_index(name="N")
            resumen["%"] = (resumen["N"] / len(gdf) * 100).round(2)
            resumen.to_excel(writer, sheet_name="Resumen clusters", index=False)
    return output.getvalue()

def interpretar_moran(I, p):
    if p > 0.05:
        return "random", "Sin autocorrelación espacial significativa", \
               f"Con I = {I:.4f} y p = {p:.4f}, el patrón es estadísticamente aleatorio. No hay evidencia de clustering espacial."
    if I > 0:
        return "positive", "✦ Autocorrelación espacial POSITIVA detectada", \
               f"I = {I:.4f} (p = {p:.4f}). Las unidades con valores similares tienden a agruparse geográficamente. Existen clusters espaciales estadísticamente significativos."
    return "negative", "⚡ Autocorrelación espacial NEGATIVA detectada", \
           f"I = {I:.4f} (p = {p:.4f}). Los valores altos y bajos se alternan entre vecinos. Patrón de dispersión espacial (efecto tablero de ajedrez)."

# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown("""
    <div class='sidebar-brand'>
        <div style='font-size:1.8rem'>🗺️</div>
        <div class='sidebar-brand-title'>GeoMoran Analyst</div>
        <div class='sidebar-brand-sub'>Autocorrelación Espacial · LISA</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='sidebar-section'>Dataset</div>", unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Subir archivo geográfico",
        type=["geojson","json","csv","xlsx"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    st.markdown("<div class='sidebar-section'>Matriz de pesos W</div>", unsafe_allow_html=True)
    metodo_w = st.selectbox("", ["Queen","Rook","KNN"], label_visibility="collapsed")
    k_vecinos = 5
    if metodo_w == "KNN":
        k_vecinos = st.slider("K vecinos", 3, 15, 5)

    st.markdown("<div class='sidebar-section'>Inferencia estadística</div>", unsafe_allow_html=True)
    permutaciones = st.select_slider("Permutaciones", [99,199,499,999], value=999)
    nivel_sig = st.select_slider("Nivel α", [0.01,0.05,0.10], value=0.05)

    st.markdown("<br>", unsafe_allow_html=True)
    analizar_btn = st.button("▶  Ejecutar análisis", use_container_width=True)

    st.markdown("""
    <div style='margin-top:2rem;padding:1rem;background:#111827;border:1px solid #1e2d4a;border-radius:10px;font-size:0.72rem;color:#475569;line-height:1.6'>
        <strong style='color:#64748b'>GeoMoran Analyst v1.0</strong><br>
        PySAL · GeoPandas · Streamlit<br>
        Proyecto de investigación universitaria
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
# MAIN AREA
# =============================================================================

# Hero header
st.markdown("""
<div class='hero-header'>
    <div class='hero-title'>GeoMoran <span>Analyst</span></div>
    <p class='hero-subtitle'>Análisis de autocorrelación espacial — Índice de Moran Global & LISA</p>
    <div class='hero-badges'>
        <span class='badge'>Moran I Global</span>
        <span class='badge'>LISA Clusters</span>
        <span class='badge'>Scatter Plot</span>
        <span class='badge'>Cluster Map</span>
        <span class='badge'>Export PDF / Excel</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Estado
if "datasets" not in st.session_state:
    st.session_state.datasets = {}
if "resultados" not in st.session_state:
    st.session_state.resultados = {}

# Cargar archivos
if uploaded_files:
    st.markdown("<div class='section-header'><div class='section-dot'></div><h3>Datasets cargados</h3></div>", unsafe_allow_html=True)
    tabs_prev = st.tabs([f"📂 {f.name}" for f in uploaded_files])
    for i, (tab, uf) in enumerate(zip(tabs_prev, uploaded_files)):
        with tab:
            gdf = cargar_geodataframe(uf)
            if gdf is not None:
                st.session_state.datasets[uf.name] = gdf
                c1, c2, c3 = st.columns(3)
                c1.metric("Unidades espaciales", len(gdf))
                c2.metric("Variables numéricas", len(gdf.select_dtypes(include="number").columns))
                c3.metric("CRS", str(gdf.crs).split(":")[-1] if gdf.crs else "N/A")
                num_cols = gdf.select_dtypes(include="number").columns.tolist()
                if num_cols:
                    var = st.selectbox(f"Variable a analizar", num_cols, key=f"var_{i}")
                    st.session_state.datasets[uf.name + "_var"] = var
                st.dataframe(
                    gdf.drop(columns="geometry").head(5),
                    use_container_width=True,
                    hide_index=True,
                )

# Ejecutar análisis
if analizar_btn and st.session_state.datasets:
    comp_list = []
    for nombre, gdf in {k:v for k,v in st.session_state.datasets.items() if not k.endswith("_var")}.items():
        var = st.session_state.datasets.get(nombre + "_var")
        if not var or var not in gdf.columns:
            continue
        with st.spinner(f"Calculando Índice de Moran para {nombre}…"):
            y = gdf[var].astype(float).values
            w = construir_pesos(gdf, metodo_w, k_vecinos)
            mg = calcular_moran_global(y, w, permutaciones)
            ml, z_vals, lag_z, clusters = calcular_moran_local(y, w, permutaciones, nivel_sig)
            gdf = gdf.copy()
            gdf["LISA_I"] = ml.Is
            gdf["LISA_p"] = ml.p_sim
            gdf["LISA_z"] = z_vals
            gdf["LISA_lag"] = lag_z
            gdf["LISA_cluster"] = clusters
            global_res = {
                "Dataset": nombre, "Variable": var,
                "I_Moran": round(mg.I,6), "E[I]": round(mg.EI,6),
                "z_score": round(mg.z_norm,4), "p_valor": round(mg.p_norm,6),
                "p_simulado": round(mg.p_sim,6), "Permutaciones": permutaciones,
                "Significativo": "Sí" if mg.p_norm < nivel_sig else "No",
            }
            st.session_state.resultados[nombre] = {
                "gdf": gdf, "y": y, "w": w, "mg": mg, "ml": ml,
                "z": z_vals, "lag_z": lag_z, "clusters": clusters,
                "var": var, "global_res": global_res,
            }
            comp_list.append({"nombre": nombre.replace(".geojson","").replace(".csv",""),
                               "I_moran": mg.I, "p_value": mg.p_norm, "variable": var})
    st.session_state.comp_list = comp_list
    st.success(f"✅ Análisis completado para {len(comp_list)} dataset(s)")

# Mostrar resultados
if st.session_state.resultados:
    nombres = list(st.session_state.resultados.keys())
    tab_names = [f"📊 {n.replace('.geojson','').replace('.csv','')[:20]}" for n in nombres]
    if len(nombres) > 1:
        tab_names.append("⚖️ Comparación")
    tabs = st.tabs(tab_names)

    for i, nombre in enumerate(nombres):
        res = st.session_state.resultados[nombre]
        gdf = res["gdf"]
        mg = res["mg"]
        var = res["var"]
        tipo, titulo, texto = interpretar_moran(mg.I, mg.p_norm)

        with tabs[i]:
            # Métricas
            sig_color = "#34d399" if mg.p_norm < nivel_sig else "#f97316"
            sig_text = "Significativo" if mg.p_norm < nivel_sig else "No significativo"
            st.markdown(f"""
            <div class='metrics-row'>
                <div class='metric-card blue'>
                    <div class='metric-label'>Índice de Moran I</div>
                    <div class='metric-value'>{mg.I:.4f}</div>
                    <div class='metric-sub'>E[I] = {mg.EI:.4f}</div>
                </div>
                <div class='metric-card purple'>
                    <div class='metric-label'>Z-score</div>
                    <div class='metric-value'>{mg.z_norm:.3f}</div>
                    <div class='metric-sub'>Distribución normal</div>
                </div>
                <div class='metric-card {"green" if mg.p_norm < nivel_sig else "amber"}'>
                    <div class='metric-label'>P-valor</div>
                    <div class='metric-value'>{mg.p_norm:.4f}</div>
                    <div class='metric-sub' style='color:{sig_color}'>{sig_text}</div>
                </div>
                <div class='metric-card amber'>
                    <div class='metric-label'>P simulado ({permutaciones} perm.)</div>
                    <div class='metric-value'>{mg.p_sim:.4f}</div>
                    <div class='metric-sub'>Variable: {var}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Interpretación
            css = {"positive":"interp-positive","negative":"interp-negative","random":"interp-random"}[tipo]
            st.markdown(f"""
            <div class='{css}'>
                <div class='interp-title'>{titulo}</div>
                <div class='interp-text'>{texto}</div>
            </div>
            """, unsafe_allow_html=True)

            # Leyenda clusters
            conteo = pd.Series(res["clusters"]).value_counts()
            badges = "".join([
                f"<div class='cluster-badge'><div class='cluster-dot' style='background:{CLUSTER_COLORS[cl]}'></div>"
                f"<span><strong>{cl}</strong> · {CLUSTER_LABELS[cl]} · <strong>{conteo.get(cl,0)}</strong></span></div>"
                for cl in ["HH","LL","HL","LH","NS"]
            ])
            st.markdown(f"<div class='cluster-legend'>{badges}</div>", unsafe_allow_html=True)

            # Tabs de visualización
            tv1, tv2, tv3, tv4 = st.tabs(["🗺️ Mapas", "📉 Scatter Plot", "🎯 Cluster Map", "📋 Tabla LISA"])

            with tv1:
                gdf_wgs = gdf.to_crs("EPSG:4326")
                cy = gdf_wgs.geometry.centroid.y.mean()
                cx = gdf_wgs.geometry.centroid.x.mean()
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("<div class='section-header'><div class='section-dot' style='background:#38bdf8'></div><h3>Mapa Coroplético</h3></div>", unsafe_allow_html=True)
                    m1 = folium.Map(location=[cy, cx], zoom_start=8, tiles="CartoDB dark_matter")
                    vals = gdf[var].astype(float)
                    vmin, vmax = vals.min(), vals.max()
                    def get_color(v):
                        t = (v - vmin) / (vmax - vmin + 1e-9)
                        if t < 0.25: return "#1e40af"
                        elif t < 0.5: return "#3b82f6"
                        elif t < 0.75: return "#f97316"
                        return "#ef4444"
                    folium.GeoJson(
                        gdf_wgs.__geo_interface__,
                        style_function=lambda f, v=var: {
                            "fillColor": get_color(f["properties"].get(v, 0) or 0),
                            "color": "#1e2d4a", "weight": 0.8, "fillOpacity": 0.8,
                        },
                        tooltip=folium.GeoJsonTooltip(fields=[var], aliases=[var]),
                    ).add_to(m1)
                    st_folium(m1, height=380, use_container_width=True, key=f"coro_{i}")

                with col2:
                    st.markdown("<div class='section-header'><div class='section-dot' style='background:#818cf8'></div><h3>LISA Cluster Map</h3></div>", unsafe_allow_html=True)
                    m2 = folium.Map(location=[cy, cx], zoom_start=8, tiles="CartoDB dark_matter")
                    folium.GeoJson(
                        gdf_wgs.__geo_interface__,
                        style_function=lambda f: {
                            "fillColor": CLUSTER_COLORS.get(f["properties"].get("LISA_cluster","NS"), "#334155"),
                            "color": "#1e2d4a", "weight": 0.8, "fillOpacity": 0.85,
                        },
                        tooltip=folium.GeoJsonTooltip(
                            fields=[var,"LISA_cluster","LISA_p"],
                            aliases=["Variable","Cluster","p-valor"],
                        ),
                    ).add_to(m2)
                    st_folium(m2, height=380, use_container_width=True, key=f"lisa_m_{i}")

            with tv2:
                st.markdown("<div class='section-header'><div class='section-dot' style='background:#f97316'></div><h3>Moran Scatter Plot</h3></div>", unsafe_allow_html=True)
                fig = go.Figure()
                fig.update_layout(
                    paper_bgcolor="#111827", plot_bgcolor="#0f172a",
                    font=dict(family="Inter", color="#94a3b8"),
                    xaxis=dict(gridcolor="#1e2d4a", zerolinecolor="#334155", title=f"{var} (estandarizado)"),
                    yaxis=dict(gridcolor="#1e2d4a", zerolinecolor="#334155", title="Lag espacial"),
                    title=dict(text=f"Moran Scatter Plot  ·  I = {mg.I:.4f}  ·  p = {mg.p_norm:.4f}",
                               font=dict(size=14, color="#e2e8f0")),
                    height=480, legend=dict(bgcolor="#111827", bordercolor="#1e2d4a"),
                    margin=dict(l=60, r=30, t=60, b=60),
                )
                for cl, color in CLUSTER_COLORS.items():
                    mask = gdf["LISA_cluster"] == cl
                    if mask.any():
                        fig.add_trace(go.Scatter(
                            x=res["z"][mask], y=res["lag_z"][mask],
                            mode="markers", name=f"{cl} · {CLUSTER_LABELS[cl]}",
                            marker=dict(color=color, size=8, opacity=0.85,
                                        line=dict(color="#0f172a", width=0.5)),
                        ))
                x_line = np.linspace(res["z"].min(), res["z"].max(), 100)
                fig.add_trace(go.Scatter(
                    x=x_line, y=mg.I * x_line, mode="lines",
                    line=dict(color="#38bdf8", width=2, dash="dot"),
                    name=f"Pendiente = I = {mg.I:.4f}", showlegend=True,
                ))
                fig.add_hline(y=0, line_color="#334155", line_width=1)
                fig.add_vline(x=0, line_color="#334155", line_width=1)
                st.plotly_chart(fig, use_container_width=True)

            with tv3:
                st.markdown("<div class='section-header'><div class='section-dot' style='background:#22c55e'></div><h3>LISA Cluster Map — Plotly</h3></div>", unsafe_allow_html=True)
                gdf_p = gdf.to_crs("EPSG:4326")
                fig_cl = px.choropleth_mapbox(
                    gdf_p, geojson=gdf_p.__geo_interface__,
                    locations=gdf_p.index, color="LISA_cluster",
                    color_discrete_map=CLUSTER_COLORS,
                    mapbox_style="carto-darkmatter",
                    center={"lat": cy, "lon": cx}, zoom=7, opacity=0.85,
                    hover_data=[var,"LISA_I","LISA_p"],
                    title="Clusters LISA significativos (p < α)",
                    category_orders={"LISA_cluster": ["HH","LL","HL","LH","NS"]},
                )
                fig_cl.update_layout(
                    paper_bgcolor="#111827", font=dict(color="#94a3b8"),
                    height=520, margin=dict(l=0,r=0,t=40,b=0),
                    legend=dict(bgcolor="#111827", bordercolor="#1e2d4a",
                                title_text="Cluster LISA"),
                )
                st.plotly_chart(fig_cl, use_container_width=True)

            with tv4:
                st.markdown("<div class='section-header'><div class='section-dot' style='background:#fbbf24'></div><h3>Resultados LISA por unidad</h3></div>", unsafe_allow_html=True)
                cols_show = [c for c in [var,"LISA_I","LISA_p","LISA_cluster","LISA_sig"] if c in gdf.columns]
                tabla = gdf[cols_show].copy()
                tabla["LISA_I"] = tabla["LISA_I"].round(4)
                tabla["LISA_p"] = tabla["LISA_p"].round(4)
                st.dataframe(tabla, use_container_width=True, height=420, hide_index=True)

            # Exportar
            st.markdown("<div class='section-header'><div class='section-dot' style='background:#818cf8'></div><h3>Exportar resultados</h3></div>", unsafe_allow_html=True)
            ec1, ec2 = st.columns(2)
            with ec1:
                xls = exportar_excel(gdf, res["global_res"])
                st.download_button("📥 Descargar Excel (.xlsx)", xls,
                    file_name=f"moran_{nombre[:20]}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True)
            with ec2:
                geo_str = gdf.to_crs("EPSG:4326").to_json()
                st.download_button("📥 Descargar GeoJSON con LISA", geo_str,
                    file_name=f"lisa_{nombre[:20]}.geojson",
                    mime="application/json", use_container_width=True)

    # Tab comparación
    if len(nombres) > 1:
        with tabs[-1]:
            comp = st.session_state.get("comp_list", [])
            if comp:
                st.markdown("<div class='section-header'><div class='section-dot'></div><h3>Comparación del Índice de Moran entre datasets</h3></div>", unsafe_allow_html=True)
                df_comp = pd.DataFrame(comp)
                colores = [CLUSTER_COLORS["HH"] if p < 0.05 else "#475569" for p in df_comp["p_value"]]
                fig_comp = go.Figure()
                fig_comp.add_trace(go.Bar(
                    x=df_comp["nombre"], y=df_comp["I_moran"],
                    marker_color=colores, text=df_comp["I_moran"].round(4),
                    textposition="outside", textfont=dict(color="#94a3b8"),
                ))
                fig_comp.add_hline(y=0, line_color="#334155", line_width=1)
                fig_comp.update_layout(
                    paper_bgcolor="#111827", plot_bgcolor="#0f172a",
                    font=dict(family="Inter", color="#94a3b8"),
                    xaxis=dict(gridcolor="#1e2d4a"), yaxis=dict(gridcolor="#1e2d4a"),
                    title=dict(text="Índice de Moran I por dataset (rojo = p < 0.05)",
                               font=dict(size=14, color="#e2e8f0")),
                    height=380, margin=dict(l=60,r=30,t=50,b=60),
                )
                st.plotly_chart(fig_comp, use_container_width=True)
                st.dataframe(df_comp.rename(columns={
                    "nombre":"Dataset","variable":"Variable",
                    "I_moran":"I de Moran","p_value":"p-valor"
                }), use_container_width=True, hide_index=True)

else:
    # Pantalla de bienvenida
    st.markdown("""
    <div class='info-card'>
        <h4>👆 Carga un dataset geográfico para comenzar</h4>
        <div class='welcome-grid'>
            <div class='welcome-card'>
                <div class='welcome-card-icon'>📂</div>
                <h4>Formatos soportados</h4>
                <p>GeoJSON · Shapefile · CSV con lat/lon · Excel con coordenadas</p>
            </div>
            <div class='welcome-card'>
                <div class='welcome-card-icon'>🔬</div>
                <h4>Qué calcula</h4>
                <p>Índice de Moran Global · LISA local · Clusters HH/LL/HL/LH · Test de permutaciones</p>
            </div>
            <div class='welcome-card'>
                <div class='welcome-card-icon'>📤</div>
                <h4>Exportación</h4>
                <p>Resultados en Excel · GeoJSON con LISA · Mapas interactivos</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='info-card'>
        <h4>📖 Guía rápida</h4>
        <ul class='steps-list'>
            <li><span class='step-num'>1</span>Sube un archivo GeoJSON o CSV con coordenadas desde el panel izquierdo</li>
            <li><span class='step-num'>2</span>Selecciona la variable numérica que deseas analizar</li>
            <li><span class='step-num'>3</span>Elige el tipo de matriz de pesos espaciales (Queen recomendado)</li>
            <li><span class='step-num'>4</span>Ajusta el número de permutaciones y nivel de significancia</li>
            <li><span class='step-num'>5</span>Pulsa <strong>Ejecutar análisis</strong> y explora los resultados</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='info-card'>
        <h4>📊 Interpretación del Índice de Moran I</h4>
        <table class='moran-table'>
            <thead><tr><th>Valor de I</th><th>p-valor</th><th>Interpretación</th><th>Patrón espacial</th></tr></thead>
            <tbody>
                <tr><td>I &gt; 0</td><td>p &lt; 0.05</td><td style='color:#34d399'>Autocorrelación positiva</td><td>Clusters de valores similares</td></tr>
                <tr><td>I ≈ 0</td><td>p &gt; 0.05</td><td style='color:#94a3b8'>Sin patrón significativo</td><td>Distribución aleatoria</td></tr>
                <tr><td>I &lt; 0</td><td>p &lt; 0.05</td><td style='color:#fbbf24'>Autocorrelación negativa</td><td>Dispersión — valores alternados</td></tr>
            </tbody>
        </table>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    pass