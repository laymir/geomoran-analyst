"""
GeoMoran Analyst — Aplicación Web con Streamlit
================================================
Interfaz web completa para análisis de autocorrelación espacial.

Instalar dependencias:
    pip install streamlit geopandas pysal esda libpysal folium
    pip install streamlit-folium plotly reportlab openpyxl

Ejecutar:
    streamlit run app_streamlit.py
"""

import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import folium
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
import libpysal
from libpysal.weights import Queen, Rook, KNN
from esda.moran import Moran, Moran_Local
import io
import json
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURACIÓN DE LA APP
# =============================================================================

st.set_page_config(
    page_title="GeoMoran Analyst",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a237e, #283593);
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #f8f9fa;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .cluster-HH { color: #C0392B; font-weight: bold; }
    .cluster-LL { color: #2980B9; font-weight: bold; }
    .cluster-HL { color: #E67E22; font-weight: bold; }
    .cluster-LH { color: #27AE60; font-weight: bold; }
    .cluster-NS { color: #7f8c8d; }
    .interpretation-positive {
        background: #e8f5e9; border-left: 4px solid #4caf50;
        padding: 1rem; border-radius: 6px;
    }
    .interpretation-negative {
        background: #fff3e0; border-left: 4px solid #ff9800;
        padding: 1rem; border-radius: 6px;
    }
    .interpretation-random {
        background: #f3f4f6; border-left: 4px solid #9e9e9e;
        padding: 1rem; border-radius: 6px;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================

CLUSTER_COLORS = {
    "HH": "#C0392B", "LL": "#2980B9",
    "HL": "#E67E22", "LH": "#27AE60", "NS": "#BDC3C7",
}


@st.cache_data
def cargar_geodataframe(uploaded_file) -> gpd.GeoDataFrame:
    """Carga y valida el archivo geográfico subido."""
    nombre = uploaded_file.name.lower()

    if nombre.endswith(".geojson") or nombre.endswith(".json"):
        gdf = gpd.read_file(io.BytesIO(uploaded_file.getvalue()))
    elif nombre.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
        lat_col = next((c for c in df.columns if c.lower() in ("lat","latitude","latitud")), None)
        lon_col = next((c for c in df.columns if c.lower() in ("lon","lng","longitude","longitud")), None)
        if not lat_col or not lon_col:
            st.error("CSV debe tener columnas latitude/longitude")
            return None
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[lon_col], df[lat_col]), crs="EPSG:4326")
    elif nombre.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
        lat_col = next((c for c in df.columns if c.lower() in ("lat","latitude","latitud")), None)
        lon_col = next((c for c in df.columns if c.lower() in ("lon","lng","longitude","longitud")), None)
        if lat_col and lon_col:
            gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[lon_col], df[lat_col]), crs="EPSG:4326")
        else:
            st.error("Excel debe tener columnas latitude/longitude")
            return None
    else:
        st.error("Formato no soportado. Use: GeoJSON, CSV, XLSX")
        return None

    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    gdf = gdf.to_crs("EPSG:4326")
    gdf = gdf.dropna(subset=["geometry"])
    gdf["geometry"] = gdf["geometry"].buffer(0)
    return gdf.reset_index(drop=True)


def construir_pesos(gdf, metodo, k=5):
    """Construye la matriz de pesos espaciales."""
    if metodo == "Queen":
        w = Queen.from_dataframe(gdf, silence_warnings=True)
    elif metodo == "Rook":
        w = Rook.from_dataframe(gdf, silence_warnings=True)
    else:  # KNN
        w = KNN.from_dataframe(gdf, k=k, silence_warnings=True)
    w.transform = "R"
    return w


def calcular_moran_global(y, w, permutaciones=999):
    """Calcula el Índice de Moran Global."""
    return Moran(y, w, permutations=permutaciones)


def calcular_moran_local(y, w, permutaciones=999, sig=0.05):
    """Calcula LISA y clasifica cuadrantes."""
    ml = Moran_Local(y, w, permutations=permutaciones)
    z = (y - y.mean()) / y.std()
    lag = libpysal.weights.lag_spatial(w, y)
    lag_z = (lag - lag.mean()) / lag.std()
    significativo = ml.p_sim < sig
    cluster = np.where(
        significativo,
        np.where(z > 0, np.where(lag_z > 0, "HH", "HL"), np.where(lag_z > 0, "LH", "LL")),
        "NS"
    )
    return ml, z, lag_z, cluster


def exportar_excel(gdf, global_res, filename="resultados.xlsx"):
    """Exporta a Excel en memoria para descarga."""
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


def interpretar_moran(I, p_value):
    """Retorna interpretación y clase CSS."""
    if p_value > 0.05:
        return ("No significativo. No hay evidencia suficiente de autocorrelación "
                "espacial. El patrón podría ser aleatorio."), "random"
    if I > 0:
        return (f"Autocorrelación espacial POSITIVA (I = {I:.4f}). "
                "Las unidades con valores similares tienden a agruparse geográficamente. "
                "Existen clusters espaciales."), "positive"
    return (f"Autocorrelación espacial NEGATIVA (I = {I:.4f}). "
            "Las unidades con valores diferentes tienden a ser vecinas. "
            "Patrón de dispersión (tablero de ajedrez)."), "negative"


# =============================================================================
# INTERFAZ PRINCIPAL
# =============================================================================

def main():
    # Header
    st.markdown("""
    <div class='main-header'>
        <h1 style='margin:0;font-size:2rem'>🗺️ GeoMoran Analyst</h1>
        <p style='margin:0.4rem 0 0;opacity:0.85'>
            Análisis de Autocorrelación Espacial — Índice de Moran Global y Local (LISA)
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuración")

        st.subheader("1. Cargar datos")
        uploaded_files = st.file_uploader(
            "Subir dataset(s) geográfico(s)",
            type=["geojson", "json", "csv", "xlsx"],
            accept_multiple_files=True,
            help="Formatos: GeoJSON, CSV (lat/lon), Excel (lat/lon)"
        )

        st.subheader("2. Parámetros de análisis")
        metodo_w = st.selectbox(
            "Matriz de pesos espaciales",
            ["Queen", "Rook", "KNN"],
            help="Queen=8 vecinos, Rook=4 vecinos, KNN=K más cercanos"
        )
        k_vecinos = 5
        if metodo_w == "KNN":
            k_vecinos = st.slider("Número de vecinos K", 3, 15, 5)

        permutaciones = st.select_slider(
            "Permutaciones (test de significancia)",
            options=[99, 199, 499, 999],
            value=999
        )
        nivel_sig = st.select_slider(
            "Nivel de significancia α",
            options=[0.01, 0.05, 0.10],
            value=0.05
        )

        analizar_btn = st.button("🔬 Ejecutar análisis", type="primary", use_container_width=True)

    # Estado de la sesión
    if "datasets" not in st.session_state:
        st.session_state.datasets = {}
    if "resultados" not in st.session_state:
        st.session_state.resultados = {}

    # Cargar y previsualizar archivos
    if uploaded_files:
        st.subheader("📂 Datasets cargados")
        tabs_preview = st.tabs([f.name for f in uploaded_files])

        for i, (tab, uf) in enumerate(zip(tabs_preview, uploaded_files)):
            with tab:
                gdf = cargar_geodataframe(uf)
                if gdf is not None:
                    st.session_state.datasets[uf.name] = gdf
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Unidades espaciales", len(gdf))
                    col2.metric("Variables disponibles", len(gdf.select_dtypes(include="number").columns))
                    col3.metric("CRS", str(gdf.crs).split(":")[-1] if gdf.crs else "N/A")

                    # Selector de variable para este dataset
                    num_cols = gdf.select_dtypes(include="number").columns.tolist()
                    if num_cols:
                        var = st.selectbox(f"Variable a analizar ({uf.name})", num_cols, key=f"var_{i}")
                        st.session_state.datasets[uf.name + "_var"] = var
                    st.dataframe(gdf.drop(columns="geometry").head(5), use_container_width=True)

    # Ejecutar análisis
    if analizar_btn and st.session_state.datasets:
        resultados_comparacion = []

        for nombre_archivo, gdf in {k: v for k, v in st.session_state.datasets.items()
                                     if not k.endswith("_var")}.items():
            var = st.session_state.datasets.get(nombre_archivo + "_var")
            if var is None or var not in gdf.columns:
                continue

            with st.spinner(f"Analizando {nombre_archivo}..."):
                y = gdf[var].astype(float).values
                w = construir_pesos(gdf, metodo_w, k_vecinos)
                mg = calcular_moran_global(y, w, permutaciones)
                ml, z_vals, lag_z, clusters = calcular_moran_local(y, w, permutaciones, nivel_sig)

                gdf["LISA_I"] = ml.Is
                gdf["LISA_p"] = ml.p_sim
                gdf["LISA_z"] = z_vals
                gdf["LISA_lag"] = lag_z
                gdf["LISA_cluster"] = clusters

                global_res = {
                    "Dataset": nombre_archivo,
                    "Variable": var,
                    "I_Moran": round(mg.I, 6),
                    "E[I]": round(mg.EI, 6),
                    "z_score": round(mg.z_norm, 4),
                    "p_valor": round(mg.p_norm, 6),
                    "p_simulado": round(mg.p_sim, 6),
                    "Permutaciones": permutaciones,
                    "Significativo": "Sí" if mg.p_norm < nivel_sig else "No",
                }

                st.session_state.resultados[nombre_archivo] = {
                    "gdf": gdf, "y": y, "w": w,
                    "moran_global": mg, "moran_local": ml,
                    "z": z_vals, "lag_z": lag_z,
                    "clusters": clusters, "var": var,
                    "global_res": global_res,
                }

                resultados_comparacion.append({
                    "nombre": nombre_archivo.replace(".geojson", "").replace(".csv", ""),
                    "I_moran": mg.I, "p_value": mg.p_norm, "variable": var,
                })

        st.session_state.resultados_comparacion = resultados_comparacion
        st.success(f"✅ Análisis completado para {len(resultados_comparacion)} dataset(s)")

    # Mostrar resultados
    if st.session_state.resultados:
        st.divider()
        st.header("📊 Resultados del análisis")

        # Tabs por dataset + comparación
        nombres = list(st.session_state.resultados.keys())
        tab_names = nombres + (["🔄 Comparación entre datasets"] if len(nombres) > 1 else [])
        tabs = st.tabs(tab_names)

        for i, nombre in enumerate(nombres):
            res = st.session_state.resultados[nombre]
            gdf = res["gdf"]
            mg = res["moran_global"]
            var = res["var"]

            with tabs[i]:
                # Métricas principales
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Índice de Moran I", f"{mg.I:.4f}")
                c2.metric("z-score", f"{mg.z_norm:.3f}")
                c3.metric("p-valor", f"{mg.p_norm:.4f}")
                c4.metric("p simulado", f"{mg.p_sim:.4f}")

                # Interpretación
                texto, tipo = interpretar_moran(mg.I, mg.p_norm)
                color_map = {"positive": "#e8f5e9", "negative": "#fff3e0", "random": "#f3f4f6"}
                border_map = {"positive": "#4caf50", "negative": "#ff9800", "random": "#9e9e9e"}
                st.markdown(f"""
                <div style='background:{color_map[tipo]};border-left:4px solid {border_map[tipo]};
                            padding:1rem;border-radius:6px;margin:1rem 0'>
                    <strong>📌 Interpretación:</strong> {texto}
                </div>
                """, unsafe_allow_html=True)

                # Visualizaciones
                tab_v1, tab_v2, tab_v3, tab_v4 = st.tabs([
                    "🗺️ Mapa interactivo", "📉 Scatter Plot", "🎯 Cluster Map", "📋 Tabla LISA"
                ])

                with tab_v1:
                    gdf_wgs = gdf.to_crs("EPSG:4326")
                    centro_y = gdf_wgs.geometry.centroid.y.mean()
                    centro_x = gdf_wgs.geometry.centroid.x.mean()

                    col_mapa1, col_mapa2 = st.columns(2)

                    with col_mapa1:
                        st.write("**Mapa Coroplético**")
                        m_coro = folium.Map(location=[centro_y, centro_x], zoom_start=8, tiles="CartoDB positron")
                        folium.GeoJson(
                            gdf_wgs.__geo_interface__,
                            style_function=lambda f, v=var: {
                                "fillColor": "#2c7bb6" if f["properties"].get(v, 0) < gdf[v].mean() else "#d7191c",
                                "color": "white", "weight": 0.5, "fillOpacity": 0.7,
                            },
                            tooltip=folium.GeoJsonTooltip(fields=[var]),
                        ).add_to(m_coro)
                        st_folium(m_coro, height=380, use_container_width=True, key=f"coro_{i}")

                    with col_mapa2:
                        st.write("**LISA Cluster Map (interactivo)**")
                        m_lisa = folium.Map(location=[centro_y, centro_x], zoom_start=8, tiles="CartoDB positron")
                        def style_lisa(feature):
                            c = feature["properties"].get("LISA_cluster", "NS")
                            return {"fillColor": CLUSTER_COLORS.get(c,"#BDC3C7"),
                                    "color":"white","weight":0.5,"fillOpacity":0.75}
                        folium.GeoJson(
                            gdf_wgs.__geo_interface__,
                            style_function=style_lisa,
                            tooltip=folium.GeoJsonTooltip(
                                fields=[var, "LISA_I", "LISA_p", "LISA_cluster"],
                                aliases=["Variable","LISA I","p-valor","Cluster"],
                            ),
                        ).add_to(m_lisa)
                        st_folium(m_lisa, height=380, use_container_width=True, key=f"lisa_map_{i}")

                with tab_v2:
                    st.write("**Moran Scatter Plot**")
                    fig_scatter = go.Figure()

                    for cl, color in CLUSTER_COLORS.items():
                        mask = gdf["LISA_cluster"] == cl
                        if mask.any():
                            fig_scatter.add_trace(go.Scatter(
                                x=res["z"][mask], y=res["lag_z"][mask],
                                mode="markers", name=cl,
                                marker=dict(color=color, size=7, opacity=0.8),
                            ))

                    x_line = np.linspace(res["z"].min(), res["z"].max(), 100)
                    fig_scatter.add_trace(go.Scatter(
                        x=x_line, y=mg.I * x_line, mode="lines",
                        line=dict(color="black", width=1.5),
                        name=f"I = {mg.I:.4f}"
                    ))
                    fig_scatter.add_hline(y=0, line_dash="dot", line_color="gray", line_width=0.8)
                    fig_scatter.add_vline(x=0, line_dash="dot", line_color="gray", line_width=0.8)
                    fig_scatter.update_layout(
                        xaxis_title=f"{var} (estandarizado)",
                        yaxis_title=f"Lag espacial",
                        title=f"Moran Scatter Plot — I = {mg.I:.4f}, p = {mg.p_norm:.4f}",
                        height=460,
                    )
                    st.plotly_chart(fig_scatter, use_container_width=True)

                with tab_v3:
                    st.write("**LISA Cluster Map**")
                    conteo = pd.Series(res["clusters"]).value_counts()
                    col_leg = st.columns(5)
                    for j, (cl, color) in enumerate(CLUSTER_COLORS.items()):
                        cnt = conteo.get(cl, 0)
                        col_leg[j].markdown(
                            f"<div style='text-align:center;background:{color};color:white;"
                            f"padding:0.5rem;border-radius:6px'><b>{cl}</b><br>{cnt}</div>",
                            unsafe_allow_html=True
                        )

                    # Usar Plotly Choropleth si es posible
                    gdf_plot = gdf.to_crs("EPSG:4326")
                    fig_cl = px.choropleth_mapbox(
                        gdf_plot,
                        geojson=gdf_plot.__geo_interface__,
                        locations=gdf_plot.index,
                        color="LISA_cluster",
                        color_discrete_map=CLUSTER_COLORS,
                        mapbox_style="carto-positron",
                        center={"lat": centro_y, "lon": centro_x},
                        zoom=7, opacity=0.75,
                        hover_data=[var, "LISA_I", "LISA_p"],
                        title="Clusters LISA (HH / LL / HL / LH / NS)",
                    )
                    fig_cl.update_layout(height=480)
                    st.plotly_chart(fig_cl, use_container_width=True)

                with tab_v4:
                    st.write("**Tabla de resultados LISA**")
                    cols_show = [var, "LISA_I", "LISA_p", "LISA_cluster", "LISA_sig"]
                    cols_show = [c for c in cols_show if c in gdf.columns]
                    tabla = gdf[cols_show].copy()
                    tabla["LISA_I"] = tabla["LISA_I"].round(4)
                    tabla["LISA_p"] = tabla["LISA_p"].round(4)
                    st.dataframe(
                        tabla.style.map(
                            lambda v: f"color:{CLUSTER_COLORS.get(v,'black')};font-weight:bold"
                            if v in CLUSTER_COLORS else "",
                            subset=["LISA_cluster"]
                        ),
                        use_container_width=True,
                        height=400,
                    )

                # Exportar
                st.divider()
                col_exp1, col_exp2 = st.columns(2)
                with col_exp1:
                    excel_data = exportar_excel(gdf, res["global_res"])
                    st.download_button(
                        "📥 Exportar a Excel",
                        excel_data,
                        file_name=f"moran_{nombre.replace('.', '_')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True,
                    )
                with col_exp2:
                    geojson_str = gdf.to_crs("EPSG:4326").to_json()
                    st.download_button(
                        "📥 Exportar GeoJSON con LISA",
                        geojson_str,
                        file_name=f"lisa_{nombre.replace('.', '_')}.geojson",
                        mime="application/json",
                        use_container_width=True,
                    )

        # Tab de comparación
        if len(nombres) > 1 and len(tab_names) > len(nombres):
            with tabs[-1]:
                st.subheader("Comparación del Índice de Moran entre datasets")
                comp = st.session_state.get("resultados_comparacion", [])
                if comp:
                    df_comp = pd.DataFrame(comp)
                    df_comp["Significativo"] = df_comp["p_value"].apply(lambda p: "✅" if p < 0.05 else "❌")

                    fig_comp = go.Figure()
                    colores = ["#2ecc71" if p < 0.05 else "#95a5a6" for p in df_comp["p_value"]]
                    fig_comp.add_trace(go.Bar(
                        x=df_comp["nombre"], y=df_comp["I_moran"],
                        marker_color=colores, text=df_comp["I_moran"].round(4),
                        textposition="outside",
                    ))
                    fig_comp.add_hline(y=0, line_dash="dot", line_color="black", line_width=1)
                    fig_comp.update_layout(
                        title="Índice de Moran I por dataset (verde = p<0.05)",
                        yaxis_title="Índice de Moran I",
                        height=400,
                    )
                    st.plotly_chart(fig_comp, use_container_width=True)

                    st.dataframe(
                        df_comp[["nombre", "variable", "I_moran", "p_value", "Significativo"]].rename(columns={
                            "nombre": "Dataset", "variable": "Variable",
                            "I_moran": "I de Moran", "p_value": "p-valor"
                        }),
                        use_container_width=True,
                    )
    else:
        # Pantalla de bienvenida
        st.info("👆 Cargue al menos un dataset geográfico en el panel lateral para comenzar el análisis.")
        st.markdown("""
        ### ¿Qué es el Índice de Moran?
        El **Índice de Moran I** mide la autocorrelación espacial global de una variable geográfica.
        Responde la pregunta: *¿Las unidades con valores similares tienden a estar próximas entre sí?*

        | Valor | Interpretación |
        |-------|---------------|
        | I > 0 (p < 0.05) | Autocorrelación **positiva** — clusters espaciales |
        | I ≈ 0 (p > 0.05) | Distribución **aleatoria** — sin patrón espacial |
        | I < 0 (p < 0.05) | Autocorrelación **negativa** — dispersión espacial |

        ### Datasets de ejemplo
        - Tasa de pobreza por distritos (shapefile INEI)
        - Índice de Desarrollo Humano (IDH) por provincias
        - Cobertura de saneamiento por distritos
        - Temperatura media anual por municipios
        - Incidencia de delitos por zonas urbanas

        ### Guía rápida
        1. Suba un archivo **GeoJSON** o **CSV con lat/lon**
        2. Seleccione la **variable numérica** a analizar
        3. Configure la **matriz de pesos espaciales**
        4. Pulse **Ejecutar análisis**
        5. Explore los mapas interactivos y exporte resultados
        """)


if __name__ == "__main__":
    main()