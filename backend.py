"""
GeoMoran Analyst — Backend FastAPI
Ejecutar: uvicorn backend:app --reload --port 8000
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
import geopandas as gpd
import pandas as pd
import numpy as np
import libpysal
from libpysal.weights import Queen, Rook, KNN
from esda.moran import Moran, Moran_Local
import io, json, warnings, os
warnings.filterwarnings("ignore")

app = FastAPI(title="GeoMoran Analyst API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Servir archivos estáticos (el index.html)
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def root():
    return FileResponse("index.html")

# ── Almacén en memoria de datasets cargados ──
DATASETS: dict[str, gpd.GeoDataFrame] = {}

@app.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """Recibe un archivo geográfico y retorna sus columnas numéricas."""
    content = await file.read()
    nombre = file.filename.lower()
    try:
        if nombre.endswith(".geojson") or nombre.endswith(".json"):
            gdf = gpd.read_file(io.BytesIO(content))
        elif nombre.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(content))
            lat = next((c for c in df.columns if c.lower() in ("lat","latitude","latitud")), None)
            lon = next((c for c in df.columns if c.lower() in ("lon","lng","longitude","longitud")), None)
            if not lat or not lon:
                raise HTTPException(400, "CSV necesita columnas latitude/longitude")
            gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[lon], df[lat]), crs="EPSG:4326")
        else:
            raise HTTPException(400, f"Formato no soportado: {nombre}")

        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")
        gdf = gdf.to_crs("EPSG:4326").dropna(subset=["geometry"])
        gdf["geometry"] = gdf["geometry"].buffer(0)
        gdf = gdf.reset_index(drop=True)
        DATASETS[file.filename] = gdf

        num_cols = gdf.select_dtypes(include="number").columns.tolist()
        preview = gdf.drop(columns="geometry").head(5).to_dict(orient="records")
        # Serializar valores numpy
        for row in preview:
            for k, v in row.items():
                if hasattr(v, "item"):
                    row[k] = v.item()
        return {
            "filename": file.filename,
            "n_features": len(gdf),
            "numeric_columns": num_cols,
            "preview": preview,
        }
    except Exception as e:
        raise HTTPException(400, str(e))


@app.post("/analyze")
async def analyze(body: dict):
    """
    Calcula Moran Global + LISA.
    Body: { filename, variable, method, k, permutations, significance }
    """
    fname       = body.get("filename")
    variable    = body.get("variable")
    method      = body.get("method", "queen")
    k           = int(body.get("k", 5))
    perm        = int(body.get("permutations", 999))
    sig         = float(body.get("significance", 0.05))

    if fname not in DATASETS:
        raise HTTPException(404, f"Dataset '{fname}' no encontrado. Sube el archivo primero.")

    gdf = DATASETS[fname].copy()
    if variable not in gdf.columns:
        raise HTTPException(400, f"Variable '{variable}' no existe en el dataset.")

    y = gdf[variable].astype(float).values

    # Matriz de pesos
    if method == "queen":
        w = Queen.from_dataframe(gdf, silence_warnings=True)
    elif method == "rook":
        w = Rook.from_dataframe(gdf, silence_warnings=True)
    else:
        w = KNN.from_dataframe(gdf, k=k, silence_warnings=True)
    w.transform = "R"

    # Moran Global
    mg = Moran(y, w, permutations=perm)

    # Moran Local LISA
    ml = Moran_Local(y, w, permutations=perm)
    z    = (y - y.mean()) / (y.std() + 1e-9)
    lag  = libpysal.weights.lag_spatial(w, y)
    lagz = (lag - lag.mean()) / (lag.std() + 1e-9)
    sig_mask = ml.p_sim < sig
    clusters = np.where(sig_mask,
        np.where(z > 0, np.where(lagz > 0, "HH", "HL"),
                        np.where(lagz > 0, "LH", "LL")),
        "NS")

    gdf["LISA_I"]       = ml.Is
    gdf["LISA_p"]       = ml.p_sim
    gdf["LISA_z"]       = z
    gdf["LISA_lag"]     = lagz
    gdf["LISA_cluster"] = clusters

    # GeoJSON de salida (solo geom + cols relevantes)
    cols_out = [variable, "LISA_I", "LISA_p", "LISA_cluster"]
    # añadir columnas de texto si existen
    text_cols = gdf.select_dtypes(include="object").columns.tolist()
    for tc in text_cols:
        if tc not in cols_out:
            cols_out.append(tc)
    gdf_out = gdf[cols_out + ["geometry"]].to_crs("EPSG:4326")
    geojson_str = gdf_out.to_json()

    # Scatter data
    scatter = [
        {"x": float(z[i]), "y": float(lagz[i]),
         "cluster": str(clusters[i]), "val": float(y[i])}
        for i in range(len(y))
    ]

    # Conteo clusters
    unique, counts = np.unique(clusters, return_counts=True)
    cluster_counts = {str(k): int(v) for k, v in zip(unique, counts)}

    return {
        "global": {
            "I":       float(mg.I),
            "EI":      float(mg.EI),
            "z_norm":  float(mg.z_norm),
            "p_norm":  float(mg.p_norm),
            "p_sim":   float(mg.p_sim),
            "significant": bool(mg.p_norm < sig),
        },
        "geojson":        json.loads(geojson_str),
        "scatter":        scatter,
        "cluster_counts": cluster_counts,
        "variable":       variable,
        "n":              int(len(y)),
    }


@app.post("/export/excel")
async def export_excel(body: dict):
    fname    = body.get("filename")
    variable = body.get("variable")
    if fname not in DATASETS:
        raise HTTPException(404, "Dataset no encontrado")
    gdf = DATASETS[fname]
    cols = [c for c in gdf.columns if c != "geometry"]
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        pd.DataFrame([body.get("global_result", {})]).to_excel(writer, sheet_name="Moran Global", index=False)
        gdf[cols].to_excel(writer, sheet_name="LISA por unidad", index=False)
        if "LISA_cluster" in gdf.columns:
            res = gdf.groupby("LISA_cluster").size().reset_index(name="N")
            res["%"] = (res["N"] / len(gdf) * 100).round(2)
            res.to_excel(writer, sheet_name="Resumen clusters", index=False)
    output.seek(0)
    return StreamingResponse(
        output,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename=moran_{fname}.xlsx"},
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=True)
