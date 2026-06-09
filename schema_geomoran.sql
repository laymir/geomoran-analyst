-- ============================================================
-- GeoMoran Analyst — Diseño de Base de Datos
-- PostgreSQL 16 + PostGIS 3.4
-- ============================================================

-- Habilitar extensión espacial
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS postgis_topology;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================================
-- TABLA: Proyectos de análisis
-- ============================================================
CREATE TABLE IF NOT EXISTS proyectos (
    id            UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    nombre        VARCHAR(200) NOT NULL,
    descripcion   TEXT,
    usuario       VARCHAR(100),
    fecha_creado  TIMESTAMPTZ DEFAULT NOW(),
    fecha_modif   TIMESTAMPTZ DEFAULT NOW(),
    activo        BOOLEAN DEFAULT TRUE
);

-- ============================================================
-- TABLA: Datasets geográficos
-- ============================================================
CREATE TABLE IF NOT EXISTS datasets (
    id              UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    proyecto_id     UUID REFERENCES proyectos(id) ON DELETE CASCADE,
    nombre          VARCHAR(200) NOT NULL,
    descripcion     TEXT,
    formato_origen  VARCHAR(20) CHECK (formato_origen IN ('shapefile','geojson','csv','xlsx','kml')),
    crs_origen      VARCHAR(50),
    n_features      INTEGER,
    variables       JSONB,           -- Metadatos de columnas disponibles
    bbox            GEOMETRY(Polygon, 4326),
    fecha_carga     TIMESTAMPTZ DEFAULT NOW(),
    hash_archivo    VARCHAR(64),     -- SHA256 para detectar duplicados
    valido          BOOLEAN DEFAULT TRUE,
    errores         TEXT[]
);

-- ============================================================
-- TABLA: Unidades espaciales (geometrías individuales)
-- ============================================================
CREATE TABLE IF NOT EXISTS unidades_espaciales (
    id            BIGSERIAL PRIMARY KEY,
    dataset_id    UUID REFERENCES datasets(id) ON DELETE CASCADE,
    fid_original  INTEGER,          -- ID del feature en el dataset original
    codigo        VARCHAR(50),       -- Código de la unidad (ubigeo, etc.)
    nombre        VARCHAR(200),
    geometry      GEOMETRY(Geometry, 4326) NOT NULL,
    atributos     JSONB,            -- Todos los atributos numéricos del feature
    area_km2      NUMERIC(12,4),
    perimetro_m   NUMERIC(12,2)
);

CREATE INDEX IF NOT EXISTS idx_unidades_geom ON unidades_espaciales USING GIST (geometry);
CREATE INDEX IF NOT EXISTS idx_unidades_dataset ON unidades_espaciales (dataset_id);

-- ============================================================
-- TABLA: Análisis de Moran (resultados globales)
-- ============================================================
CREATE TABLE IF NOT EXISTS analisis_moran (
    id                 UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    dataset_id         UUID REFERENCES datasets(id) ON DELETE CASCADE,
    variable           VARCHAR(200) NOT NULL,
    metodo_pesos       VARCHAR(20) CHECK (metodo_pesos IN ('queen','rook','knn','distance','kernel')),
    k_vecinos          INTEGER,
    distancia_umbral   NUMERIC(12,2),
    permutaciones      INTEGER DEFAULT 999,
    nivel_significancia NUMERIC(4,3) DEFAULT 0.05,

    -- Resultados globales
    i_moran            NUMERIC(10,6),
    e_i                NUMERIC(10,6),       -- Esperanza teórica
    var_i              NUMERIC(12,8),       -- Varianza
    z_score            NUMERIC(10,4),
    p_valor_normal     NUMERIC(10,6),
    p_valor_simulado   NUMERIC(10,6),
    significativo      BOOLEAN,
    interpretacion     TEXT,

    -- Metadatos del análisis
    n_unidades         INTEGER,
    n_vecinos_media    NUMERIC(6,2),
    s0                 NUMERIC(12,4),       -- Suma total de pesos
    fecha_analisis     TIMESTAMPTZ DEFAULT NOW(),
    duracion_seg       NUMERIC(8,2)
);

-- ============================================================
-- TABLA: Resultados LISA por unidad espacial
-- ============================================================
CREATE TABLE IF NOT EXISTS resultados_lisa (
    id               BIGSERIAL PRIMARY KEY,
    analisis_id      UUID REFERENCES analisis_moran(id) ON DELETE CASCADE,
    unidad_id        BIGINT REFERENCES unidades_espaciales(id) ON DELETE CASCADE,
    fid_original     INTEGER,

    -- Estadísticos LISA
    lisa_i           NUMERIC(10,6),        -- Índice LISA local
    lisa_p_sim       NUMERIC(10,6),        -- p-valor simulado
    lisa_z           NUMERIC(10,6),        -- Variable estandarizada
    lisa_lag_z       NUMERIC(10,6),        -- Lag espacial estandarizado
    cluster_tipo     VARCHAR(2) CHECK (cluster_tipo IN ('HH','LL','HL','LH','NS')),
    significativo    BOOLEAN,
    cuadrante        INTEGER               -- 1=HH, 2=LH, 3=LL, 4=HL
);

CREATE INDEX IF NOT EXISTS idx_lisa_analisis ON resultados_lisa (analisis_id);
CREATE INDEX IF NOT EXISTS idx_lisa_cluster ON resultados_lisa (cluster_tipo);

-- ============================================================
-- TABLA: Comparaciones entre datasets
-- ============================================================
CREATE TABLE IF NOT EXISTS comparaciones (
    id               UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    proyecto_id      UUID REFERENCES proyectos(id),
    nombre           VARCHAR(200),
    descripcion      TEXT,
    analisis_ids     UUID[],           -- IDs de los análisis comparados
    fecha_creado     TIMESTAMPTZ DEFAULT NOW(),
    resumen_json     JSONB             -- Tabla comparativa serializada
);

-- ============================================================
-- TABLA: Reportes generados
-- ============================================================
CREATE TABLE IF NOT EXISTS reportes (
    id               UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    analisis_id      UUID REFERENCES analisis_moran(id),
    tipo             VARCHAR(10) CHECK (tipo IN ('pdf','xlsx','geojson','csv')),
    nombre_archivo   VARCHAR(200),
    ruta_almacen     TEXT,             -- Path en S3 o filesystem
    tamano_bytes     BIGINT,
    fecha_generado   TIMESTAMPTZ DEFAULT NOW(),
    descargado       BOOLEAN DEFAULT FALSE
);

-- ============================================================
-- TABLA: Logs de auditoría
-- ============================================================
CREATE TABLE IF NOT EXISTS audit_log (
    id          BIGSERIAL PRIMARY KEY,
    usuario     VARCHAR(100),
    accion      VARCHAR(50),          -- 'carga', 'analisis', 'exportar', etc.
    entidad     VARCHAR(50),
    entidad_id  UUID,
    detalle     JSONB,
    ip_origen   INET,
    timestamp   TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================
-- VISTAS ÚTILES
-- ============================================================

-- Vista: Resumen de análisis con geometrías
CREATE OR REPLACE VIEW v_analisis_completo AS
SELECT
    am.id AS analisis_id,
    d.nombre AS dataset,
    am.variable,
    am.metodo_pesos,
    am.i_moran,
    am.z_score,
    am.p_valor_normal,
    am.significativo,
    am.interpretacion,
    am.n_unidades,
    am.fecha_analisis,
    p.nombre AS proyecto
FROM analisis_moran am
JOIN datasets d ON d.id = am.dataset_id
LEFT JOIN proyectos p ON p.id = d.proyecto_id
ORDER BY am.fecha_analisis DESC;

-- Vista: Distribución de clusters LISA
CREATE OR REPLACE VIEW v_distribucion_clusters AS
SELECT
    am.id AS analisis_id,
    d.nombre AS dataset,
    am.variable,
    rl.cluster_tipo,
    COUNT(*) AS n,
    ROUND(COUNT(*) * 100.0 / am.n_unidades, 2) AS porcentaje
FROM resultados_lisa rl
JOIN analisis_moran am ON am.id = rl.analisis_id
JOIN datasets d ON d.id = am.dataset_id
GROUP BY am.id, d.nombre, am.variable, rl.cluster_tipo, am.n_unidades
ORDER BY am.id, rl.cluster_tipo;

-- ============================================================
-- FUNCIONES ÚTILES
-- ============================================================

-- Función: Obtener vecinos Queen de una unidad
CREATE OR REPLACE FUNCTION get_vecinos_queen(p_unidad_id BIGINT)
RETURNS TABLE(vecino_id BIGINT, codigo VARCHAR, nombre VARCHAR) AS $$
    SELECT u2.id, u2.codigo, u2.nombre
    FROM unidades_espaciales u1
    JOIN unidades_espaciales u2 ON (
        u1.dataset_id = u2.dataset_id
        AND u1.id != u2.id
        AND ST_Touches(u1.geometry, u2.geometry)
    )
    WHERE u1.id = p_unidad_id;
$$ LANGUAGE SQL;

-- ============================================================
-- DATOS DE EJEMPLO (seed)
-- ============================================================

INSERT INTO proyectos (nombre, descripcion, usuario) VALUES
    ('Análisis IDH Perú 2023', 'Evaluación espacial del IDH por provincias del Perú', 'investigador1'),
    ('Pobreza Multidimensional Altiplano', 'Clusters de pobreza en la región del altiplano peruano', 'investigador2')
ON CONFLICT DO NOTHING;

-- ============================================================
-- ÍNDICES DE RENDIMIENTO
-- ============================================================
CREATE INDEX IF NOT EXISTS idx_analisis_dataset ON analisis_moran (dataset_id);
CREATE INDEX IF NOT EXISTS idx_analisis_fecha ON analisis_moran (fecha_analisis DESC);
CREATE INDEX IF NOT EXISTS idx_datasets_proyecto ON datasets (proyecto_id);
CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log (timestamp DESC);
