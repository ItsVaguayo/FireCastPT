"""
FireCast PT — Añadir lag features meteorológicos (últimos 3 días)
=================================================================
Ejecutar en tu máquina local donde la API de Open-Meteo funciona.

Requisitos:
    pip install pandas requests tqdm openpyxl

Uso:
    Coloca este script en la misma carpeta que firecast_pt_dataset.csv
    y el Excel original, luego ejecuta:
        python agregar_lags.py

Columnas nuevas que genera:
    temp_lag1, temp_lag2, temp_lag3   (temperatura a las 12h, días -1, -2, -3)
    rh_lag1,   rh_lag2,   rh_lag3    (humedad relativa a las 12h)
    ws_lag1,   ws_lag2,   ws_lag3    (viento a las 12h)
    pr_lag1,   pr_lag2,   pr_lag3    (precipitación total del día)
"""

import math
import time
from datetime import date, timedelta

import pandas as pd
import requests
from tqdm import tqdm

# =============================================================================
# PARÁMETROS — ajusta si hace falta
# =============================================================================

from pathlib import Path

# Carpeta donde está el script
BASE = Path(__file__).parent

ARCHIVO_ENTRADA  = BASE / "firecast_pt_dataset.csv"
ARCHIVO_EXCEL    = BASE / "Registos_Incendios_SGIF_2021_2025.xlsx"
ARCHIVO_SALIDA   = BASE / "firecast_pt_con_lags.csv"
PAUSA_API        = 0.12   # segundos entre llamadas (respeta el rate limit)
MAX_REINTENTOS   = 3

# =============================================================================
# PASO 1 — Recuperar el año real de cada fila positiva desde el Excel
# =============================================================================
# El CSV generado no tiene columna Año. La recuperamos del Excel para los
# positivos (incendio=1). Para los negativos, el script de generación usó
# año ± 1 respecto al incendio original, así que los reconstruimos también.

print("Cargando datos...")
df = pd.read_csv(ARCHIVO_ENTRADA, encoding="utf-8-sig")
print(f"  Filas en el dataset: {len(df)}")

# Cargar el Excel para obtener fechas reales
excel = pd.read_excel(ARCHIVO_EXCEL, sheet_name="SGIF_2021_2025")
excel = excel.dropna(subset=["Latitude", "Longitude", "FWI"])
excel["_fecha"] = pd.to_datetime(excel["DataHoraAlerta"])
excel["_año"]   = excel["_fecha"].dt.year
excel["_mes"]   = excel["_fecha"].dt.month
excel["_dia"]   = excel["_fecha"].dt.day
excel["_lat_r"] = excel["Latitude"].round(3)
excel["_lon_r"] = excel["Longitude"].round(3)

# Índice: (lat_r, lon_r, mes, dia) → año
mapa_año = {}
for _, r in excel.iterrows():
    clave = (r["_lat_r"], r["_lon_r"], int(r["_mes"]), int(r["_dia"]))
    mapa_año[clave] = int(r["_año"])

# Asignar año a cada fila
def inferir_año(row):
    clave = (round(row["Latitude"], 3), round(row["Longitude"], 3),
             int(row["Mes"]), int(row["Dia"]))
    if clave in mapa_año:
        return mapa_año[clave]
    # Para negativos: buscamos cualquier año del mismo punto con mismo mes
    for año_candidato in [2021, 2022, 2023, 2024, 2025, 2020, 2019]:
        try:
            date(año_candidato, int(row["Mes"]), int(row["Dia"]))
            return año_candidato
        except ValueError:
            continue
    return 2023  # fallback

df["_año"] = df.apply(inferir_año, axis=1)
print(f"  Años asignados. Distribución:")
print(df["_año"].value_counts().sort_index().to_string())

# =============================================================================
# PASO 2 — Caché de meteo por rango corto (igual que el script original)
# =============================================================================

cache_meteo = {}  # clave: (lat_r, lon_r, fecha) → {temp, rh, ws, pr}


def obtener_meteo_rango(lat, lon, fecha_inicio, fecha_fin):
    """
    Descarga datos horarios de Open-Meteo para un rango de fechas.
    Devuelve dict: {fecha: {temp, rh, ws, pr}}
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude":        lat,
        "longitude":       lon,
        "start_date":      fecha_inicio.isoformat(),
        "end_date":        fecha_fin.isoformat(),
        "hourly":          "temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation",
        "timezone":        "Europe/Lisbon",
        "wind_speed_unit": "kmh",
    }

    for intento in range(MAX_REINTENTOS):
        try:
            resp = requests.get(url, params=params, timeout=25)
            resp.raise_for_status()
            h = resp.json()["hourly"]

            n_dias    = (fecha_fin - fecha_inicio).days + 1
            resultado = {}

            for d in range(n_dias):
                ini  = d * 24
                fin  = ini + 24
                fdia = fecha_inicio + timedelta(days=d)

                temp_h = h["temperature_2m"][ini:fin]
                rh_h   = h["relative_humidity_2m"][ini:fin]
                ws_h   = h["wind_speed_10m"][ini:fin]
                pr_h   = h["precipitation"][ini:fin]

                resultado[fdia] = {
                    "temp": temp_h[12] if len(temp_h) > 12 and temp_h[12] is not None else float("nan"),
                    "rh":   rh_h[12]   if len(rh_h)   > 12 and rh_h[12]   is not None else float("nan"),
                    "ws":   ws_h[12]   if len(ws_h)    > 12 and ws_h[12]   is not None else float("nan"),
                    "pr":   sum(v for v in pr_h if v is not None),
                }

            time.sleep(PAUSA_API)
            return resultado

        except Exception as e:
            if intento < MAX_REINTENTOS - 1:
                time.sleep(2 ** intento)
            else:
                print(f"\n  ⚠ Error API ({lat:.2f},{lon:.2f},{fecha_inicio}→{fecha_fin}): {e}")
                return None


def obtener_lags(lat, lon, fecha_objetivo, n_lags=3):
    """
    Devuelve los datos de los n_lags días anteriores a fecha_objetivo.
    Usa caché para evitar llamadas duplicadas.
    """
    lat_r = round(lat, 2)
    lon_r = round(lon, 2)

    # Fechas que necesitamos
    fechas_necesarias = [fecha_objetivo - timedelta(days=lag) for lag in range(1, n_lags + 1)]

    # Cuáles faltan en caché
    fechas_faltantes = [f for f in fechas_necesarias if (lat_r, lon_r, f) not in cache_meteo]

    if fechas_faltantes:
        # Descargamos de golpe todas las faltantes en un rango continuo
        f_min = min(fechas_faltantes)
        f_max = max(fechas_faltantes)

        datos = obtener_meteo_rango(lat, lon, f_min, f_max)

        if datos:
            for f, vals in datos.items():
                cache_meteo[(lat_r, lon_r, f)] = vals
        else:
            # Marcar como fallido para no reintentar
            for f in fechas_faltantes:
                cache_meteo[(lat_r, lon_r, f)] = None

    # Recuperar de caché
    lags = {}
    for lag in range(1, n_lags + 1):
        f   = fecha_objetivo - timedelta(days=lag)
        val = cache_meteo.get((lat_r, lon_r, f))
        if val is None:
            lags[lag] = {"temp": float("nan"), "rh": float("nan"),
                         "ws":   float("nan"), "pr":  float("nan")}
        else:
            lags[lag] = val

    return lags


# =============================================================================
# PASO 3 — Añadir columnas de lag al dataset
# =============================================================================

# Inicializar columnas
for lag in range(1, 4):
    for var in ["temp", "rh", "ws", "pr"]:
        df[f"{var}_lag{lag}"] = float("nan")

print(f"\nDescargando lags para {len(df)} filas...")
print("(Las llamadas a la API se agrupan por rango de fechas para cada punto geográfico)\n")

errores = 0

for idx, row in tqdm(df.iterrows(), total=len(df)):
    lat = row["Latitude"]
    lon = row["Longitude"]
    mes = int(row["Mes"])
    dia = int(row["Dia"])
    año = int(row["_año"])

    try:
        fecha_objetivo = date(año, mes, dia)
    except ValueError:
        errores += 1
        continue

    lags = obtener_lags(lat, lon, fecha_objetivo, n_lags=3)

    for lag_n, vals in lags.items():
        df.at[idx, f"temp_lag{lag_n}"] = round(vals["temp"], 2) if not math.isnan(vals["temp"]) else float("nan")
        df.at[idx, f"rh_lag{lag_n}"]   = round(vals["rh"],   2) if not math.isnan(vals["rh"])   else float("nan")
        df.at[idx, f"ws_lag{lag_n}"]   = round(vals["ws"],   2) if not math.isnan(vals["ws"])   else float("nan")
        df.at[idx, f"pr_lag{lag_n}"]   = round(vals["pr"],   2) if not math.isnan(vals["pr"])   else float("nan")

# Eliminar columna auxiliar
df.drop(columns=["_año"], inplace=True)

# Guardar
df.to_csv(ARCHIVO_SALIDA, index=False, encoding="utf-8-sig")

print(f"\n✓ Dataset guardado: {ARCHIVO_SALIDA}")
print(f"  Total filas    : {len(df)}")
print(f"  Columnas totales: {len(df.columns)}")
print(f"  Errores de fecha: {errores}")
print(f"  Filas con NaN en temp_lag1: {df['temp_lag1'].isna().sum()}")

print(f"\nColumnas finales:")
print(df.columns.tolist())

print(f"\nPrimeras 3 filas (columnas de lag):")
cols_lag = [c for c in df.columns if "lag" in c]
print(df[["Mes", "Dia", "temp_12h", "rh_12h"] + cols_lag].head(3).to_string())
