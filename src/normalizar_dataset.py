# =============================================================================
# FireCast PT — Normalización del dataset
# =============================================================================
# Qué hace este script:
#   1. Lee el CSV generado por el script anterior
#   2. Normaliza las variables numéricas a rango 0-1 (Min-Max)
#   3. Codifica Mes y Dia con seno/coseno (variables cíclicas)
#   4. Guarda el dataset normalizado y los rangos usados (para predecir luego)
#
# Por qué guardamos los rangos:
#   Cuando el modelo reciba datos nuevos para predecir, hay que normalizarlos
#   con los MISMOS mínimos y máximos del entrenamiento. Si no, los valores
#   no estarían en la misma escala y las predicciones serían incorrectas.
#
# Instalación de dependencias:
#   pip install pandas numpy
# =============================================================================

import json
import numpy as np
import pandas as pd

# =============================================================================
# PARÁMETROS
# =============================================================================

ARCHIVO_ENTRADA = "firecast_pt_dataset.csv"
ARCHIVO_SALIDA  = "firecast_pt_normalizado.csv"
ARCHIVO_RANGOS  = "rangos_normalizacion.json"   # guarda min/max de cada columna

# Columnas que se normalizan con Min-Max (0-1)
COLUMNAS_MINMAX = [
    "temp_12h",   # temperatura °C
    "rh_12h",     # humedad relativa %
    "ws_12h",     # viento km/h
    "pr_dia",     # precipitación mm
    "FFMC",       # Fine Fuel Moisture Code
    "ISI",        # Initial Spread Index
    "DMC",        # Duff Moisture Code
    "DC",         # Drought Code
    "BUI",        # Build-Up Index
    "FWI",        # Fire Weather Index
    "DSR",        # Daily Severity Rating
]

# Columnas cíclicas (se codifican con seno y coseno)
# periodo = cuántos valores distintos tiene la variable
COLUMNAS_CICLICAS = {
    "Mes": 12,    # 1 a 12
    "Dia": 31,    # 1 a 31
}

# Columnas que se quedan tal cual (no se tocan)
COLUMNAS_SIN_CAMBIOS = [
    "Latitude",
    "Longitude",
    "Distrito",
    "Concelho",
    "incendio",   # el label NUNCA se normaliza
]

# =============================================================================
# CARGAR DATOS
# =============================================================================

print("Cargando dataset...")
df = pd.read_csv(ARCHIVO_ENTRADA)
print(f"  Filas cargadas: {len(df)}")
print(f"  Columnas: {df.columns.tolist()}")

# =============================================================================
# NORMALIZACIÓN MIN-MAX (0-1)
# =============================================================================
# Fórmula: valor_normalizado = (valor - mínimo) / (máximo - mínimo)
# Si mínimo == máximo (columna constante), dejamos todo a 0 para evitar /0

print("\nNormalizando columnas numéricas (Min-Max 0-1)...")

rangos = {}  # guardamos min y max de cada columna para uso futuro

for col in COLUMNAS_MINMAX:
    if col not in df.columns:
        print(f"  ⚠ Columna '{col}' no encontrada, saltando")
        continue

    minimo = df[col].min()
    maximo = df[col].max()
    rango  = maximo - minimo

    rangos[col] = {"min": minimo, "max": maximo}

    if rango == 0:
        df[col] = 0.0
        print(f"  {col}: constante ({minimo}) → puesto a 0")
    else:
        df[col] = (df[col] - minimo) / rango
        print(f"  {col}: [{minimo:.2f}, {maximo:.2f}] → [0, 1]")

# =============================================================================
# CODIFICACIÓN CÍCLICA (seno y coseno)
# =============================================================================
# Fórmula:
#   col_sin = sin(2π × valor / periodo)
#   col_cos = cos(2π × valor / periodo)
#
# Ejemplo para Mes:
#   Enero  (1) → sin=0.50,  cos=0.87
#   Julio  (7) → sin=-0.50, cos=-0.87
#   Dic   (12) → sin=-0.50, cos=0.87  ← cerca de enero, correcto

print("\nCodificando variables cíclicas (sin/cos)...")

for col, periodo in COLUMNAS_CICLICAS.items():
    if col not in df.columns:
        print(f"  ⚠ Columna '{col}' no encontrada, saltando")
        continue

    df[f"{col.lower()}_sin"] = np.sin(2 * np.pi * df[col] / periodo).round(6)
    df[f"{col.lower()}_cos"] = np.cos(2 * np.pi * df[col] / periodo).round(6)

    print(f"  {col} → {col.lower()}_sin, {col.lower()}_cos")

# Eliminar las columnas originales de Mes y Dia (ya están codificadas)
df = df.drop(columns=[col for col in COLUMNAS_CICLICAS if col in df.columns])

# =============================================================================
# REORDENAR COLUMNAS
# =============================================================================
# Orden final: sin_cambios | min_max | ciclicas | label

cols_ciclicas = [f"{col.lower()}_sin" for col in COLUMNAS_CICLICAS] + \
                [f"{col.lower()}_cos" for col in COLUMNAS_CICLICAS]

cols_finales = (
    COLUMNAS_SIN_CAMBIOS +
    [c for c in COLUMNAS_MINMAX if c in df.columns] +
    [c for c in cols_ciclicas if c in df.columns]
)

# Asegurarse de que incendio queda al final
cols_finales.remove("incendio")
cols_finales.append("incendio")

df = df[cols_finales]

# =============================================================================
# GUARDAR DATASET NORMALIZADO
# =============================================================================

df.to_csv(ARCHIVO_SALIDA, index=False, encoding="utf-8-sig")
print(f"\nDataset normalizado guardado en: {ARCHIVO_SALIDA}")

# =============================================================================
# GUARDAR RANGOS (para normalizar datos nuevos al predecir)
# =============================================================================

with open(ARCHIVO_RANGOS, "w") as f:
    json.dump(rangos, f, indent=2)
print(f"Rangos guardados en: {ARCHIVO_RANGOS}")

# =============================================================================
# RESUMEN FINAL
# =============================================================================

print("\n--- Resumen del dataset normalizado ---")
print(f"  Filas      : {len(df)}")
print(f"  Columnas   : {len(df.columns)}")
print(f"  Positivos  : {df['incendio'].sum()}")
print(f"  Negativos  : {(df['incendio'] == 0).sum()}")
print(f"\n  Columnas finales:")
for col in df.columns:
    print(f"    {col}: min={df[col].min() if df[col].dtype != 'O' else 'texto'}, "
          f"max={df[col].max() if df[col].dtype != 'O' else 'texto'}")
