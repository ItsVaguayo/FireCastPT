# =============================================================================
# FireCast PT — Modelo Random Forest v1 (versión básica)
# =============================================================================
# Qué hace este script:
#   1. Carga el CSV normalizado
#   2. Separa features (X) y label (y)
#   3. Divide en train/test (80% / 20%)
#   4. Entrena un Random Forest
#   5. Evalúa el modelo con accuracy, matriz de confusión y classification report
#   6. Guarda el modelo entrenado en disco
#
# Instalación de dependencias:
#   pip install scikit-learn pandas joblib
# =============================================================================

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# =============================================================================
# PARÁMETROS
# =============================================================================

ARCHIVO_DATOS  = "firecast_pt_normalizado.csv"
ARCHIVO_MODELO = "modelo_firecast.pkl"

# Columnas que el modelo usará como entrada
FEATURES = [
    "temp_12h",  # Air temperature at 12:00 (°C, normalized 0-1)
    "rh_12h",    # Relative humidity at 12:00 (%, normalized 0-1)
    "ws_12h",    # Wind speed at 10m at 12:00 (km/h, normalized 0-1)
    "pr_dia",    # Total daily precipitation (mm, normalized 0-1)
    "FWI",       # Fire Weather Index — overall fire danger rating (normalized 0-1)
    "DSR",       # Daily Severity Rating — exponential transform of FWI for averaging (normalized 0-1)
]

LABEL = "incendio"

# =============================================================================
# PASO 1: CARGAR DATOS
# =============================================================================

print("Cargando datos...")
df = pd.read_csv(ARCHIVO_DATOS)
print(f"  Filas totales: {len(df)}")
print(f"  Incendios    : {df[LABEL].sum()}")
print(f"  No incendios : {(df[LABEL] == 0).sum()}")

# =============================================================================
# PASO 2: SEPARAR FEATURES Y LABEL
# =============================================================================
# X = features (lo que el modelo ve para hacer la predicción)
# y = label   (lo que el modelo debe predecir)

X = df[FEATURES]   # 11 columnas
y = df[LABEL]      # 1 columna (0 o 1)

print(f"\nFeatures: {X.shape[1]} columnas, {X.shape[0]} filas")

# =============================================================================
# PASO 3: DIVIDIR EN TRAIN/TEST
# =============================================================================
# 80% para entrenar, 20% para evaluar.
# El modelo NUNCA ve los datos de test durante el entrenamiento.
# random_state=42 fija la aleatoriedad para que los resultados sean reproducibles.
# stratify=y mantiene la proporción de positivos/negativos en ambos conjuntos.

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

print(f"\nTrain: {len(X_train)} filas")
print(f"Test : {len(X_test)} filas")

# =============================================================================
# PASO 4: ENTRENAR EL MODELO
# =============================================================================
# n_estimators = número de árboles del bosque (más árboles = más preciso pero más lento)
# random_state = semilla para reproducibilidad
# n_jobs=-1    = usa todos los núcleos del procesador

print("\nEntrenando Random Forest...")
modelo = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
)
modelo.fit(X_train, y_train)
print("  Entrenamiento terminado")

# =============================================================================
# PASO 5: EVALUAR
# =============================================================================
# El modelo predice los labels de X_test (que nunca había visto).
# Comparamos las predicciones con los labels reales y_test.

predicciones = modelo.predict(X_test)

accuracy = accuracy_score(y_test, predicciones)
print(f"\nAccuracy: {accuracy:.4f}  ({accuracy*100:.2f}%)")

print("\nMatriz de confusión:")
print("                Predicho 0   Predicho 1")
matriz = confusion_matrix(y_test, predicciones)
print(f"  Real 0      {matriz[0][0]:>10}   {matriz[0][1]:>10}")
print(f"  Real 1      {matriz[1][0]:>10}   {matriz[1][1]:>10}")

print("\nClassification report:")
print(classification_report(y_test, predicciones, target_names=["No incendio", "Incendio"]))

# =============================================================================
# PASO 6: IMPORTANCIA DE LAS FEATURES
# =============================================================================
# Qué columnas pesan más en las decisiones del modelo

print("Importancia de cada feature:")
importancias = pd.Series(modelo.feature_importances_, index=FEATURES).sort_values(ascending=False)
for nombre, importancia in importancias.items():
    print(f"  {nombre:12} {importancia:.4f}")

# =============================================================================
# PASO 7: GUARDAR EL MODELO
# =============================================================================
# joblib es más eficiente que pickle para modelos de sklearn

joblib.dump(modelo, ARCHIVO_MODELO)
print(f"\nModelo guardado en: {ARCHIVO_MODELO}")