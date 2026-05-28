# =============================================================================
# FireCast PT — Modelo CNN 1D (versión básica)
# =============================================================================
# Qué hace este script:
#   1. Carga el CSV normalizado
#   2. Separa features (X) y label (y)
#   3. Divide en train/test
#   4. Da forma a los datos para una CNN 1D
#   5. Construye y entrena una red neuronal convolucional
#   6. Evalúa el modelo
#   7. Guarda el modelo entrenado
#
# NOTA IMPORTANTE:
#   Una CNN está pensada para datos espaciales (imágenes). Aquí la aplicamos
#   a datos tabulares tratando las features de cada fila como una "señal 1D".
#   Es un ejercicio de aprendizaje — no se espera que supere al Random Forest
#   en este tipo de datos.
#
# Instalación de dependencias:
#   pip install tensorflow scikit-learn pandas
# =============================================================================

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


ARCHIVO_DATOS  = "firecast_pt_normalizado.csv"
ARCHIVO_MODELO = "modelo_firecast_cnn.keras"

# Columnas que el modelo usará como entrada
FEATURES = [
    "temp_12h",  # Air temperature at 12:00 (°C, normalized 0-1)
    "rh_12h",    # Relative humidity at 12:00 (%, normalized 0-1)
    "ws_12h",    # Wind speed at 10m at 12:00 (km/h, normalized 0-1)
    "pr_dia",    # Total daily precipitation (mm, normalized 0-1)
]

LABEL = "incendio"

# Fijar semillas para reproducibilidad
np.random.seed(42)
tf.random.set_seed(42)

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

X = df[FEATURES].values   # .values convierte a array de numpy (lo que Keras espera)
y = df[LABEL].values

print(f"\nFeatures: {X.shape[1]} columnas, {X.shape[0]} filas")

# =============================================================================
# PASO 3: DIVIDIR EN TRAIN/TEST
# =============================================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

print(f"\nTrain: {len(X_train)} filas")
print(f"Test : {len(X_test)} filas")

# =============================================================================
# PASO 4: DAR FORMA A LOS DATOS PARA LA CNN 1D
# =============================================================================
# Una CNN 1D espera datos con forma (n_filas, n_features, n_canales).
# Nuestras features son una señal 1D con 1 solo canal, así que añadimos
# una dimensión al final.
#
#   Antes:  (6777, 4)        → 6777 filas, 4 features
#   Después:(6777, 4, 1)     → 6777 filas, 4 features, 1 canal

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test  = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

n_features = X_train.shape[1]

print(f"\nForma de X_train: {X_train.shape}")

# =============================================================================
# PASO 5: CONSTRUIR LA RED NEURONAL
# =============================================================================
# Capas de la red, en orden:
#   Conv1D       → detecta patrones locales entre features vecinas
#   MaxPooling1D → reduce el tamaño quedándose con lo más importante
#   Flatten      → aplana a una sola dimensión para las capas densas
#   Dense        → capa totalmente conectada (neuronas clásicas)
#   Dropout      → apaga aleatoriamente neuronas para evitar overfitting
#   Dense(1)     → neurona final: probabilidad de incendio (0 a 1)

modelo = keras.Sequential([
    keras.Input(shape=(n_features, 1)),

    layers.Conv1D(filters=32, kernel_size=2, activation="relu", padding="same"),
    layers.MaxPooling1D(pool_size=2),

    layers.Conv1D(filters=64, kernel_size=2, activation="relu", padding="same"),

    layers.Flatten(),

    layers.Dense(64, activation="relu"),
    layers.Dropout(0.3),

    layers.Dense(1, activation="sigmoid"),   # sigmoid → salida entre 0 y 1
])

# Compilar: definir cómo aprende la red
#   optimizer = adam      → algoritmo que ajusta los pesos
#   loss = binary_crossentropy → función de error para clasificación binaria
#   metrics = accuracy    → qué medir durante el entrenamiento
modelo.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

print("\nArquitectura de la red:")
modelo.summary()

# =============================================================================
# PASO 6: ENTRENAR
# =============================================================================
# epochs       = cuántas veces la red ve todo el dataset completo
# batch_size   = cuántas filas procesa antes de ajustar los pesos
# validation_split = parte del train que se reserva para validar cada época

print("\nEntrenando la red neuronal...")
historial = modelo.fit(
    X_train, y_train,
    epochs=30,
    batch_size=32,
    validation_split=0.2,
    verbose=1,
)

# =============================================================================
# PASO 7: EVALUAR
# =============================================================================
# La red devuelve probabilidades (0 a 1). Las convertimos a 0/1 con umbral 0.5.

probabilidades = modelo.predict(X_test)
predictions = (probabilidades > 0.5).astype(int).flatten()

accuracy = accuracy_score(y_test, predictions)

matrix = confusion_matrix(y_test, predictions)
tn, fp, fn, tp = matrix.ravel()
# tn = True Negatives   → no fire day, model was right
# fp = False Positives  → no fire day, model said FIRE (false alarm)
# fn = False Negatives  → fire day, model missed it (dangerous!)
# tp = True Positives   → fire day, model detected it

total_fires = fn + tp
total_safe  = tn + fp

print("\n" + "=" * 60)
print("MODEL RESULTS (CNN)")
print("=" * 60)

print(f"\nOverall accuracy: {accuracy*100:.1f}%")
print(f"  Out of every 100 predictions, the model gets {accuracy*100:.0f} right")

print(f"\n--- Days WITH fire (test set contains {total_fires}) ---")
print(f"  Correctly detected   : {tp:>5}  ({tp/total_fires*100:.1f}%)")
print(f"  Missed (dangerous!)  : {fn:>5}  ({fn/total_fires*100:.1f}%)")

print(f"\n--- Days WITHOUT fire (test set contains {total_safe}) ---")
print(f"  Correctly flagged safe: {tn:>5}  ({tn/total_safe*100:.1f}%)")
print(f"  False alarm           : {fp:>5}  ({fp/total_safe*100:.1f}%)")

print(f"\n--- When the model predicts 'FIRE' ---")
total_alarms = tp + fp
if total_alarms > 0:
    print(f"  Correct    : {tp/total_alarms*100:.1f}% of the time")
    print(f"  Wrong      : {fp/total_alarms*100:.1f}% (false alarm)")

print(f"\n--- When the model predicts 'SAFE' ---")
total_calm = tn + fn
if total_calm > 0:
    print(f"  Correct    : {tn/total_calm*100:.1f}% of the time")
    print(f"  Wrong      : {fn/total_calm*100:.1f}% (missed fire)")

print("\n" + "=" * 60)

# =============================================================================
# PASO 8: GUARDAR EL MODELO
# =============================================================================
# Keras usa su propio formato .keras (no joblib/pkl como sklearn)

modelo.save(ARCHIVO_MODELO)
print(f"\nModel saved in: {ARCHIVO_MODELO}")