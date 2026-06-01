# =============================================================================
# FireCast PT — Modelo Random Forest v2
#Diferencia respecto a la v1:
#   En lugar de fijar n_estimators a mano, Grid Search prueba varias
#   combinaciones de hiperparámetros y elige automáticamente la mejor
#   usando 5-fold cross-validation.
 
# Instalación de dependencias:
#   pip install scikit-learn pandas joblib
# =============================================================================

import joblib
import pandas as pd
 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
 
 
ARCHIVO_DATOS  = "firecast_pt_normalizado.csv"
ARCHIVO_MODELO = "modelo_firecast.pkl"
 
# Columnas que el modelo usará como entrada
FEATURES = [
    "temp_12h",  # Air temperature at 12:00 (°C, normalized 0-1)
    "rh_12h",    # Relative humidity at 12:00 (%, normalized 0-1)
    "ws_12h",    # Wind speed at 10m at 12:00 (km/h, normalized 0-1)
    "pr_dia",    # Total daily precipitation (mm, normalized 0-1)
]
 
LABEL = "incendio"
 
 
# =============================================================================
# CARGAR DATOS
# =============================================================================
 
print("Cargando datos...")
df = pd.read_csv(ARCHIVO_DATOS)
print(f"  Filas totales: {len(df)}")
print(f"  Incendios    : {df[LABEL].sum()}")
print(f"  No incendios : {(df[LABEL] == 0).sum()}")
 
# =============================================================================
# SEPARAR FEATURES Y LABEL
# =============================================================================
# X = features (lo que el modelo ve para hacer la predicción)
# y = label   (lo que el modelo debe predecir)
 
X = df[FEATURES]
y = df[LABEL]
 
print(f"\nFeatures: {X.shape[1]} columnas, {X.shape[0]} filas")
 
# =============================================================================
# DIVIDIR EN TRAIN/TEST
# =============================================================================
# 80% para entrenar, 20% para evaluar.
# El modelo NUNCA ve los datos de test durante el entrenamiento.
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
# GRID SEARCH + ENTRENAR
# =============================================================================
# Grid Search prueba TODAS las combinaciones de la rejilla con 5-fold CV
# y se queda con la que mejor accuracy media obtiene.
#
# Combinaciones: 3 × 3 × 3 = 27
# Entrenamientos: 27 × 5 folds = 135
 
param_grid = {
    "n_estimators":      [100, 200, 300],   # número de árboles
    "max_depth":         [10, 20, None],    # profundidad máxima (None = sin límite)
    "min_samples_split": [2, 5, 10],        # mínimo de muestras para dividir un nodo
}
 
# Modelo base: sin hiperparámetros fijos, los pone Grid Search
modelo_base = RandomForestClassifier(random_state=42, n_jobs=-1)
 
print("\nBuscando los mejores hiperparámetros con Grid Search...")
print("(esto puede tardar varios minutos)")
 
grid = GridSearchCV(
    estimator=modelo_base,
    param_grid=param_grid,
    cv=5,                  # 5-fold cross-validation
    scoring="accuracy",    # métrica que optimiza
    n_jobs=-1,             # usa todos los núcleos
    verbose=2,             # muestra el progreso
)
grid.fit(X_train, y_train)
 
# El mejor modelo encontrado, ya entrenado
modelo = grid.best_estimator_
 
print(f"\nMejores hiperparámetros: {grid.best_params_}")
print(f"Mejor accuracy en cross-validation: {grid.best_score_:.4f}")
 
# =============================================================================
# EVALUAR
# =============================================================================
# El modelo predice los labels de X_test (que nunca había visto).
# Comparamos las predicciones con los labels reales y_test.
 
predictions = modelo.predict(X_test)
 
accuracy = accuracy_score(y_test, predictions)
 
# Confusion matrix: unpack into 4 named variables for clarity
matrix = confusion_matrix(y_test, predictions)
tn, fp, fn, tp = matrix.ravel()
# tn = True Negatives   → no fire day, model was right
# fp = False Positives  → no fire day, model said FIRE (false alarm)
# fn = False Negatives  → fire day, model missed it (dangerous!)
# tp = True Positives   → fire day, model detected it
 
total       = tn + fp + fn + tp
total_fires = fn + tp
total_safe  = tn + fp
 
print("\n" + "=" * 60)
print("MODEL RESULTS")
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
# IMPORTANCIA DE LAS FEATURES
# =============================================================================
# Qué columnas pesan más en las decisiones del modelo
 
print("Importance of each feature:")
importancias = pd.Series(modelo.feature_importances_, index=FEATURES).sort_values(ascending=False)
for nombre, importancia in importancias.items():
    print(f"  {nombre:12} {importancia:.4f}")
 
# =============================================================================
# GUARDAR EL MODELO
# =============================================================================
 
joblib.dump(modelo, ARCHIVO_MODELO)
print(f"\nModel saved in: {ARCHIVO_MODELO}")
 
