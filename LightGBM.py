import pandas as pd
import lightgbm as lgb
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer
import time
import joblib

# Configuración
RANDOM_STATE = 42
N_JOBS = -1  # Usar todos los núcleos disponibles

# Rutas de archivos
INPUT_FILE = 'D:/Desktop/Tesis/Data/Datos_previo_imputar/Datos_previo_imputar.psv'
OUTPUT_PSV = 'D:/Desktop/Tesis/Data/Imputaciones/LightGBM_Iterative.psv'
OUTPUT_EXCEL = OUTPUT_PSV.replace('.psv', '.xlsx')
OUTPUT_PSV_STD = OUTPUT_PSV.replace('.psv', '_estandarizado.psv')
OUTPUT_EXCEL_STD = OUTPUT_PSV_STD.replace('.psv', '.xlsx')
MODEL_SAVE_PATH = 'D:/Desktop/Tesis/Data/Imputaciones/LightGBM_iterative_imputer_model.joblib'

# Cargar y procesar datos
df = pd.read_csv(INPUT_FILE, sep='|', index_col=0)
df.columns = df.columns.str.replace('/', '_')
numeric_cols = df.iloc[:, 0:58].columns

# Función para evaluar la imputación
def imputation_scorer(estimator, X, y=None):

    X_ = X.copy()
    mse_scores = [] #Para almacenar errores
    
    for col_idx in range(X.shape[1]): #Itera sobre cada columna
        col_data = X[:, col_idx]      #Datos de la columna actual
        non_missing = ~np.isnan(col_data)  #Valores reales de la columna
        
        sample_idx = np.random.choice(
            np.where(non_missing)[0],  #Encuentra los índices de los valores no faltantes
            size=int(0.1 * sum(non_missing)),  replace=False) # #Muestra el 10% de los valores no faltantes, no repite datos
        
        true_values = X_[sample_idx, col_idx].copy()
        X_[sample_idx, col_idx] = np.nan #Nan artificiales en el 10% de los datos
        
        # Imputar y calcular MSE
        imputed = estimator.transform(X_)[sample_idx, col_idx]  #Imputamos los Nan
        mse = np.mean((true_values - imputed) ** 2) # Calcular el error cuadrático medio
        mse_scores.append(mse)
        
        # Restaurar valores originales
        X_[sample_idx, col_idx] = true_values
    
    return -np.mean(mse_scores) if mse_scores else -np.inf  # Negativo para maximizar

# Espacio de búsqueda de hiperparámetros
param_dist = {
    'estimator__n_estimators': [200, 300, 400, 500, 600],
    'estimator__learning_rate': [0.01, 0.03, 0.1],
    'estimator__max_depth': [5, 7, 10],
    'estimator__subsample': [0.8, 0.9, 1],
    'estimator__colsample_bytree': [0.7, 0.8, 0.9, 1],
}

# Configurar el estimador base
base_estimator = lgb.LGBMRegressor(
    random_state=RANDOM_STATE,
    n_jobs=1,  # Se paraleliza a nivel de HalvingRandomSearchCV
    verbosity=-1, # Silencia la salida de LightGBM
    force_col_wise=True,  # Evita problemas de memoria con columnas dispersas
)

# Configurar IterativeImputer
imputer = IterativeImputer(
    estimator=base_estimator,
    random_state=RANDOM_STATE,
    verbose=2 # Mostrar detalles de progreso
)
# Configurar HalvingRandomSearchCV
print("\nConfigurando búsqueda de hiperparámetros...")
search = HalvingRandomSearchCV(
    estimator=imputer,
    param_distributions=param_dist, #
    factor=2,
    resource='max_iter',
    max_resources=15,
    min_resources=5,
    scoring=imputation_scorer,
    cv=3,
    random_state=RANDOM_STATE,
    n_jobs=N_JOBS, # Usar todos los núcleos disponibles
    verbose=1 # Mostrar detalles de progreso
)

# Ejecutar la búsqueda de hiperparámetros
print("\nIniciando optimización de hiperparámetros...")
start_time = time.time()
search.fit(df[numeric_cols].values)  # Usamos .values para mejor rendimiento

print(f"\nBúsqueda completada en {time.time() - start_time:.2f} segundos")
print("\nMejores parámetros encontrados:")
print(search.best_params_)
print(f"\nMejor score (MSE negativo): {search.best_score_:.4f}")

# Aplicar el mejor modelo encontrado
best_imputer = search.best_estimator_
df_imputed = df.copy()
df_imputed[numeric_cols] = best_imputer.transform(df[numeric_cols].values)

# Guardar el modelo entrenado
joblib.dump(best_imputer, MODEL_SAVE_PATH)

# Validar resultados
missing_after = df_imputed[numeric_cols].isna().sum().sum() # Contar valores faltantes
print(f"\nValores faltantes después de imputación: {missing_after}")

# Guardar resultados imputados
df_imputed.to_csv(OUTPUT_PSV, sep='|', index=True)
df_imputed.to_excel(OUTPUT_EXCEL, index=True)

# Estandarización
scaler = StandardScaler()
df_standardized = df_imputed.copy()
df_standardized[numeric_cols] = scaler.fit_transform(df_imputed[numeric_cols])
df_standardized.to_csv(OUTPUT_PSV_STD, sep='|', index=True)
df_standardized.to_excel(OUTPUT_EXCEL_STD, index=True)

print("\nProceso completado exitosamente!")