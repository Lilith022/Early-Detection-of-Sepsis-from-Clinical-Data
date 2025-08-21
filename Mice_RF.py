import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
import time
import warnings
warnings.filterwarnings('ignore')

# Configuración
RANDOM_STATE = 42
MAX_ITER = 15
N_ESTIMATORS = 500
RF_MAX_DEPTH = 10
RF_MAX_FEATURES = 0.6  # Proporción de características a considerar
RF_MIN_SAMPLES_LEAF = 5  # Mínimo de muestras por hoja
CONVERGENCE_TOL = 0.01
N_JOBS = -1  # Utilizar todos los núcleos disponibles de la CPU
PATIENCE = 3  # Número de iteraciones sin mejora para detener

# Rutas de archivos
INPUT_FILE = 'C:/Users/Lilit/Desktop/Datos_previo_imputar.psv'
OUTPUT_PSV = 'C:/Users/Lilit/Desktop/Mice_RF.psv'
OUTPUT_EXCEL = OUTPUT_PSV.replace('.psv', '.xlsx')
OUTPUT_PSV_STD = OUTPUT_PSV.replace('.psv', '_estandarizado.psv')
OUTPUT_EXCEL_STD = OUTPUT_PSV_STD.replace('.psv', '.xlsx')

# Cargar y procesar datos
df = pd.read_csv(INPUT_FILE, sep='|', index_col=0)
df.columns = df.columns.str.replace('/', '_')  # Reemplazar '/' por '_' en nombres de columnas
numeric_cols = df.iloc[:, 0:40].columns  # Selección de columnas numéricas

# Configurar el imputador con RandomForest como estimador base
print("\nConfigurando imputador con RandomForest...")

imputer = IterativeImputer(
    estimator=RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        max_features=RF_MAX_FEATURES,
        min_samples_leaf=RF_MIN_SAMPLES_LEAF,
        n_jobs=N_JOBS,
        random_state=RANDOM_STATE
    ),
    max_iter=MAX_ITER,
    tol=CONVERGENCE_TOL,
    random_state=RANDOM_STATE,
    verbose=2  # Mostrar detalles de progreso
)

# Función de monitoreo de convergencia
class ConvergenceMonitor:
    def __init__(self, patience=PATIENCE, tol=CONVERGENCE_TOL):
        self.iter = 0
        self.best_loss = np.inf
        self.no_improve = 0
        self.patience = patience
        self.tol = tol

    def __call__(self, X, y):
        self.iter += 1
        current_loss = np.nanmean(np.abs(X - y))  # Calcular la pérdida (error absoluto medio)

        if current_loss < (self.best_loss - self.tol):
            self.best_loss = current_loss  # Si hay mejora, actualizar la mejor pérdida
            self.no_improve = 0
        else:
            self.no_improve += 1

        if self.no_improve >= self.patience:  # Si no hay mejora durante "patience" iteraciones
            print("\nDetención temprana activada debido a falta de mejora.")
            raise StopIteration()  # Detener el proceso de imputación

# Agregar el callback de convergencia al imputador
imputer.callback = ConvergenceMonitor(patience=PATIENCE, tol=CONVERGENCE_TOL)

# Imputación con IterativeImputer (MICE Random Forest)
print("\nIniciando imputación con IterativeImputer (MICE Random Forest)...")
start_time = time.time()

# Ejecutar imputación
try:
    imputed_values = imputer.fit_transform(df[numeric_cols])
except StopIteration:
    print("Convergencia alcanzada antes de completar todas las iteraciones.")

print("\nTiempo de imputación:", round(time.time() - start_time, 2), "segundos")

# Verificar resultados
missing_values = pd.isna(imputed_values).sum()
print("Valores faltantes restantes:", missing_values.sum())

# Crear DataFrame con los valores imputados
imputed_df = df.copy()  # Hacer una copia del DataFrame original
imputed_df[numeric_cols] = imputed_values  # Reemplazar las columnas numéricas imputadas

# Guardar resultados imputados
print("\nGuardando resultados imputados...")
imputed_df.to_csv(OUTPUT_PSV, sep='|', index=True)
imputed_df.to_excel(OUTPUT_EXCEL, index=True)

# Estandarizar resultados imputados
print("\nAplicando estandarización Z-score...")
scaler = StandardScaler()
df_standardized = imputed_df.copy()
df_standardized[numeric_cols] = scaler.fit_transform(imputed_df[numeric_cols])

# Guardar resultados estandarizados
df_standardized.to_csv(OUTPUT_PSV_STD, sep='|', index=True)
df_standardized.to_excel(OUTPUT_EXCEL_STD, index=True)

print("Proceso completado exitosamente")

