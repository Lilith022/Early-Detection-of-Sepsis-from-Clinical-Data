import pandas as pd
import numpy as np
import miceforest as mf
import time
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# =======================
# CONFIGURACIÓN
# =======================
RANDOM_STATE = 42
N_DATASETS = 4
N_ITERATIONS = 4

# Rutas de archivos
INPUT_FILE = 'D:/Desktop/Tesis/Data/Datos_previo_imputar/Datos_previo_imputar.psv'
OUTPUT_PSV = 'D:/Desktop/Tesis/Data/Imputaciones/MiceForest.psv'
OUTPUT_EXCEL = OUTPUT_PSV.replace('.psv', '.xlsx')
OUTPUT_PSV_STD = OUTPUT_PSV.replace('.psv', '_estandarizado.psv')
OUTPUT_EXCEL_STD = OUTPUT_PSV_STD.replace('.psv', '.xlsx')

# =======================
# 1. Cargar y preparar datos
# =======================
df = pd.read_csv(INPUT_FILE, sep='|')
df.columns = df.columns.str.replace('/', '', regex=False)
df = df.reset_index(drop=True)

# Columnas numéricas a imputar
numeric_cols = df.columns[1:53].tolist()

# Conversión segura a numérico
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')


print(df[numeric_cols].notna().sum().sort_values())

# =======================
# 2. Imputación MICE
# =======================
print("\nIniciando imputación MICE...")
start_time = time.time()

kernel = mf.ImputationKernel(
    data=df[numeric_cols],
    num_datasets=N_DATASETS,
    mean_match_candidates=2,
    save_all_iterations_data=False,
    random_state=RANDOM_STATE
)

# Ejecutar MICE
kernel.mice(iterations=N_ITERATIONS, verbose=True)

print(f"\nTiempo de imputación: {round(time.time() - start_time, 2)} segundos")

# =======================
# 3. Promediar imputaciones solo en NaN originales
# =======================
print("\nProcesando resultados...")
mask_nan = df[numeric_cols].isna()
imputed_datasets = [kernel.complete_data(dataset=i) for i in range(N_DATASETS)]
imputed_array = np.stack([d.values for d in imputed_datasets], axis=0)

final_df = df.copy()
mean_imputed = np.mean(imputed_array, axis=0)
final_df.loc[:, numeric_cols] = np.where(mask_nan, mean_imputed, df[numeric_cols])

# =======================
# 4. Guardar resultados
# =======================
print("\nGuardando resultados...")
final_df.to_csv(OUTPUT_PSV, sep='|', index=False)
final_df.to_excel(OUTPUT_EXCEL, index=False)

# =======================
# 5. Estandarización Z-score
# =======================
print("\nAplicando estandarización Z-score...")
scaler = StandardScaler()
df_standardized = final_df.copy()
df_standardized[numeric_cols] = scaler.fit_transform(final_df[numeric_cols])

df_standardized.to_csv(OUTPUT_PSV_STD, sep='|', index=False)
df_standardized.to_excel(OUTPUT_EXCEL_STD, index=False)

# =======================
# 6. Resumen
# =======================
print("\nProceso completado exitosamente!")
print(f"Total de valores imputados: {mask_nan.sum().sum()}")

