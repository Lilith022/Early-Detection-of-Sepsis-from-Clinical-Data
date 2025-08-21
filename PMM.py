import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import time
import warnings
from concurrent.futures import ThreadPoolExecutor

warnings.filterwarnings('ignore')

# Configuración
RANDOM_STATE = 42
MAX_ITER = 10
N_ESTIMATORS = 300
RF_MAX_DEPTH = 8
RF_MAX_FEATURES = 0.6
RF_MIN_SAMPLES_LEAF = 5
N_DONORS = 5
N_JOBS = 4

# Rutas
INPUT_FILE = 'C:/Users/Lilit/Desktop/TESIS_EARLY_DETECTION_SEPSIS_2025_2/Datos_Previo_Imputar/Datos_previo_imputar.psv'
OUTPUT_PSV = 'C:/Users/Lilit/Desktop/TESIS_EARLY_DETECTION_SEPSIS_2025_2/Resultados_imputaciones/correcion_PMM.psv'
OUTPUT_EXCEL = OUTPUT_PSV.replace('.psv', '.xlsx')
OUTPUT_PSV_STD = OUTPUT_PSV.replace('.psv', '_estandarizado.psv')
OUTPUT_EXCEL_STD = OUTPUT_PSV_STD.replace('.psv', '.xlsx')

# Cargar datos
df = pd.read_csv(INPUT_FILE, sep='|', index_col=0)
df.columns = df.columns.str.replace('/', '_')
numeric_cols = df.iloc[:, 1:44].columns
cols_with_missing = [col for col in numeric_cols if df[col].isna().any()]

# Configurar modelos base
base_models = {
    col: RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        max_features=RF_MAX_FEATURES,
        min_samples_leaf=RF_MIN_SAMPLES_LEAF,
        random_state=RANDOM_STATE
    ) for col in cols_with_missing
}

# Función que procesa cada columna
def process_column(col):
    total_loss = 0
    count = 0

    complete_mask = ~df_imputed[col].isna()
    X_train = df_imputed.loc[complete_mask, numeric_cols.difference([col])]
    y_train = df_imputed.loc[complete_mask, col]

    model = base_models[col]
    model.fit(X_train, y_train)
    y_pred = model.predict(df_imputed[numeric_cols.difference([col])])

    missing_mask = df_imputed[col].isna()
    if missing_mask.sum() == 0:
        return total_loss, count

    knn = NearestNeighbors(n_neighbors=N_DONORS)
    knn.fit(y_pred[complete_mask].reshape(-1, 1))
    distances, indices = knn.kneighbors(y_pred[missing_mask].reshape(-1, 1))

    total_loss += np.mean(distances)
    count += 1

    for i, idx in enumerate(df_imputed.index[missing_mask]):
        donor_idx = np.random.choice(indices[i])
        df_imputed.loc[idx, col] = y_train.iloc[donor_idx]

    return total_loss, count

# Imputación PMM
print("\nIniciando imputación PMM...")
start_time = time.time()
df_imputed = df.copy()

with tqdm(total=MAX_ITER, desc="Progreso PMM") as pbar:
    for iteration in range(MAX_ITER):
        total_loss = 0
        count = 0
        with ThreadPoolExecutor(max_workers=N_JOBS) as executor:
            results = list(executor.map(process_column, cols_with_missing))
        for loss, c in results:
            total_loss += loss
            count += c
        pbar.update(1)
        pbar.set_postfix({'Pérdida': f"{total_loss/count:.4f}" if count > 0 else 'N/A'})

# Guardar resultados
missing_values = df_imputed[numeric_cols].isna().sum().sum()
print(f"\nTiempo de imputación: {round(time.time()-start_time, 2)} segundos")
print(f"Valores faltantes restantes: {missing_values}")

df_imputed.to_csv(OUTPUT_PSV, sep='|', index=True)
df_imputed.to_excel(OUTPUT_EXCEL, index=True)

# Estandarizar

scaler = StandardScaler()
df_standardized = df_imputed.copy()
df_standardized[numeric_cols] = scaler.fit_transform(df_imputed[numeric_cols])
df_standardized.to_csv(OUTPUT_PSV_STD, sep='|', index=True)
df_standardized.to_excel(OUTPUT_EXCEL_STD, index=True)


print("\nProceso completado exitosamente")
print(f"- Sin estandarizar: {OUTPUT_PSV} y {OUTPUT_EXCEL}")
print(f"- Estandarizado: {OUTPUT_PSV_STD} y {OUTPUT_EXCEL_STD}")


