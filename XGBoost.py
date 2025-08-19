import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from fancyimpute import SoftImpute  # ¡Asegúrate de instalar fancyimpute primero!
import time

# Configuración
RANDOM_STATE = 42
N_JOBS = -1  # Usar todos los núcleos disponibles

# Rutas de archivos
INPUT_FILE = 'C:/Users/Lilit/Desktop/Datos_previo_imputar.psv'
OUTPUT_PSV = 'C:/Users/Lilit/Desktop/Imputacion_XGBoost_SoftImpute.psv'
OUTPUT_EXCEL = OUTPUT_PSV.replace('.psv', '.xlsx')
OUTPUT_PSV_STD = OUTPUT_PSV.replace('.psv', '_estandarizado.psv')
OUTPUT_EXCEL_STD = OUTPUT_PSV_STD.replace('.psv', '.xlsx')

# Cargar datos
df = pd.read_csv(INPUT_FILE, sep='|', index_col=0)
df.columns = df.columns.str.replace('/', '_')
numeric_cols = df.iloc[:, 0:58].columns

# Prefilling con SoftImpute
print("\nAplicando SoftImpute...")
soft_imputer = SoftImpute(max_iters=100, verbose=False)
df_prefill = df.copy()
df_prefill[numeric_cols] = soft_imputer.fit_transform(df[numeric_cols].values)

# Imputación con XGBoost
df_imputed = df.copy()

print("\nIniciando imputación con XGBoost...")
for target_col in numeric_cols:
    if df[target_col].isna().any():
        not_nan = df[target_col].notna()
        to_impute = df[target_col].isna()
        
        X_full = df_prefill[numeric_cols].drop(columns=[target_col])
        y_full = df[target_col]

        X_train, X_val, y_train, y_val = train_test_split(
            X_full[not_nan], y_full[not_nan], 
            test_size=0.2, 
            random_state=RANDOM_STATE
        )

        model = xgb.XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=7,
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
            early_stopping_rounds=50  # ¡Movido aquí desde fit()!
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        df_imputed.loc[to_impute, target_col] = model.predict(X_full.loc[to_impute])
        df_prefill.loc[to_impute, target_col] = df_imputed.loc[to_impute, target_col]

# --- Guardar resultados ---
df_imputed.to_csv(OUTPUT_PSV, sep='|', index=True)
df_imputed.to_excel(OUTPUT_EXCEL, index=True)

# --- Estandarización ---
scaler = StandardScaler()
df_standardized = df_imputed.copy()
df_standardized[numeric_cols] = scaler.fit_transform(df_imputed[numeric_cols])
df_standardized.to_csv(OUTPUT_PSV_STD, sep='|', index=True)
df_standardized.to_excel(OUTPUT_EXCEL_STD, index=True)


print("\nResumen de modelos ajustados por columna:")
print("\n¡Proceso completado con SoftImpute + XGBoost optimizado!")