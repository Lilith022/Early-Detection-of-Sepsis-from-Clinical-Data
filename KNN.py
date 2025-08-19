import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler # Z-Score standardization
from tqdm import tqdm 
import time
import numpy as np

# Configuración
N_NEIGHBORS = 5  # Número de vecinos para KNN _ referencia_articulo

# Rutas de archivos
INPUT_FILE = 'C:/Users/Lilit/Downloads/Datos_previo_imputar.psv'
OUTPUT_PSV = 'C:/Users/Lilit/Desktop/imputacion_KNN.psv'
OUTPUT_EXCEL = OUTPUT_PSV.replace('.psv', '.xlsx')
OUTPUT_PSV_STD = OUTPUT_PSV.replace('.psv', '_estandarizado.psv')
OUTPUT_EXCEL_STD = OUTPUT_PSV_STD.replace('.psv', '.xlsx')

# Cargar y procesar datos
df = pd.read_csv(INPUT_FILE, sep='|', index_col=0) #columnas-1col:Paciente
df.columns = df.columns.str.replace('/', '_') # Reemplazar '/' por '_' para las variables añadidas: por la variable añadida
numeric_cols = df.iloc[:, 0:58].columns  

# Imputación KNN
print("\nIniciando imputación KNN...")
start_time = time.time()
imputer = KNNImputer(n_neighbors=N_NEIGHBORS)
imputed_values = imputer.fit_transform(df[numeric_cols]) # Imputación KNN
print("\nTiempo de imputación:", round(time.time() - start_time, 2), "segundos")

# Convertir el array numpy de vuelta a DataFrame para verificar valores faltantes
imputed_df_temp = pd.DataFrame(imputed_values, columns=numeric_cols, index=df.index)
missing_values = imputed_df_temp.isna().sum().sum()# Verificar si quedan valores faltantes fila y columna
print("Valores faltantes restantes:", missing_values)

# Guardar resultados imputados
imputed_df = df.copy()  # Hacer una copia del DataFrame original
imputed_df[numeric_cols] = imputed_values  # Reemplazar las columnas numéricas imputadas
imputed_df.to_csv(OUTPUT_PSV, sep='|', index=True)
imputed_df.to_excel(OUTPUT_EXCEL, index=True)

# Estandarizar resultados imputados
scaler = StandardScaler() # Z-Score standardization
df_standardized = df.copy()
df_standardized[numeric_cols] = scaler.fit_transform(imputed_df[numeric_cols]) # Estandarización de las columnas numéricas

# Guardar resultados estandarizados
df_standardized.to_csv(OUTPUT_PSV_STD, sep='|', index=True)
df_standardized.to_excel(OUTPUT_EXCEL_STD, index=True)

print("Proceso completado exitosamente")