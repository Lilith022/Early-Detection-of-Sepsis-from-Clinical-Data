import pandas as pd
import numpy as np
import os

# Ruta del archivo original
INPUT_FILE = 'C:/Users/Lilit/Desktop/TESIS_EARLY_DETECTION_SEPSIS_2025_2/Resultados_21horas/Ventanas_Deslizantes_Estadisticas.psv'

# Ruta del archivo de salida (nuevo nombre)
OUTPUT_FILE = 'C:/Users/Lilit/Desktop/TESIS_EARLY_DETECTION_SEPSIS_2025_2/Resultados_21horas/Ventanas_deslizantes_variables_eliminadas.psv'

# Lista de variables a eliminar
variables_eliminadas = [
    'Temp_var', 'Temp_last', 'DBP_var', 'DBP_last',
    'EtCO2_min', 'EtCO2_max', 'EtCO2_mean', 'EtCO2_var', 'EtCO2_last',
    'BaseExcess_min', 'BaseExcess_max', 'HCO3_min', 'HCO3_max',
    'FiO2_min', 'FiO2_max', 'pH_min', 'pH_max', 'PaCO2_min', 'PaCO2_max',
    'SaO2_min', 'SaO2_max', 'AST_min', 'AST_max', 'Alkalinephos_min', 'Alkalinephos_max',
    'Calcium_min', 'Calcium_max', 'Chloride_min', 'Chloride_max',
    'Creatinine_min', 'Creatinine_max', 'Glucose_min', 'Glucose_max',
    'Lactate_min', 'Lactate_max', 'Magnesium_min', 'Magnesium_max',
    'Phosphate_min', 'Phosphate_max', 'Potassium_min', 'Potassium_max',
    'Hct_min', 'Hct_max', 'Hgb_min', 'Hgb_max',
    'PTT_min', 'PTT_max', 'WBC_min', 'WBC_max', 'Platelets_min', 'Platelets_max'
]

# Función para eliminar columnas
def eliminar_variables(df, variables_eliminadas):
    return df.drop(columns=[col for col in variables_eliminadas if col in df.columns], errors='ignore')

# Leer el archivo
df = pd.read_csv(INPUT_FILE, sep='|')

# Eliminar variables
df = eliminar_variables(df, variables_eliminadas)

# Guardar archivo resultante en la misma ruta con otro nombre
df.to_csv(OUTPUT_FILE, sep='|', index=False)

print("Archivo procesado guardado en:", OUTPUT_FILE)
print("Dimensiones finales del DataFrame:", df.shape)