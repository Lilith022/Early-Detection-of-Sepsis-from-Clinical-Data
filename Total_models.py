import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from scipy.stats import norm

# Modelos
from catboost import CatBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier 
from sklearn.svm import SVC
# ========================
# CONFIGURACIÓN
# ========================
INPUT_FILE = "C:/Users/Lilit/Desktop/TESIS_EARLY_DETECTION_SEPSIS_2025_2/Datos_Previo_Imputar/Datos_previo_imputar.psv"
OUTPUT_EXCEL = "C:/Users/Lilit/Desktop/TESIS_EARLY_DETECTION_SEPSIS_2025_2/Resultados_modelos/No_imputados/Datos_sin_imputar.xlsx"
N_FOLDS = 5
RANDOM_STATE = 42
Z = 1.96  # Para intervalo de confianza del 95%

# ========================
# CARGA DE DATOS
# ========================

print("Cargando datos imputados...")
df = pd.read_csv(INPUT_FILE, sep='|')
numeric_cols = df.iloc[:, 1:36].columns 
imputed_df = df.copy() 
# Estandarizar resultados imputados
scaler = StandardScaler() # Z-Score standardization
df_standardized = df.copy()
df_standardized[numeric_cols] = scaler.fit_transform(imputed_df[numeric_cols])

X = df_standardized.drop(columns=["Paciente", "SepsisLabel"]).to_numpy()
y = df_standardized["SepsisLabel"].to_numpy()

# ========================
# DEFINICIÓN DE MODELOS
# ========================
modelos = {
    "CatBoost": CatBoostClassifier(verbose=0, random_state=RANDOM_STATE),
    "ExtraTrees": ExtraTreesClassifier(random_state=RANDOM_STATE),
    "GradientBoosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    "RandomForest": RandomForestClassifier(random_state=RANDOM_STATE),
    "LogisticRegression": LogisticRegression(random_state=RANDOM_STATE),
    "KNN": KNeighborsClassifier(),
    "DecisionTree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "NaiveBayes": GaussianNB(),
    "XGBoost": XGBClassifier(random_state=RANDOM_STATE),
    "LightGBM": LGBMClassifier(random_state=RANDOM_STATE),
    "MLP": MLPClassifier(random_state=RANDOM_STATE),
    "SVC": SVC(random_state=RANDOM_STATE)
}

# ========================
# ESTRUCTURAS PARA RESULTADOS
# ========================
resultados_metricas = []
resultados_auc = []
matrices_confusion = {}
kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

# ========================
# BUCLE PRINCIPAL DE EVALUACIÓN
# ========================
for nombre, modelo in modelos.items():
    print(f"\nEvaluando modelo: {nombre}")
    
    # Listas para almacenar resultados por fold
    fold_auc_roc = []
    fold_auc_pr = []
    fold_acc = []
    fold_prec = []
    fold_rec = []
    fold_f1 = []
    
    all_y_true = []
    all_y_pred = []
    all_y_proba = []
    
    # Validación cruzada
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Entrenamiento y predicción
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_val)
        
        if hasattr(modelo, "predict_proba"):
            y_proba = modelo.predict_proba(X_val)[:, 1]
        else:
            y_proba = modelo.decision_function(X_val)
        
        # Métricas por fold
        fold_auc_roc.append(roc_auc_score(y_val, y_proba))
        precision_vals, recall_vals, _ = precision_recall_curve(y_val, y_proba)
        fold_auc_pr.append(auc(recall_vals, precision_vals))
        fold_acc.append(accuracy_score(y_val, y_pred))
        fold_prec.append(precision_score(y_val, y_pred))
        fold_rec.append(recall_score(y_val, y_pred))
        fold_f1.append(f1_score(y_val, y_pred))
        
        # Acumular para métricas globales
        all_y_true.extend(y_val)
        all_y_pred.extend(y_pred)
        all_y_proba.extend(y_proba)
    
    # ========================
    # CÁLCULO DE INTERVALOS DE CONFIANZA
    # ========================
    # AUC-ROC
    mean_auc_roc = np.mean(fold_auc_roc)
    std_auc_roc = np.std(fold_auc_roc)
    margin_error_roc = Z * (std_auc_roc / np.sqrt(N_FOLDS))
    ci_lower_roc = mean_auc_roc - margin_error_roc
    ci_upper_roc = mean_auc_roc + margin_error_roc
    
    # AUC-PR
    mean_auc_pr = np.mean(fold_auc_pr)
    std_auc_pr = np.std(fold_auc_pr)
    margin_error_pr = Z * (std_auc_pr / np.sqrt(N_FOLDS))
    ci_lower_pr = mean_auc_pr - margin_error_pr
    ci_upper_pr = mean_auc_pr + margin_error_pr
    
    # Guardar resultados AUC con intervalos
    resultados_auc.append({
        "Modelo": nombre,
        "AUC_ROC": f"{mean_auc_roc:.3f} [{ci_lower_roc:.3f}, {ci_upper_roc:.3f}]",
        "AUC_PR": f"{mean_auc_pr:.3f} [{ci_lower_pr:.3f}, {ci_upper_pr:.3f}]"
    })
    
    # ========================
    # MÉTRICAS GLOBALES
    # ========================
    cm_global = confusion_matrix(all_y_true, all_y_pred)
    matrices_confusion[nombre] = cm_global
    
    acc_global = accuracy_score(all_y_true, all_y_pred)
    prec_global = precision_score(all_y_true, all_y_pred)
    rec_global = recall_score(all_y_true, all_y_pred)
    f1_global = f1_score(all_y_true, all_y_pred)
    auc_roc_global = roc_auc_score(all_y_true, all_y_proba)
    
    precision_vals, recall_vals, _ = precision_recall_curve(all_y_true, all_y_proba)
    auc_pr_global = auc(recall_vals, precision_vals)
    
    resultados_metricas.append({
        "Modelo": nombre,
        "Accuracy": acc_global,
        "Precision": prec_global,
        "Recall": rec_global,
        "F1": f1_global,
        "AUC_ROC": auc_roc_global,
        "AUC_PR": auc_pr_global,
        "AUC_ROC_IC95": f"[{ci_lower_roc:.3f}, {ci_upper_roc:.3f}]",
        "AUC_PR_IC95": f"[{ci_lower_pr:.3f}, {ci_upper_pr:.3f}]"
    })
    
    print(f"  AUC-ROC: {mean_auc_roc:.3f} [{ci_lower_roc:.3f}, {ci_upper_roc:.3f}]")
    print(f"  AUC-PR: {mean_auc_pr:.3f} [{ci_lower_pr:.3f}, {ci_upper_pr:.3f}]")

    fpr, tpr, _ = roc_curve(all_y_true, all_y_proba)
    auc_roc = roc_auc_score(all_y_true, all_y_proba)
    plt.plot(fpr, tpr, label=f"{nombre} (AUC={auc_roc:.2f})")

# ========================
# GUARDAR RESULTADOS EN UN SOLO BLOQUE
# ========================
with pd.ExcelWriter(OUTPUT_EXCEL, engine='openpyxl') as writer:
    pd.DataFrame(resultados_metricas).to_excel(writer, sheet_name="Metricas", index=False)
    pd.DataFrame(resultados_auc).to_excel(writer, sheet_name="AUC_Intervalos", index=False)
    for nombre, cm in matrices_confusion.items():
        cm_df = pd.DataFrame(cm, columns=["Pred 0", "Pred 1"], index=["Real 0", "Real 1"])
        cm_df.to_excel(writer, sheet_name=f"CM_{nombre[:20]}")

print(f"\nResultados guardados en: {OUTPUT_EXCEL}")

# ========================
# GRÁFICAS COMPARATIVAS
# ========================

plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("Tasa de Falsos Positivos (FPR)")
plt.ylabel("Tasa de Verdaderos Positivos (TPR)")
plt.title("Curvas ROC - Comparación de Modelos")
plt.legend()
plt.grid(True)
plt.show()

print("\nResumen de resultados (ordenado por AUC-ROC):")
resultados_df = pd.DataFrame(resultados_metricas)
print(resultados_df.sort_values(by="AUC_ROC", ascending=False))
