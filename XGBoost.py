#Código Final Final

# Código funcional con SHAP + Utility metric integrada
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
)
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt

import optuna
import shap
import os
import seaborn as sns
import warnings
from xgboost import XGBClassifier

# ---------- NUEVAS FUNCIONES (UTILITY) ----------
def compute_patient_utility(
    labels: np.ndarray, # Verdaderos valores de la etiqueta
    preds: np.ndarray, # Predicciones del modelo
    dt_early: int = -6,
    dt_optimal: int = 0,
    dt_late: int = 9,
    max_u_tp: float = 1.0,
    min_u_fn: float = -2.0,
    u_fp: float = -0.05,
    u_tn: float = 0.0,
    check_errors: bool = True
) -> float:

    if check_errors:
        if len(labels) != len(preds):
            raise ValueError("labels y preds deben tener la misma longitud.")
        if not all((l in (0, 1) for l in labels)):
            raise ValueError("labels debe contener solo 0/1.")
        if not all((p in (0, 1) for p in preds)):
            raise ValueError("preds debe contener solo 0/1.")
        if not (dt_early < dt_optimal < dt_late):
            raise ValueError("Se requiere dt_early < dt_optimal < dt_late (offsets relativos al onset).")

    L = len(labels) # longitud de la serie temporal 16 por paciente
    u_t = np.zeros(L)

    # ¿hay sepsis?
    if np.any(labels == 1):
        onset = int(np.argmax(labels)) #Detecta el primer 1
        is_septic = True
    else:
        onset = None
        is_septic = False

    denom1 = float(dt_optimal - dt_early) # pendiente recompensa positiva
    denom2 = float(dt_late - dt_optimal) # pendiente recompensa negativa (tardía)
    # pendientes
    m1 = max_u_tp / denom1 if denom1 != 0 else 0.0         # crece hasta max_u_tp
    m2 = -max_u_tp / denom2 if denom2 != 0 else 0.0        # decrece desde max_u_tp hasta 0
    m3 = min_u_fn / denom2 if denom2 != 0 else 0.0         # penalización FN tardío

    for t in range(L): #Recorre todos los registros del paciente 
        if not is_septic:
            # paciente sano: FP/Punto (constante)
            if preds[t] == 1:
                u_t[t] = u_fp
            else:
                u_t[t] = u_tn
            continue

        x = t - onset  # tiempo relativo al onset, cuánto falta o cuánto ha pasado respecto a t-optimal nos coloca optimal en 0

        if x > dt_late: # fuera de ventana útil
            u_t[t] = 0.0
            continue

        if preds[t] == 1:
            # Predicho positivo
            if x < dt_early:
                # demasiado temprano -> tratado como FP
                u_t[t] = u_fp
            elif dt_early <= x <= dt_optimal:
                # tramo creciente: 0 en dt_early -> max_u_tp en dt_optimal
                u_t[t] = m1 * (x - dt_early)
            elif dt_optimal < x <= dt_late:
                # tramo decreciente: max_u_tp en dt_optimal -> 0 en dt_late
                u_t[t] = max_u_tp + m2 * (x - dt_optimal)
            else:
                u_t[t] = 0.0
        else:
            # Predicho negativo -> posible FN tardío
            if x <= dt_optimal:
                u_t[t] = 0.0
            elif dt_optimal < x <= dt_late:
                u_t[t] = m3 * (x - dt_optimal)
            else:
                u_t[t] = 0.0

    return float(np.sum(u_t)) # Suma de utilidades por paciente


def compute_normalized_utility(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    dt_early: int = -6,
    dt_optimal: int = 0,
    dt_late: int = 9,
    max_u_tp: float = 1.0,
    min_u_fn: float = -2.0,
    u_fp: float = -0.05,
    u_tn: float = 0.0,
    return_details: bool = False
):

    y_true = np.asarray(y_true) # Verdaderos positivos
    y_proba = np.asarray(y_proba) # Probabilidades predichas
    y_pred = np.asarray(y_pred) # Predicciones
    groups = np.asarray(groups) # Grupos de pacientes

    unique_patients = np.unique(groups) # IDs únicos de pacientes
    observed_utils = []
    best_utils = []
    inaction_utils = []
    worst_utils = []
    patient_ids = []

    for pid in unique_patients: # Recorre cada paciente
        idx = np.where(groups == pid)[0] # Identifica los índices del paciente
        if idx.size == 0:
            continue # No hay datos para este paciente
        lab = y_true[idx].astype(int) # Etiquetas verdaderas del paciente
        pred_obs = y_pred[idx].astype(int) # Predicciones observadas del paciente
        L = len(lab)

        # observed utility
        u_obs = compute_patient_utility( # Utilidad observada
            lab, pred_obs,
            dt_early=dt_early, dt_optimal=dt_optimal, dt_late=dt_late,
            max_u_tp=max_u_tp, min_u_fn=min_u_fn, u_fp=u_fp, u_tn=u_tn
        )

        # best (1's en ventana utile si el paciente será séptico)
        best_pred = np.zeros(L, dtype=int)
        if np.any(lab == 1): # Si hay algún positivo en las etiquetas
            onset = int(np.argmax(lab))
            start_idx = max(0, onset + dt_early) # tearly
            end_idx = min(L, onset + dt_late + 1) # tlate
            if start_idx < end_idx:
                best_pred[start_idx:end_idx] = 1

        #La utilidad solo detecta el primer uno y recompensa o penaliza
        u_best = compute_patient_utility( 
            lab, best_pred,
            dt_early=dt_early, dt_optimal=dt_optimal, dt_late=dt_late,
            max_u_tp=max_u_tp, min_u_fn=min_u_fn, u_fp=u_fp, u_tn=u_tn
        )

        # inaction (todo cero)
        inaction_pred = np.zeros(L, dtype=int)
        u_inaction = compute_patient_utility(
            lab, inaction_pred,
            dt_early=dt_early, dt_optimal=dt_optimal, dt_late=dt_late,
            max_u_tp=max_u_tp, min_u_fn=min_u_fn, u_fp=u_fp, u_tn=u_tn
        )

        worst_pred = 1 - best_pred
        u_worst = compute_patient_utility(
            lab, worst_pred,
            dt_early=dt_early, dt_optimal=dt_optimal, dt_late=dt_late,
            max_u_tp=max_u_tp, min_u_fn=min_u_fn, u_fp=u_fp, u_tn=u_tn
        )

        patient_ids.append(pid)
        observed_utils.append(u_obs)
        best_utils.append(u_best)
        inaction_utils.append(u_inaction)
        worst_utils.append(u_worst)

    un_obs = float(np.sum(observed_utils))
    un_best = float(np.sum(best_utils))
    un_inaction = float(np.sum(inaction_utils))

    denom = (un_best - un_inaction)
    if denom == 0:
        normalized = float('nan')
    else:
        normalized = (un_obs - un_inaction) / denom

    details = {
        'patient_ids': np.array(patient_ids),
        'observed_utils': np.array(observed_utils),
        'best_utils': np.array(best_utils),
        'inaction_utils': np.array(inaction_utils),
        'worst_utils': np.array(worst_utils),
        'unnormalized': {'observed': un_obs, 'best': un_best, 'inaction': un_inaction}
    }

    if return_details:
        return normalized, details
    else:
        return normalized


INPUT_FILE = "/mnt/Home-Group/csantos_cps/JosmanDeisy/imputaciones/Resultados_imputaciones/LIGHTGBM.psv"
df = pd.read_csv(INPUT_FILE, sep='|')

# ELIMINAR LAS VARIABLES HospAdmTime 
df = df.drop(columns=['HospAdmTime'], errors='ignore')

def sensitivity_specificity(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp+fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn+fp) > 0 else 0
    return sensitivity, specificity

def best_threshold(y_true, y_proba, alpha):
    thresholds = np.linspace(0, 1, 200)
    f1_scores, recalls, combined_scores = [], [], []

    for thr in thresholds:
        preds = (y_proba >= thr).astype(int)
        f1_val = f1_score(y_true, preds, zero_division=0)
        rec_val = recall_score(y_true, preds, zero_division=0)

        f1_scores.append(f1_val)
        recalls.append(rec_val)
        combined_scores.append(alpha * f1_val + (1 - alpha) * rec_val)

    best_idx = np.argmax(combined_scores)
    return thresholds[best_idx], thresholds, f1_scores, recalls, combined_scores

def mean_ci(data):
    arr = np.array(data)
    mean = arr.mean()
    std = arr.std(ddof=1)
    n = len(arr)
    se = std / np.sqrt(n)
    h = se * 1.96
    return mean, h

# Cargar y preparar datos
X = df.drop(columns=["Paciente", "SepsisLabel"])
y = df["SepsisLabel"]
groups = df["Paciente"]

print(f"X shape: {X.shape} | Distribución clases:\n{y.value_counts()}")

# Dividir 85% train y 15% test (estratificado por clases)
X_train, X_test, y_train, y_test, groups_train, groups_test = train_test_split(
    X, y, groups, test_size=0.15, stratify=y, random_state=42
)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")

#------------------------------------------------------------------------------------



# Validación cruzada por grupos
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
# Añadimos 'utility' a las métricas a recolectar
scoring = ['accuracy', 'f1', 'precision', 'recall', 'roc_auc', 'sensitivity', 'specificity', 'utility']

val_scores = {metric: [] for metric in scoring}
all_y_true, all_y_proba = [], []
models, fold_metrics, fold_thresholds, all_fold_indices = [], [], [], []

# ENTRENAMIENTO CON VALIDACIÓN CRUZADA (FORMA CORRECTA)
for fold, (train_idx, val_idx) in enumerate(tqdm(cv.split(X_train, y_train, groups=groups_train), total=5, desc="Procesando pliegues")):
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

    # GUARDAR ÍNDICES PARA USAR DESPUÉS
    all_fold_indices.append((train_idx.copy(), val_idx.copy()))

    # Calcula un vector de pesos por muestra
    sample_weights = np.where(y_train == 0, 1, (((len(y_train) - sum(y_train)) / sum(y_train)) * 0.5))

    # Entrena con sample_weight
    model = XGBClassifier(
        n_jobs=-1,
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss"
    )
    model.fit(X_train, y_train, sample_weight=sample_weights)



    y_proba = model.predict_proba(X_val)[:, 1]


    # Threshold óptimo por F1 en este fold
    thr, thresholds, f1s, recalls, combined = best_threshold(y_val, y_proba, alpha=0.6)
    y_pred = (y_proba >= thr).astype(int)

    # Calcular todas las métricas
    metrics_fold = {
        "accuracy": accuracy_score(y_val, y_pred),
        "f1": f1_score(y_val, y_pred, zero_division=0),
        "precision": precision_score(y_val, y_pred, zero_division=0),
        "recall": recall_score(y_val, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_val, y_proba) if len(np.unique(y_val)) >= 2 else np.nan
    }

    sens, spec = sensitivity_specificity(y_val, y_pred)
    metrics_fold["sensitivity"] = sens
    metrics_fold["specificity"] = spec

    # --- NUEVO: calcular utility para este fold (validación) ---
    try:
        util_fold = compute_normalized_utility(
            y_true=y_val.values,
            y_proba=y_proba,
            y_pred=y_pred,
            groups=groups_train.iloc[val_idx].values,
            dt_early=-6, dt_optimal=0, dt_late=9,
            max_u_tp=1.0, min_u_fn=-2.0, u_fp=-0.05, u_tn=0.0
        )
    except Exception as e:
        util_fold = float('nan')
        warnings.warn(f"Error calculando utility en fold {fold}: {e}")

    metrics_fold["utility"] = util_fold
    # --- fin nuevo ---

    # Guardar en resultados generales
    for metric in scoring:
        val_scores[metric].append(metrics_fold[metric])

    # Guardar modelo, métricas y threshold
    models.append(model)
    fold_metrics.append(metrics_fold)
    fold_thresholds.append(thr)

    all_y_true.extend(y_val)
    all_y_proba.extend(y_proba)

# Identificar mejor modelo
best_fold = np.argmax([0.3 * m["f1"] + 0.7 * m["sensitivity"] for m in fold_metrics])
best_model = models[best_fold]
best_threshold_fold = fold_thresholds[best_fold]

print(f"\n>>> Mejor modelo: fold {best_fold+1}")
print(f"F1: {fold_metrics[best_fold]['f1']:.4f}")
print(f"Sensitivity: {fold_metrics[best_fold]['sensitivity']:.4f}")
print(f"Threshold: {best_threshold_fold:.3f}")

# Obtener datos del mejor fold USANDO LOS ÍNDICES GUARDADOS
train_idx_best, val_idx_best = all_fold_indices[best_fold]
X_tr_best, X_val_best = X_train.iloc[train_idx_best], X_train.iloc[val_idx_best]
y_tr_best, y_val_best = y_train.iloc[train_idx_best], y_train.iloc[val_idx_best]

# OPTUNA sobre el fold ganador
def objective(trial):
    params_optuna = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 2000),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
        "random_state": 42,
        "n_jobs": -1,
        "use_label_encoder": False,
        "eval_metric": "logloss"
    }

    model = XGBClassifier(**params_optuna)

    # generar sample_weights con el y de este pliegue
    sample_weights = np.where(y_tr_best == 0, 1, (((len(y_tr_best) - sum(y_tr_best)) / sum(y_tr_best)) * 0.5))

    model.fit(X_tr_best, y_tr_best, sample_weight=sample_weights)

    y_proba = model.predict_proba(X_val_best)[:, 1]
    y_pred = (y_proba >= best_threshold_fold).astype(int)

    sensitivity = recall_score(y_val_best, y_pred, zero_division=0)
    auc_score = roc_auc_score(y_val_best, y_proba)
    f1 = f1_score(y_val_best, y_pred, zero_division=0)

    return 3 * sensitivity + 1 * f1 + 0.5 * auc_score



study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30, show_progress_bar=True)

print("\n>>> Mejores hiperparámetros Optuna:")
for key, value in study.best_params.items():
    print(f"{key}: {value}")

# Entrenar modelo final con mejores parámetros
optuna_model = XGBClassifier(
     **study.best_params,
     use_label_encoder=False,
     eval_metric='logloss',
     random_state=42,
     n_jobs=-1,
     scale_pos_weight=sample_weights
 )
optuna_model.fit(X_tr_best, y_tr_best)


# Recalcular mejor threshold con modelo Optuna
y_val_proba_optuna = optuna_model.predict_proba(X_val_best)[:, 1]
best_thr_optuna, _, _, _, _ = best_threshold(y_val_best, y_val_proba_optuna, alpha=0.6)

print(f">>> Mejor threshold tras Optuna: {best_thr_optuna:.3f}")

# EVALUACIÓN EN TEST SET
y_test_proba_optuna = optuna_model.predict_proba(X_test)[:, 1]
y_test_pred_optuna = (y_test_proba_optuna >= best_thr_optuna).astype(int)

# Métricas finales en TEST
test_metrics = {
    "accuracy": accuracy_score(y_test, y_test_pred_optuna),
    "f1": f1_score(y_test, y_test_pred_optuna, zero_division=0),
    "precision": precision_score(y_test, y_test_pred_optuna, zero_division=0),
    "recall": recall_score(y_test, y_test_pred_optuna, zero_division=0),
    "roc_auc": roc_auc_score(y_test, y_test_proba_optuna) if len(np.unique(y_test)) >= 2 else np.nan
}

sens_test, spec_test = sensitivity_specificity(y_test, y_test_pred_optuna)
test_metrics["sensitivity"] = sens_test
test_metrics["specificity"] = spec_test

# --- NUEVO: utility en TEST (normalizada de cohorte) ---
normalized_utility_test, util_details_test = compute_normalized_utility(
    y_true=y_test.values,
    y_proba=y_test_proba_optuna,
    y_pred=y_test_pred_optuna,
    groups=groups_test.values,
    dt_early=-6, dt_optimal=0, dt_late=9,
    max_u_tp=1.0, min_u_fn=-2.0, u_fp=-0.05, u_tn=0.0,
    return_details=True
)
test_metrics["utility"] = normalized_utility_test
# --- FIN NUEVO ---

print("\n" + "="*60)
print("=== MÉTRICAS FINALES EN TEST SET ===")
print("="*60)
for metric, value in test_metrics.items():
    if not np.isnan(value):
        print(f"{metric:15}: {value:.4f}")

# Resultados de validación cruzada
print("\n" + "="*60)
print("=== RESULTADOS VALIDACIÓN CRUZADA (5-FOLD) ===")
print("="*60)
for metric in scoring:
    scores = [m[metric] for m in fold_metrics if not np.isnan(m[metric])]
    if scores:
        mean, h = mean_ci(scores)
        print(f"{metric:15}: {mean:.3f} ± {h:.3f} (95% CI)")

# Matriz de confusión final
cm_test = confusion_matrix(y_test, y_test_pred_optuna)
print(f"\nMatriz de confusión (Test):\n{cm_test}")

# ========================
# SHAP CORREGIDO - POR PACIENTES (GRUPOS)
# ========================
print("\nCalculando SHAP valores por PACIENTES...")

# Obtener SHAP values para todos los datos de TEST
explainer = shap.TreeExplainer(optuna_model)
shap_values_all = explainer.shap_values(X_test)

# Para binary class, clase positiva
if isinstance(shap_values_all, list):
    shap_values_all = shap_values_all[1]


# Crear DataFrame con los SHAP values y los IDs de pacientes
shap_df = pd.DataFrame(shap_values_all, index=groups_test, columns=X_test.columns)

# Agrupar por paciente (promedio de SHAP values por paciente)
shap_values_patient = shap_df.groupby(level=0).mean()

# También agrupar las características originales por paciente
X_test_patient = X_test.groupby(groups_test).mean()

print(f"SHAP values originales: {shap_values_all.shape} (por horas)")
print(f"SHAP values agregados: {shap_values_patient.shape} (por pacientes)")

# Carpeta de resultados
output_dir = "/mnt/Home-Group/csantos_cps/JosmanDeisy/Modelos/Resultados_LIGHTGBM_Grande"
os.makedirs(output_dir, exist_ok=True)

# Guardar detalle por paciente de utilidad (CSV) para inspección
util_patients_df = pd.DataFrame({
    'Paciente': util_details_test['patient_ids'],
    'ObservedUtility': util_details_test['observed_utils'],
    'BestUtility': util_details_test['best_utils'],
    'InactionUtility': util_details_test['inaction_utils'],
    'WorstUtility': util_details_test['worst_utils'],
})
util_patients_df.to_csv(os.path.join(output_dir, "utility_por_paciente_test.csv"), index=False)

# --- Gráficas CORREGIDAS ---
# 1. Curva ROC
fpr, tpr, _ = roc_curve(y_test, y_test_proba_optuna)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC Curve (AUC = {roc_auc:.3f})")
plt.plot([0,1],[0,1], color="gray", lw=2, linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "ROC_curve_optuna.pdf"), dpi=300)
plt.savefig(os.path.join(output_dir, "ROC_curve_optuna2.pdf"), format="pdf", bbox_inches="tight")

plt.close()

# 2. Matriz de confusión
plt.figure(figsize=(8,6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm_test)
disp.plot(cmap=plt.cm.Blues, values_format='d')
plt.title("Confusion Matrix - Test set")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "confusion_matrix_optuna.pdf"), dpi=300)
plt.savefig(os.path.join(output_dir, "confusion_matrix_optuna2.pdf"), format="pdf", bbox_inches="tight")
plt.close()

# 3. SHAP - Importance global por PACIENTES (bar plot)
plt.figure(figsize=(12,6))
shap.summary_plot(shap_values_patient.values, X_test_patient, plot_type="bar", show=False)
plt.title("Global Feature Importance", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "shap_global_importance.pdf"), dpi=300)

plt.close()

# 4. SHAP - Beeswarm por PACIENTES
plt.figure(figsize=(12,8))
custom_cmap = plt.cm.get_cmap("cool")
shap.summary_plot(shap_values_patient.values, X_test_patient, show=False, cmap=custom_cmap)
plt.title("Local Explanation Summary ", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "shap_beeswarm_pacientes.pdf"), dpi=300)
plt.close()

# 5. Ranking de variables por PACIENTES
mean_abs_shap_patient = np.abs(shap_values_patient.values).mean(axis=0)
shap_importance_df = pd.DataFrame({
    "Variable": X_test_patient.columns,
    "Importance (%)": 100 * mean_abs_shap_patient / mean_abs_shap_patient.sum()
}).sort_values(by="Importance (%)", ascending=False)

shap_importance_df.to_csv(os.path.join(output_dir, "shap_ranking_pacientes.csv"), index=False)

# 6. Gráfico de ranking top 15
top15 = shap_importance_df.head(15)
plt.figure(figsize=(10,6))
ax = sns.barplot(
    data=top15,
    x="Importance (%)",
    y="Variable",
    hue="Variable",
    dodge=False,
    legend=False,
    palette="viridis"
)

for i, v in enumerate(top15["Importance (%)"]):
    ax.text(v + 0.3, i, f"{v:.2f}", color="black", va="center")

plt.title("Global Importance of Variables SHAP)")
plt.xlabel("Importance (%)")
plt.ylabel("")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "shap_ranking_top15_pacientes.pdf"), dpi=300)
plt.close()

# 7. SHAP - Dependence plots para las top 5 variables
top5_vars = shap_importance_df.head(5)["Variable"].tolist()
for i, var in enumerate(top5_vars):
    plt.figure(figsize=(10, 6))
    shap.dependence_plot(
        var, 
        shap_values_patient.values, 
        X_test_patient, 
        show=False
    )
    plt.title(f"SHAP Dependence Plot - {var}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"shap_dependence_{var}.pdf"), dpi=300)
    plt.close()

# Guardar métricas y modelo
metrics_df = pd.DataFrame({
    'CV_Mean': [np.mean([m[metric] for m in fold_metrics if not np.isnan(m[metric])]) for metric in scoring],
    'CV_Std': [np.std([m[metric] for m in fold_metrics if not np.isnan(m[metric])]) for metric in scoring],
    'Test': [test_metrics[metric] for metric in scoring]
}, index=scoring)

metrics_df.to_csv(os.path.join(output_dir, "metricas_comparativas.csv"))



print(f"\nResultados guardados en: {output_dir}")
print(f" SHAP calculado para {shap_values_patient.shape[0]} pacientes")
print(f" Top 5 variables: {', '.join(top5_vars)}")
print("\n Proceso completado exitosamente!")