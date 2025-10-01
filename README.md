# 🧬 Early Detection of Sepsis using Machine Learning

## 🎯 Objective
Develop a robust and interpretable machine learning pipeline for **early detection of sepsis**, leveraging temporal structuring, advanced imputation, and boosting-based models (LightGBM, XGBoost, CatBoost).

---

## 🔬 Methodology
The methodology follows a **systematic and clinically-grounded pipeline**:
1. **Data Preprocessing**  
   - Patient-level temporal segmentation (21-hour observation windows).  
   - Sliding windows of 6 hours with 1-hour stride to capture temporal dynamics.  
   - Feature engineering (vital signs, labs, demographics, derived features).  

2. **Imputation**  
   - Multivariate imputation using **LightGBM Iterative Imputer**.  
   - Hyperparameter tuning with **HalvingRandomSearchCV**.  
   - Z-score normalization of variables.  

3. **Model Development**  
   - Gradient Boosting family: **LightGBM, XGBoost, CatBoost**.  
   - Hyperparameter optimization via **Optuna**.  
   - Threshold optimization prioritizing sensitivity and F1-score.  

4. **Evaluation**  
   - Metrics: Accuracy, F1, Precision, Recall, ROC-AUC, Sensitivity, Specificity.  
   - **Clinical Utility Score** (patient-level utility).  
   - 5-fold Stratified Group Cross-Validation.  

5. **Interpretability**  
   - **SHAP values** for feature importance (global and patient-level).  
   - Visualization of key predictors (ICULOS, BUN, respiratory patterns).  

---

## 📂 Repository Structure

