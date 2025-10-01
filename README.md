# 🧬 Early Detection of Sepsis from Clinical Data using a Machine Learning Model

 the aim of this study is to develop a model for early sepsis detection by implementing a rigorous, clinically-grounded data preprocessing pipeline. This approach prioritizes careful patient selection, advanced handling of missing values, and temporal alignment to create an analysis-ready dataset. This foundation enables the subse quent selection of a machine learning model using a broad set of metrics, including the utility score. 

---

## 🔬 Methodology
The methodology follows a **systematic and clinically-grounded pipeline**:
1. **Data Preprocessing**  
   - Patient-level temporal segmentation (21-hour observation windows).  
   - Sliding windows of 6 hours with 1-hour stride to capture temporal dynamics.  
   - Feature engineering (vital signs, labs, demographics, derived features).  

2. **Imputation**  
   - Multivariate imputation using **LightGBM**.  
   - Hyperparameter tuning with **HalvingRandomSearchCV**.  
   - Z-score normalization of variables.  

3. **Model Development**
   -
   - Gradient Boosting family: **LightGBM, Gradient Boosting, CatBoost, XGBoost**.  
   - Hyperparameter optimization via **Optuna**.  
   - Threshold optimization.  

5. **Evaluation**  
   - Metrics: Accuracy, F1 Score, Precision, ROC-AUC, Sensitivity, Specificity, **Utiity Score**.  
   - 5-fold Stratified Group Cross-Validation.  

6. **Interpretability**  
   - **SHAP values** for feature importance (global and patient-level).  
   - Visualization of key predictors (ICULOS, BUN, respiratory patterns).  

---

## 📂 Repository Structure

