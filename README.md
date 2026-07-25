# Hospital Readmission Risk Prediction

Machine Learning pipeline to predict **30-day hospital readmission risk** using clinical, demographic, laboratory, and hospitalization history data from **101K+ patient encounters**.

## Problem

Hospital readmissions increase healthcare costs and strain hospital resources. This project predicts patients at high risk of readmission, enabling hospitals to plan early interventions and improve patient outcomes.

## Dataset

* **Diabetes 130-US Hospitals Dataset**
* **100K+ hospital encounters**
* Clinical, demographic, medication, laboratory, and admission history features

## Tech Stack

**Python • Pandas • NumPy • Scikit-learn • XGBoost • LightGBM • SHAP • Matplotlib • Seaborn**

## Project Highlights

* Cleaned and preprocessed real-world healthcare data
* Performed Exploratory Data Analysis (EDA) to identify readmission patterns
* Engineered predictive features from admission history, medications, chronic diseases, and lab results
* Compared **Logistic Regression, Random Forest, XGBoost, and LightGBM**
* Evaluated models using **ROC-AUC, Precision, Recall, F1-Score, and Brier Score**
* Built an interpretable prediction pipeline with **SHAP explainability**
* Exported the best-performing model for deployment

## Key Insights

* Elderly patients showed higher readmission rates.
* Congestive Heart Failure and Acute Kidney Failure were associated with higher readmission risk.
* Previous admissions, chronic disease burden, medication count, and hospital stay duration were among the strongest predictors.


