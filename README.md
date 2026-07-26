# Hospital Readmission Risk Prediction

🔗 **Live App:** [hospital-readmission-risk-prediction-using-ml.streamlit.app](https://hospital-readmission-risk-prediction-using-ml.streamlit.app/)

An end-to-end ML pipeline that predicts a diabetic patient's risk of hospital readmission within 30 days of discharge, using clinical, demographic, laboratory, and admission history data from **101,000+ patient encounters**. Deployed as an interactive Streamlit app with single-patient and batch scoring modes.

## Problem
Hospital readmissions drive up healthcare costs and strain hospital capacity. This model flags high-risk patients at discharge so care teams can prioritize follow-up and reduce preventable readmissions.

## Results
- **LightGBM model, ROC-AUC of 0.74**, selected after comparing Logistic Regression, Random Forest, XGBoost, and LightGBM
- SHAP-based explainability layer for per-patient, interpretable risk drivers
- Deployed with a live prediction interface supporting both single-patient input and batch CSV scoring

## Key Insights
- Elderly patients showed markedly higher readmission rates
- Congestive Heart Failure and Acute Kidney Failure were associated with elevated readmission risk
- Prior admission count, chronic disease burden, active medication count, and length of stay were the strongest predictors

## Tech Stack
Python • Pandas • NumPy • Scikit-learn • XGBoost • LightGBM • SHAP • Streamlit • Matplotlib • Seaborn

## Approach
1. Cleaned and preprocessed 101K+ raw hospital encounter records, excluding death/hospice outcomes
2. Engineered features from admission history (prior visits, avg. stay length), medication count, chronic disease burden, and lab abnormality flags
3. Trained and benchmarked four classifiers on ROC-AUC, Precision, Recall, F1, and Brier Score
4. Selected LightGBM as the final model based on discrimination and calibration
5. Built SHAP explainability (TreeExplainer) to surface top risk drivers per prediction
6. Deployed as a Streamlit app with single-patient and batch scoring workflows

## Dataset
[Diabetes 130-US Hospitals dataset](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008) (UCI Machine Learning Repository) — 101,766 encounters across 130 US hospitals.

---
*This tool is a decision-support aid for educational purposes and is not a certified medical device.*
