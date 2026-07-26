"""
Hospital 30-Day Readmission Risk Prediction
Streamlit application built on top of the LightGBM model trained in
Hospital_Readmission_Risk_Prediction.ipynb (Diabetes 130-US Hospitals dataset).

Run with:
    streamlit run app.py
"""

import io
import warnings

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------------------
st.set_page_config(
    page_title="Readmission Risk Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = "best_model.joblib"

ALL_MED_COLS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
    "examide", "citoglipton", "insulin", "glyburide-metformin",
    "glipizide-metformin", "glimepiride-pioglitazone", "metformin-rosiglitazone",
    "metformin-pioglitazone",
]

ICD9_MAP = {
    "428": "Congestive Heart Failure", "410": "Acute MI", "486": "Pneumonia",
    "491": "Chronic Bronchitis", "250": "Diabetes", "584": "Acute Kidney Failure",
    "038": "Septicemia", "427": "Cardiac Dysrhythmias",
}

DEATH_HOSPICE_CODES = [11, 13, 14, 19, 20, 21]

FEATURE_TEMPLATES = {
    "num_previous_admissions": {
        "high": "a history of multiple prior hospital admissions",
        "low": "few or no prior admissions",
    },
    "chronic_disease_count": {
        "high": "a high number of chronic conditions on record",
        "low": "a low chronic disease burden",
    },
    "avg_previous_stay": {
        "high": "a pattern of prolonged hospital stays in the past",
        "low": "typically short hospital stays",
    },
    "num_medications_active": {
        "high": "a large number of active medications",
        "low": "a relatively simple medication regimen",
    },
    "any_lab_abnormal": {
        "high": "abnormal recent lab results (glucose/A1C)",
        "low": "normal recent lab results",
    },
    "time_in_hospital": {
        "high": "an unusually long current hospital stay",
        "low": "a short current hospital stay",
    },
}

RECOMMENDATIONS = {
    "num_previous_admissions": "Consider scheduling an early follow-up visit.",
    "chronic_disease_count": "A care coordination review across specialists may help.",
    "avg_previous_stay": "Discharge planning should be reviewed carefully.",
    "num_medications_active": "A medication review to check for adherence issues is recommended.",
    "any_lab_abnormal": "Lab results should be rechecked before discharge.",
    "time_in_hospital": "Ensure a clear discharge and follow-up plan is in place.",
}


# --------------------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model...")
def load_model_bundle(path=MODEL_PATH):
    bundle = joblib.load(path)
    return bundle["model"], bundle["encoders"], bundle["feature_names"]


@st.cache_resource(show_spinner=False)
def get_explainer(_model):
    return shap.TreeExplainer(_model)


def safe_label_encode(encoder, value):
    """Encode a categorical value, falling back to 'Unknown' or the first
    known class if the value was never seen during training."""
    classes = list(encoder.classes_)
    if value in classes:
        return int(encoder.transform([value])[0])
    if "Unknown" in classes:
        return int(encoder.transform(["Unknown"])[0])
    return int(encoder.transform([classes[0]])[0])


def get_diag_group(diag_code):
    prefix = str(diag_code).split(".")[0]
    return ICD9_MAP.get(prefix, "Other")


# --------------------------------------------------------------------------------------
# Feature engineering (single patient form -> model-ready row)
# --------------------------------------------------------------------------------------
def build_feature_row(raw, encoders, feature_names):
    row = dict(raw)

    # Derived diagnosis grouping
    row["diag_1_group"] = get_diag_group(row["diag_1"])

    # visits_since_last mirrors num_previous_admissions in the training pipeline
    row["visits_since_last"] = row["num_previous_admissions"]

    # Medication count
    row["num_medications_active"] = sum(
        1 for c in ALL_MED_COLS if row.get(c, "No") != "No"
    )

    # Chronic disease count = number of distinct non-Unknown diagnosis codes
    diag_values = [row["diag_1"], row["diag_2"], row["diag_3"]]
    row["chronic_disease_count"] = len(
        {d for d in diag_values if d and d != "Unknown"}
    )

    # Lab abnormality flags
    row["glucose_abnormal"] = int(row["max_glu_serum"] in [">200", ">300"])
    row["a1c_abnormal"] = int(row["A1Cresult"] in [">7", ">8"])
    row["any_lab_abnormal"] = int(bool(row["glucose_abnormal"] or row["a1c_abnormal"]))

    # Encode categoricals
    encoded = {}
    for col in feature_names:
        if col in encoders:
            encoded[col] = safe_label_encode(encoders[col], row.get(col, "Unknown"))
        else:
            encoded[col] = row.get(col, 0)

    return pd.DataFrame([encoded], columns=feature_names)


def risk_bucket(score):
    if score >= 0.7:
        return "high", "🔴"
    if score >= 0.4:
        return "moderate", "🟠"
    return "low", "🟢"


def generate_explanation(risk_score, top_contributions):
    risk_level, _ = risk_bucket(risk_score)

    reasons = []
    top_feature = None
    for feature, value in top_contributions.items():
        if feature not in FEATURE_TEMPLATES:
            continue
        direction = "high" if value > 0 else "low"
        reasons.append(FEATURE_TEMPLATES[feature][direction])
        if top_feature is None and value > 0:
            top_feature = feature
        if len(reasons) == 3:
            break

    if not reasons:
        reason_text = "a combination of clinical and utilization factors"
    elif len(reasons) == 1:
        reason_text = reasons[0]
    else:
        reason_text = ", ".join(reasons[:-1]) + f", and {reasons[-1]}"

    recommendation = RECOMMENDATIONS.get(
        top_feature, "Consider a standard post-discharge follow-up."
    )

    return (
        f"This patient has a **{risk_level} predicted readmission risk ({risk_score:.0%})** "
        f"driven by {reason_text}. {recommendation}"
    )


# --------------------------------------------------------------------------------------
# Batch pipeline (raw diabetic_data.csv-style upload)
# --------------------------------------------------------------------------------------
def run_batch_pipeline(df_raw, model, encoders, feature_names):
    df = df_raw.copy()
    df.replace("?", np.nan, inplace=True)

    for col in ["weight", "payer_code", "medical_specialty", "encounter_id"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    if "discharge_disposition_id" in df.columns:
        df = df[~df["discharge_disposition_id"].isin(DEATH_HOSPICE_CODES)]

    cat_like = df.select_dtypes(include="object").columns
    for col in cat_like:
        df[col] = df[col].fillna("Unknown")

    if "diag_1" in df.columns:
        df["diag_1_group"] = df["diag_1"].astype(str).str.split(".").str[0].map(ICD9_MAP).fillna("Other")
    else:
        df["diag_1_group"] = "Other"

    if "patient_nbr" in df.columns:
        df = df.sort_values(["patient_nbr"]).copy()
        df["encounter_order"] = df.groupby("patient_nbr").cumcount()
        df["num_previous_admissions"] = df["encounter_order"]
        df["cumulative_time_in_hospital"] = df.groupby("patient_nbr")["time_in_hospital"].cumsum()
        df["avg_previous_stay"] = (
            (df["cumulative_time_in_hospital"] - df["time_in_hospital"])
            / df["num_previous_admissions"].replace(0, np.nan)
        ).fillna(0)
        df["visits_since_last"] = df["num_previous_admissions"]
        df.drop(columns=["cumulative_time_in_hospital", "encounter_order"], inplace=True)
    else:
        df["num_previous_admissions"] = 0
        df["avg_previous_stay"] = 0.0
        df["visits_since_last"] = 0

    present_meds = [c for c in ALL_MED_COLS if c in df.columns]
    if present_meds:
        df["num_medications_active"] = (df[present_meds] != "No").sum(axis=1)
    else:
        df["num_medications_active"] = 0

    diag_cols = [c for c in ["diag_1", "diag_2", "diag_3"] if c in df.columns]
    if diag_cols:
        df["chronic_disease_count"] = df[diag_cols].apply(
            lambda r: r.replace("Unknown", np.nan).nunique(), axis=1
        )
    else:
        df["chronic_disease_count"] = 0

    df["glucose_abnormal"] = df.get("max_glu_serum", pd.Series("Unknown", index=df.index)).isin([">200", ">300"]).astype(int)
    df["a1c_abnormal"] = df.get("A1Cresult", pd.Series("Unknown", index=df.index)).isin([">7", ">8"]).astype(int)
    df["any_lab_abnormal"] = df["glucose_abnormal"] | df["a1c_abnormal"]

    encoded = pd.DataFrame(index=df.index)
    for col in feature_names:
        if col in encoders:
            enc = encoders[col]
            series = df.get(col, pd.Series("Unknown", index=df.index)).astype(str).fillna("Unknown")
            encoded[col] = series.apply(lambda v: safe_label_encode(enc, v))
        else:
            encoded[col] = pd.to_numeric(df.get(col, 0), errors="coerce").fillna(0)

    proba = model.predict_proba(encoded[feature_names])[:, 1]
    out = df_raw.loc[df.index].copy()
    out["readmission_risk"] = proba
    out["risk_level"] = [risk_bucket(p)[0] for p in proba]
    return out.sort_values("readmission_risk", ascending=False)


# --------------------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------------------
def sidebar_info():
    st.sidebar.title("🏥 Readmission Risk")
    st.sidebar.markdown(
        "Predicts the probability that a diabetic patient will be "
        "**readmitted within 30 days** of hospital discharge, using a "
        "LightGBM model trained on the Diabetes 130-US Hospitals dataset."
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**Risk bands**\n\n"
        "🟢 Low: < 40%\n\n"
        "🟠 Moderate: 40 – 70%\n\n"
        "🔴 High: ≥ 70%"
    )
    st.sidebar.markdown("---")
    st.sidebar.caption(
        "This tool is a decision-support aid only and does not replace "
        "clinical judgment."
    )


def selectbox_for(col, encoders, label=None, default=None, help_text=None):
    options = list(encoders[col].classes_)
    index = options.index(default) if default in options else 0
    return st.selectbox(label or col, options, index=index, help=help_text, key=f"in_{col}")


def single_patient_tab(model, encoders, feature_names, explainer):
    st.header("Single Patient Risk Assessment")
    st.caption(
        "Fill in the patient's clinical and administrative details, then "
        "click **Predict Readmission Risk**."
    )

    with st.form("patient_form"):
        st.subheader("Demographics")
        c1, c2, c3 = st.columns(3)
        with c1:
            race = selectbox_for("race", encoders, default="Caucasian")
        with c2:
            gender = selectbox_for("gender", encoders, default="Female")
        with c3:
            age = selectbox_for("age", encoders, default="[70-80)")

        st.subheader("Admission Details")
        c1, c2, c3 = st.columns(3)
        with c1:
            admission_type_id = st.number_input(
                "Admission type ID", min_value=1, max_value=8, value=1,
                help="1=Emergency, 2=Urgent, 3=Elective, 4=Newborn, 5-8=Other/Unknown",
            )
        with c2:
            discharge_disposition_id = st.number_input(
                "Discharge disposition ID", min_value=1, max_value=30, value=1,
                help="1=Discharged to home. See dataset codebook for full list.",
            )
        with c3:
            admission_source_id = st.number_input(
                "Admission source ID", min_value=1, max_value=25, value=7,
                help="7=Emergency room, 1=Physician referral, etc.",
            )

        st.subheader("Current Stay & Utilization")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            time_in_hospital = st.number_input("Days in hospital (current stay)", 1, 14, 4)
        with c2:
            num_lab_procedures = st.number_input("Lab procedures performed", 0, 132, 43)
        with c3:
            num_procedures = st.number_input("Procedures performed", 0, 6, 1)
        with c4:
            num_medications = st.number_input("Distinct medications ordered", 1, 81, 16)

        c1, c2, c3 = st.columns(3)
        with c1:
            number_outpatient = st.number_input("Outpatient visits (past year)", 0, 40, 0)
        with c2:
            number_emergency = st.number_input("Emergency visits (past year)", 0, 40, 0)
        with c3:
            number_inpatient = st.number_input("Inpatient visits (past year)", 0, 40, 0)

        c1, c2, c3 = st.columns(3)
        with c1:
            number_diagnoses = st.number_input("Number of diagnoses recorded", 1, 16, 9)
        with c2:
            num_previous_admissions = st.number_input(
                "Prior admissions for this patient", 0, 50, 0,
                help="Count of this patient's hospital encounters prior to this one.",
            )
        with c3:
            avg_previous_stay = st.number_input(
                "Average length of prior stays (days)", 0.0, 30.0, 0.0, step=0.5,
                help="Average 'time in hospital' across the patient's prior admissions. Leave 0 if no prior admissions.",
            )

        st.subheader("Diagnoses (ICD-9 codes)")
        c1, c2, c3 = st.columns(3)
        with c1:
            diag_1 = st.text_input("Primary diagnosis code (diag_1)", value="250", help="e.g. 250=Diabetes, 428=CHF, 486=Pneumonia")
        with c2:
            diag_2 = st.text_input("Secondary diagnosis code (diag_2)", value="401")
        with c3:
            diag_3 = st.text_input("Tertiary diagnosis code (diag_3)", value="272")

        st.subheader("Lab Results")
        c1, c2 = st.columns(2)
        with c1:
            max_glu_serum = selectbox_for("max_glu_serum", encoders, "Max glucose serum test result", default="Unknown")
        with c2:
            a1c_result = selectbox_for("A1Cresult", encoders, "A1C test result", default="Unknown")

        st.subheader("Diabetes Management")
        c1, c2 = st.columns(2)
        with c1:
            change = selectbox_for("change", encoders, "Medication change during encounter", default="No")
        with c2:
            diabetes_med = selectbox_for("diabetesMed", encoders, "On diabetes medication", default="Yes")

        with st.expander("Individual medication statuses (defaults to 'No')"):
            med_values = {}
            med_cols_ui = st.columns(4)
            for i, med in enumerate(ALL_MED_COLS):
                with med_cols_ui[i % 4]:
                    med_values[med] = selectbox_for(med, encoders, label=med, default="No")

        submitted = st.form_submit_button("🔍 Predict Readmission Risk", use_container_width=True)

    if not submitted:
        return

    raw = {
        "race": race, "gender": gender, "age": age,
        "admission_type_id": admission_type_id,
        "discharge_disposition_id": discharge_disposition_id,
        "admission_source_id": admission_source_id,
        "time_in_hospital": time_in_hospital,
        "num_lab_procedures": num_lab_procedures,
        "num_procedures": num_procedures,
        "num_medications": num_medications,
        "number_outpatient": number_outpatient,
        "number_emergency": number_emergency,
        "number_inpatient": number_inpatient,
        "diag_1": diag_1.strip(), "diag_2": diag_2.strip(), "diag_3": diag_3.strip(),
        "number_diagnoses": number_diagnoses,
        "max_glu_serum": max_glu_serum, "A1Cresult": a1c_result,
        "change": change, "diabetesMed": diabetes_med,
        "num_previous_admissions": num_previous_admissions,
        "avg_previous_stay": avg_previous_stay,
        **med_values,
    }

    X_row = build_feature_row(raw, encoders, feature_names)
    risk_score = float(model.predict_proba(X_row)[0, 1])
    risk_level, emoji = risk_bucket(risk_score)

    st.markdown("---")
    st.subheader("Prediction Result")

    r1, r2 = st.columns([1, 2])
    with r1:
        st.metric("Predicted 30-day readmission risk", f"{risk_score:.1%}")
        st.markdown(f"### {emoji} Risk level: **{risk_level.upper()}**")
        st.progress(min(max(risk_score, 0.0), 1.0))

    with r2:
        shap_values = explainer(X_row)
        vals = shap_values.values[0]
        contributions = pd.Series(vals, index=X_row.columns).sort_values(key=abs, ascending=False)
        explanation = generate_explanation(risk_score, contributions.head(8))
        st.info(explanation)

        top8 = contributions.head(8).sort_values()
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = ["#d62728" if v > 0 else "#2ca02c" for v in top8.values]
        ax.barh(top8.index, top8.values, color=colors)
        ax.set_xlabel("SHAP value (impact on predicted risk)")
        ax.set_title("Top contributing features")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with st.expander("Show full SHAP waterfall plot"):
        fig2 = plt.figure(figsize=(9, 6))
        shap.plots.waterfall(shap_values[0], show=False)
        st.pyplot(fig2)
        plt.close(fig2)

    with st.expander("Show model input row (encoded features sent to model)"):
        st.dataframe(X_row.T.rename(columns={0: "value"}))


def batch_tab(model, encoders, feature_names):
    st.header("Batch Scoring")
    st.caption(
        "Upload a CSV in the same raw format as the training data "
        "(`diabetic_data.csv` from the Diabetes 130-US Hospitals dataset) "
        "to score many encounters at once."
    )

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is None:
        st.info("Waiting for a CSV upload. Expected columns include: race, gender, age, "
                "admission_type_id, discharge_disposition_id, admission_source_id, "
                "time_in_hospital, diag_1, diag_2, diag_3, medication columns, etc.")
        return

    try:
        df_raw = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        return

    st.write(f"Loaded **{df_raw.shape[0]:,}** rows, **{df_raw.shape[1]}** columns.")

    with st.spinner("Running feature engineering and scoring..."):
        try:
            result = run_batch_pipeline(df_raw, model, encoders, feature_names)
        except Exception as e:
            st.error(f"Pipeline failed: {e}")
            return

    st.success("Scoring complete.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Patients scored", f"{len(result):,}")
    c2.metric("Avg. predicted risk", f"{result['readmission_risk'].mean():.1%}")
    c3.metric("High-risk patients (≥70%)", f"{(result['readmission_risk'] >= 0.7).sum():,}")

    st.bar_chart(result["risk_level"].value_counts())

    st.dataframe(
        result[[c for c in result.columns if c in
                ["patient_nbr", "age", "race", "gender", "readmission_risk", "risk_level"]] +
               [c for c in result.columns if c not in
                ["patient_nbr", "age", "race", "gender", "readmission_risk", "risk_level"]]],
        use_container_width=True,
        height=420,
    )

    csv_bytes = result.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download scored results (CSV)",
        data=csv_bytes,
        file_name="readmission_risk_scores.csv",
        mime="text/csv",
        use_container_width=True,
    )


def about_tab():
    st.header("About this Model")
    st.markdown(
        """
This app serves a **LightGBM** binary classifier trained to predict whether a
diabetic patient will be readmitted to hospital **within 30 days** of discharge.

**Dataset:** Diabetes 130-US Hospitals dataset (~101,000 encounters).

**Modeling pipeline (from the accompanying notebook):**
- Removed encounters ending in death/hospice (not meaningfully "readmittable").
- Engineered features: prior admission count, average prior stay length,
  active medication count, chronic disease count, and lab abnormality flags.
- Compared Logistic Regression, Random Forest, XGBoost, and LightGBM.
- **LightGBM was selected as the final model (ROC-AUC ≈ 0.738)** based on
  discrimination and calibration performance.
- SHAP (TreeExplainer) is used here to explain individual predictions.

**Risk bands used in this app:**
- 🟢 Low risk: predicted probability < 40%
- 🟠 Moderate risk: 40% – 70%
- 🔴 High risk: ≥ 70%

**Important:** This tool is intended for decision support and educational
purposes only. It is not a certified medical device and should not be used
as the sole basis for clinical decisions.
        """
    )


def main():
    sidebar_info()
    model, encoders, feature_names = load_model_bundle()
    explainer = get_explainer(model)

    tab1, tab2, tab3 = st.tabs(["🧑‍⚕️ Single Patient", "📊 Batch Scoring", "ℹ️ About"])
    with tab1:
        single_patient_tab(model, encoders, feature_names, explainer)
    with tab2:
        batch_tab(model, encoders, feature_names)
    with tab3:
        about_tab()


if __name__ == "__main__":
    main()