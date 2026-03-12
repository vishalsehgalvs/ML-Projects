import streamlit as st
import pandas as pd
import joblib

# =============================================================================
# Heart Disease Prediction — Streamlit Web App
# =============================================================================
# This app loads the KNN model trained in ml_project.py and wraps it in a
# simple browser form. The user fills in their health details, clicks Predict,
# and gets a risk assessment.
# =============================================================================

# Load the three files saved during training — model, scaler, and column list.
# All three must come from the same training run or predictions will be wrong.
model = joblib.load("KNN_Model_heart_attack_prediction.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

st.title("Heart Stroke Prediction")
st.markdown("Provide the following details to check your heart stroke risk:")

# -----------------------------------------------------------------------------
# Input form — collect all 11 clinical features from the user
# Streamlit renders these as sliders, dropdowns, and number inputs automatically
# -----------------------------------------------------------------------------
age = st.slider("Age", 18, 100, 40)
sex = st.selectbox("Sex", ["M", "F"])
chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
max_hr = st.slider("Max Heart Rate", 60, 220, 150)
exercise_angina = st.selectbox("Exercise-Induced Angina", ["Y", "N"])
oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

# Everything below runs only when the user clicks the Predict button
if st.button("Predict"):

    # -------------------------------------------------------------------------
    # Step 1 — Build the input as a dictionary
    # Numeric columns go in directly. Categorical columns (Sex, ChestPainType
    # etc.) were one-hot encoded during training, so we replicate that here by
    # building column names like 'Sex_M' or 'ChestPainType_ASY' and setting
    # them to 1. All other one-hot columns for that group will be filled with 0.
    # -------------------------------------------------------------------------
    raw_input = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
        'Sex_' + sex: 1,
        'ChestPainType_' + chest_pain: 1,
        'RestingECG_' + resting_ecg: 1,
        'ExerciseAngina_' + exercise_angina: 1,
        'ST_Slope_' + st_slope: 1
    }

    # Step 2 — Turn the dictionary into a single-row dataframe
    input_df = pd.DataFrame([raw_input])

    # -------------------------------------------------------------------------
    # Step 3 — Fill any missing one-hot columns with 0
    # The model expects every encoded column to be present. If the user picked
    # 'ASY', the other ChestPainType columns (ATA, NAP, TA) won't be in the
    # dict — this loop adds them back as 0.
    # -------------------------------------------------------------------------
    for col in columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # -------------------------------------------------------------------------
    # Step 4 — Reorder columns to exactly match the training data order
    # The model maps features by position, not by name. If the column order
    # doesn't match what it saw during training, predictions will be wrong
    # with no error thrown.
    # -------------------------------------------------------------------------
    input_df = input_df[columns]

    # Step 5 — Scale the input using the same scaler fitted on training data
    scaled_input = scaler.transform(input_df)

    # Step 6 — Run the prediction and show the result
    prediction = model.predict(scaled_input)[0]

    if prediction == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")