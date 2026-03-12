# Heart Disease Prediction — Multi-Model Comparison

## What it does

Trains and compares five classification algorithms on a heart disease dataset to find the best-performing model, then saves it for use in a Streamlit web app that predicts heart disease risk from patient inputs.

## Dataset

`heart.csv` — 918 rows, 12 columns  
Target: `HeartDisease` (1 = disease present, 0 = no disease)  
508 positive cases (55.3%), 410 negative cases (44.7%)  
No null values. Zeroes in Cholesterol and RestingBP replaced with column mean.

**Features:** Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope

## Model Comparison

| Model                   | Accuracy   | F1 Score   |
| ----------------------- | ---------- | ---------- |
| **Logistic Regression** | **86.96%** | **88.57%** |
| KNN                     | 86.41%     | 88.15%     |
| Naive Bayes             | 85.33%     | 86.83%     |
| SVM                     | 84.78%     | 86.79%     |
| Decision Tree           | 78.26%     | 80.39%     |

Winner: **Logistic Regression** — 86.96% accuracy  
Saved model: **KNN** (exported to `.pkl` for the frontend)

## Project Structure

```
Projects/
├── ml_project.py                        # Model training + comparison
├── heart.csv                            # Dataset
├── KNN_Model_heart_attack_prediction.pkl  # Saved KNN model
├── scaler.pkl                           # Saved StandardScaler
├── columns.pkl                          # Saved feature column list
└── Heart_Disease_Prediction_App/
    └── frontend_project_for_heart_attack_prediction.py  # Streamlit app
```

## Run

**Training & comparison:**

```bash
python ml_project.py
```

**Streamlit frontend:**

```bash
cd Heart_Disease_Prediction_App
streamlit run frontend_project_for_heart_attack_prediction.py
```

_Requires: pandas, numpy, scikit-learn, seaborn, matplotlib, joblib, streamlit_
