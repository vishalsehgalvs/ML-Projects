# Notes — Heart Disease Prediction Frontend (Streamlit)

This was the part that made the whole project feel real. Up until this point everything was just numbers printed in a terminal. Building the frontend turned it into something you could actually hand to someone and say "try this".

---

## What this does

It's a web app that runs in the browser. You open a terminal, run one command, and a form pops up in your browser. Fill in your health details, hit Predict, and it tells you whether you're at high or low risk for heart disease.

Streamlit makes this possible without writing any HTML or setting up a web server. You just write Python, and Streamlit draws the buttons, sliders, and dropdowns for you.

---

## The three saved files it depends on

Back in `ml_project.py`, three files were saved to disk after training:

- `KNN_Model_heart_attack_prediction.pkl` — the trained KNN model itself
- `scaler.pkl` — the scaler that was used to normalise the training data
- `columns.pkl` — the exact list of feature columns, in the exact order, after encoding

All three need to be present and they need to come from the same training run. The scaler "remembers" the min/max values from the training data — if you swap it for a new one, the numbers won't be on the right scale and the model will give wrong results. Same story with columns — the model was trained expecting features in a specific order, and if that changes, predictions silently go wrong.

---

## The input form

Streamlit widgets collect all 11 clinical features:

| Input                     | Widget type  | Options / Range   |
| ------------------------- | ------------ | ----------------- |
| Age                       | Slider       | 18 – 100          |
| Sex                       | Dropdown     | M, F              |
| Chest Pain Type           | Dropdown     | ATA, NAP, TA, ASY |
| Resting Blood Pressure    | Number input | 80 – 200 mm Hg    |
| Cholesterol               | Number input | 100 – 600 mg/dL   |
| Fasting Blood Sugar > 120 | Dropdown     | 0, 1              |
| Resting ECG               | Dropdown     | Normal, ST, LVH   |
| Max Heart Rate            | Slider       | 60 – 220          |
| Exercise Angina           | Dropdown     | Y, N              |
| Oldpeak                   | Slider       | 0.0 – 6.0         |
| ST Slope                  | Dropdown     | Up, Flat, Down    |

---

## What happens when Predict is clicked

This is the trickiest part. You can't just dump the form values straight into the model — you have to do the same preparation steps that were done on the training data, in the same order.

**Step 1 — Build the raw input**

The model doesn't understand words like "Male" or "ASY". During training, one-hot encoding turned those text values into numbered columns — `Sex_M`, `ChestPainType_ASY`, etc. The frontend manually does the same thing by constructing column names from the user's selection:

```python
'Sex_' + sex: 1               # if sex = 'M', this becomes Sex_M: 1
'ChestPainType_' + chest_pain: 1   # e.g. ChestPainType_ASY: 1
```

The selected option gets a 1. All other options for that group are left out of the dictionary (and filled with 0 in step 2).

**Step 2 — Fill in the blanks with 0**

The model expects every encoded column to be present even if it wasn't selected. If the user picked ASY, then ChestPainType_ATA, ChestPainType_NAP, and ChestPainType_TA all need to exist in the input with a value of 0. This loop handles that.

**Step 3 — Put the columns in the right order**

The model was trained on columns in a specific order. It doesn't read column names during prediction — it reads column positions. So column 0 must be the same feature it was at training time, column 1 must match, and so on. `columns.pkl` stores that exact order, and `input_df[columns]` enforces it.

**Step 4 — Scale**

The saved scaler applies the same numeric transformation that was applied to the training data. This is why saving the scaler matters — it remembers the exact values from training.

**Step 5 — Predict and show the result**

```python
prediction = model.predict(scaled_input)[0]
```

- 1 → red warning — High Risk
- 0 → green success — Low Risk

---

## Why column order is so annoying and so important

This is the kind of thing that trips everyone up at least once. Sklearn models remember your training features by _position_, not by _name_. So if the training data had `[Age, Sex_F, Sex_M, ...]` and your prediction input arrives as `[Sex_M, Age, Sex_F, ...]`, the model silently maps Sex_M's value into Age's slot. The prediction comes out wrong and there's no error message telling you why.

`input_df[columns]` is one line but it's doing something critical — it forces the input to always match the training order exactly.

---

## Running it

```bash
cd Heart_Disease_Prediction_App
streamlit run frontend_project_for_heart_attack_prediction.py
```

Opens in the browser at `http://localhost:8501`.

Make sure the three `.pkl` files are in the same directory as the script when you run it.

---

## Things to remember

- Streamlit is the simplest way to turn a saved sklearn model into a browser app — no web dev knowledge needed.
- The pkl files (model, scaler, columns) are a matched set — update all three if you retrain.
- One-hot encoding must be replicated manually on the frontend input, or predictions will be garbage.
- Column reordering before prediction is critical and silent — wrong order produces wrong results with no error.
- The `for col in columns: if col not in input_df` loop is doing the job of making sure all expected columns are present even when a category wasn't selected.

---

_Frontend for the multi-model heart disease comparison project in the parent Projects/ folder._
