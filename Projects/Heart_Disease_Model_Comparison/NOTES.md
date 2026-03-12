# Notes — Heart Disease Prediction (Multi-Model Comparison Project)

This was the first time I actually built something end-to-end — not just running a model to see the accuracy, but cleaning real messy data, comparing five different models in one go, saving the best one, and wrapping it in a working web interface. Felt like a proper project rather than an exercise.

---

## The dataset

`heart.csv` — 918 rows, 12 columns plus the target. No downloaded files needed, it ships with the project. Completely no null values either, which was a nice surprise. But "no nulls" doesn't mean clean — more on that in a second.

The target column is `HeartDisease` — 1 means the patient has heart disease, 0 means they don't. Out of 918 patients, 508 have heart disease and 410 don't. Slightly imbalanced but not badly so.

The features in plain English:

| Column         | What it means                                                                          |
| -------------- | -------------------------------------------------------------------------------------- |
| Age            | Patient age                                                                            |
| Sex            | M or F                                                                                 |
| ChestPainType  | ATA (atypical angina), NAP (non-anginal pain), TA (typical angina), ASY (asymptomatic) |
| RestingBP      | Resting blood pressure in mm Hg                                                        |
| Cholesterol    | Cholesterol level in mg/dL                                                             |
| FastingBS      | Fasting blood sugar > 120 mg/dL (1 = yes, 0 = no)                                      |
| RestingECG     | Resting ECG reading (Normal, ST, LVH)                                                  |
| MaxHR          | Maximum heart rate achieved during stress test                                         |
| ExerciseAngina | Did exercise cause angina? (Y/N)                                                       |
| Oldpeak        | ST depression induced by exercise (a number reflecting heart stress)                   |
| ST_Slope       | Slope of the peak exercise ST segment (Up, Flat, Down)                                 |

---

## The data cleaning problem

Even though there were no null values, cholesterol and resting blood pressure both had rows with value 0. That's medically impossible — your cholesterol and blood pressure can't actually be zero if you're alive. These were clearly bad data entries, probably where measurements weren't taken and someone filled in 0 as a placeholder.

The fix: calculate the average from all the rows that had valid non-zero readings, then replace the zeros with that average. It's not perfect — you're making up values — but it's better than leaving them as 0 which would actively mislead the model.

Same approach for both columns:

- Cholesterol mean (from valid rows): ~244.64
- RestingBP mean (from valid rows): ~132.36

---

## Encoding and scaling

All the text columns — Sex, ChestPainType, RestingECG, ExerciseAngina, ST_Slope — had to be turned into numbers because models only understand numbers. One-hot encoding does this by creating a separate column for each category and marking it 1 or 0.

`drop_first=True` removes one column from each group. If Sex has two options (M and F) and the M column is 0, the model already knows it's F. So you can drop one without losing any information. This avoids a problem called multicollinearity.

Scaling was applied here to all the numeric columns — Age, RestingBP, Cholesterol, MaxHR, Oldpeak. Even for models that technically don't _need_ scaling (like Decision Tree), it doesn't hurt, and since we're running all five models in the same pipeline it was simplest to scale once and feed scaled data to everyone.

---

## Running all five models in a loop

Instead of writing the same fit-predict-evaluate block five times, all five models were put into a dictionary and iterated:

```python
models = {
    'Logistic_Regression': LogisticRegression(),
    'KNN': KNeighborsClassifier(),
    'Naive_Bayes': GaussianNB(),
    'Decision_Tree': DecisionTreeClassifier(),
    'SVM': SVC()
}
```

One loop trains each one, makes predictions, and stores the accuracy and F1 score. Clean and easy to add more models later.

---

## Results

```
Model                Accuracy    F1 Score
-------------------------------------------
Logistic Regression   86.96%      88.57%   ← winner
KNN                   86.41%      88.15%
Naive Bayes           85.33%      86.83%
SVM                   84.78%      86.79%
Decision Tree         78.26%      80.39%
```

**Logistic Regression came out on top** — 86.96% accuracy and an F1 of 0.8857.

A few things stand out:

The top four models (LR, KNN, Naive Bayes, SVM) are all within about 2% of each other. That's a tight cluster. On this dataset, the differences between most models aren't dramatic — they're all picking up more or less the same signal.

Decision Tree is the outlier, dropping to 78%. It tends to memorise the training data rather than generalise, which is the classic overfitting problem. Without tuning (setting max depth, min samples split etc.), it builds a tree that fits the training set perfectly but stumbles on anything new.

This is actually the opposite result from the Titanic comparison earlier where SVM came out on top. Different datasets, different winners — which is why you always compare.

---

## Why Logistic Regression won

Heart disease data tends to have reasonably linear relationships with the outcome — higher age, higher cholesterol, higher blood pressure all push the risk up in a fairly direct way. Logistic Regression is built for exactly this kind of data. It draws a straight boundary (in the scaled feature space) and the heart disease data seems to sit naturally on either side of that boundary.

SVM with RBF kernel can handle curved, non-linear boundaries which is a more powerful tool, but sometimes simpler is better — if the data doesn't need complex curves, Logistic Regression's straightforward approach means less risk of overfitting and a cleaner separation.

---

## Saving the model

Even though Logistic Regression won, the KNN model was saved as the production model. Three files were serialised using `joblib`:

- `KNN_Model_heart_attack_prediction.pkl` — the trained KNN model
- `scaler.pkl` — the fitted scaler (must be the same one used during training)
- `columns.pkl` — the list of feature column names after encoding

Saving the scaler and columns alongside the model is important. When a new prediction comes in, it needs to be transformed exactly the same way the training data was. If you lose the scaler, you can't make valid predictions from the saved model.

---

## The frontend (Streamlit)

`Heart_Disease_Prediction_App/frontend_project_for_heart_attack_prediction.py` loads all three saved files and builds a simple web interface using Streamlit.

The user fills in sliders and dropdowns for all the clinical values — age, sex, chest pain type, blood pressure, cholesterol, etc. When they hit the Predict button, the app:

1. Builds a raw input dictionary from the user's selections
2. One-hot encodes the categorical fields (Sex_M, ChestPainType_ASY, etc.)
3. Fills any missing encoded columns with 0
4. Reorders the columns to match the training data exactly
5. Scales the input with the saved scaler
6. Feeds it to the loaded KNN model
7. Shows either "⚠️ High Risk of Heart Disease" or "✅ Low Risk of Heart Disease"

The column reordering step is easy to miss but critical — if the columns arrive in a different order than the model was trained on, the predictions will be nonsense even though no error is thrown.

---

## Things to remember

- Real data has dirty values even when nulls are zero — always check for impossible values.
- One-hot encoding + drop_first avoids multicollinearity but you have to stay consistent between training and inference.
- Always save the scaler and feature columns alongside the model, not just the model itself.
- Running all models in a loop is cleaner than copy-pasting the same block five times.
- Logistic Regression is often the right first model to try on medical tabular data — it's interpretable, fast, and more robust than trees on small-medium datasets.
- Different datasets produce different winners — the Titanic experiments showed SVM winning; here it's Logistic Regression. There's no universal best.

---

_Final project pulling together all five classification algorithms from the Algorithms/Classification series._
