# Heart Attack Prediction

Kaggle dataset on heart disease. I wanted to work with medical data for a change — the usual Titanic/iris stuff gets boring. The goal here is to figure out which patient details are actually useful for predicting heart disease, and get the data into a shape where a model can actually be trained on it.

---

## Files

- **heartattack.py** — all the code, from loading the data to final cleaned output
- **heart.csv** — the dataset (918 patients, 12 columns)
- **images/** — charts saved during exploration

---

## Dataset

918 rows, one per patient. 12 columns total.

| Column         | What it means                                                                         |
| -------------- | ------------------------------------------------------------------------------------- |
| Age            | age in years                                                                          |
| Sex            | M or F                                                                                |
| ChestPainType  | ASY = no symptoms, NAP = non-anginal pain, ATA = atypical angina, TA = typical angina |
| RestingBP      | blood pressure at rest (mm Hg)                                                        |
| Cholesterol    | cholesterol in mg/dl                                                                  |
| FastingBS      | fasting blood sugar — 1 if over 120 mg/dl, 0 if not                                   |
| RestingECG     | ECG result at rest — Normal, ST, or LVH                                               |
| MaxHR          | highest heart rate recorded                                                           |
| ExerciseAngina | chest pain during exercise — Y or N                                                   |
| Oldpeak        | ST depression on the ECG (a numeric reading)                                          |
| ST_Slope       | slope of the ST peak — Up, Flat, or Down                                              |
| HeartDisease   | 1 = has it, 0 = doesn't                                                               |

508 patients with heart disease, 410 without. Not perfectly even but close enough — didn't need to do anything special for imbalance.

---

## What I did

Started by just loading the data and running the usual checks — head, info, describe, null counts. This dataset was actually quite clean, no missing values and no duplicate rows, which was a nice surprise.

After that I spent a while just plotting things. Distributions to see the spread of each column, boxplots to check for outliers, countplots for the category columns, a violin plot to compare age across heart disease groups, and a heatmap to see which columns have any relationship with each other. All of that plotting code is commented out in the script — just uncomment whichever chart you want to see.

While looking at the data I noticed two columns had zeros that make no sense medically — cholesterol and resting blood pressure. Zero cholesterol or zero blood pressure in a patient record means the data wasn't collected, it got entered as 0 by mistake. Fixed both by replacing 0s with the mean of all the valid (non-zero) values in that column.

Text columns had to be converted to numbers since the model can't work with words. Used `pd.get_dummies()` with `drop_first=True` and then `.astype(int)` to get proper 0/1 integers. The affected columns were Sex, ChestPainType, RestingECG, ExerciseAngina, and ST_Slope.

Finally scaled the numeric columns (Age, RestingBP, Cholesterol, MaxHR, Oldpeak) with `StandardScaler`. Age ranges from 28–77 but Cholesterol can be up to 600 — without scaling those big numbers would throw things off during training.

---

## Visualizations

### Feature Distributions

![All Numeric Distributions](images/all_numeric_distributions.png)
![Distributions After Fix](images/numeric_distributions_after_fix.png)

### Age

![Age Boxplot](images/age_boxplot.png)
![Age vs Heart Disease Violin](images/age_vs_heart_disease_violin.png)

### Gender

![Gender Distribution](images/gender_distribution.png)
![Gender Boxplot](images/gender_boxplot.png)
![Gender vs Heart Disease](images/gender_vs_heart_disease.png)

### Chest Pain

![Chest Pain Types](images/chest_pain_types_distribution.png)
![Chest Pain Boxplot](images/chest_pain_boxplot.png)
![Chest Pain vs Heart Disease](images/chest_pain_vs_heart_disease.png)

### Cholesterol

![Cholesterol Zero Anomaly](images/cholesterol_zero_anomaly.png)
![Cholesterol After Fix](images/cholesterol_after_fix.png)
![Cholesterol Boxplot](images/cholesterol_boxplot.png)
![Cholesterol vs Heart Disease](images/cholesterol_vs_heart_disease_boxplot.png)

### Resting Blood Pressure

![RestingBP Boxplot](images/resting_bp_boxplot.png)
![RestingBP After Fix](images/resting_bp_after_fix.png)

### Other Features

![Max Heart Rate](images/max_heart_rate_boxplot.png)
![Exercise Angina](images/exercise_angina_boxplot.png)
![Fasting Blood Sugar](images/fasting_blood_sugar_boxplot.png)
![Oldpeak](images/oldpeak_boxplot.png)
![Resting ECG](images/resting_ecg_boxplot.png)
![ST Slope](images/st_slope_boxplot.png)

### Heart Disease

![Heart Disease Distribution](images/heart_disease_distribution.png)
![Heart Disease Boxplot](images/heart_disease_boxplot.png)

### Correlation

![Feature Correlation Heatmap](images/feature_correlation_heatmap.png)

---

## Libraries

`numpy`, `pandas`, `seaborn`, `matplotlib`, `scikit-learn`

---

## A few rows from the dataset

| Age | Sex | ChestPainType | RestingBP | Cholesterol | HeartDisease |
| --- | --- | ------------- | --------- | ----------- | ------------ |
| 40  | M   | ATA           | 140       | 289         | 0            |
| 49  | F   | NAP           | 160       | 180         | 1            |
| 37  | M   | ATA           | 130       | 283         | 0            |
