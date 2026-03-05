# Layman Notes — Insurance ML Project

A simple plain-English explanation of every concept used in this project.
No jargon, no maths — just what each thing _actually means_.

---

## 1. Libraries (Tools We Borrowed)

Think of libraries as toolboxes. Instead of building every tool from scratch, we import ready-made ones.

| Library        | What it does in plain English                                               |
| -------------- | --------------------------------------------------------------------------- |
| `pandas`       | Works like Excel in Python — lets you load, edit, and filter tables of data |
| `numpy`        | Does fast maths on lists of numbers                                         |
| `seaborn`      | Draws nice-looking charts and graphs                                        |
| `matplotlib`   | The engine behind the charts — seaborn sits on top of this                  |
| `scikit-learn` | The main ML toolkit — has tools for scaling, encoding, and training models  |
| `scipy`        | Does statistical calculations (correlations, hypothesis tests)              |

```python
warnings.filterwarnings('ignore')
```

> This just tells Python to stop printing yellow warning messages that clutter the output. Nothing important is being hidden — they're just informational.

---

## 2. Loading the Dataset

```python
df = pd.read_csv('insurance.csv')
```

> We're reading a CSV file (basically a spreadsheet saved as plain text) and loading it into a **DataFrame** — which is just a fancy word for a table with rows and columns in Python.

### What we checked after loading:

| Command             | What it tells you                                                                   |
| ------------------- | ----------------------------------------------------------------------------------- |
| `df.head()`         | Show me the first 5 rows — a quick peek at the data                                 |
| `df.info()`         | What columns exist, what type of data is in each (number, text, etc.), any missing? |
| `df.describe()`     | For number columns: what's the average, min, max, spread                            |
| `df.isnull().sum()` | Count missing (blank) values per column — lucky, we had none                        |
| `df.columns`        | List all column names                                                               |

---

## 3. Exploratory Data Analysis (EDA)

EDA = "Let me look at this data before doing anything with it."

Before building any model, a good data scientist always visualises the data first to understand its shape and spot anything unusual.

### Distribution Plot (Histogram + KDE)

> Shows how spread out values are. For example — are most patients aged 20–30, or is it spread across all ages? The smooth curve (KDE) shows the overall trend.

### Boxplot

> A box with whiskers. The box represents the middle 50% of values. Dots outside the whiskers are **outliers** — unusual values that are very far from the rest.

### Count Plot

> A simple bar chart that counts how many times each category appears. For example: how many smokers vs non-smokers in the dataset.

### Heatmap (Correlation Matrix)

> A colour-coded grid. Each cell shows how strongly two columns are related. Dark colours = strong relationship, light = weak. Helps you quickly spot which features might predict charges.

---

## 4. Data Cleaning

### Removing Duplicates

```python
df_cleaned.drop_duplicates(inplace=True)
```

> Some rows were exact copies of each other (1338 → 1337 rows after removing them). Duplicates can confuse a model by making it think some patterns appear more often than they do.

### Checking for Nulls

> "Null" or "NaN" means a cell is blank/missing. ML models can't handle missing data — they need a number in every cell. In our case, we were lucky: no missing values at all.

---

## 5. Encoding — Turning Words into Numbers

ML models only understand numbers. Text columns like `"male"/"female"` or `"yes"/"no"` must be converted.

### Label Encoding (for 2-option columns)

```python
df_cleaned['sex'] = df_cleaned['sex'].map({'male': 0, 'female': 1})
df_cleaned['smoker'] = df_cleaned['smoker'].map({'yes': 1, 'no': 0})
```

> We simply replace each word with a number. Male = 0, Female = 1. No = 0, Yes = 1.
> Columns were also renamed to `is_female` and `is_smoker` to make the meaning of 0/1 obvious.

### One-Hot Encoding (for columns with more than 2 options)

```python
df_cleaned = pd.get_dummies(df_cleaned, columns=['region'], drop_first=True)
```

> `region` has 4 values: northwest, northeast, southwest, southeast.
> We can't just do 1/2/3/4 — that would imply northwest < northeast < southwest, which isn't true.
> Instead we create **separate yes/no columns** for each region:

| region_northwest | region_southeast | region_southwest |
| ---------------- | ---------------- | ---------------- |
| 1                | 0                | 0                |
| 0                | 1                | 0                |

> `drop_first=True` removes one region column to avoid redundancy (if all three are 0, it must be northeast — no need for a 4th column). This is called avoiding the **dummy variable trap**.

---

## 6. Feature Engineering — Creating New Columns

```python
df_cleaned['bmi_category'] = pd.cut(df_cleaned['bmi'], bins=[0, 18.5, 24.9, 29.9, float('inf')],
                                     labels=['underweight', 'normal', 'overweight', 'obese'])
```

> Sometimes raw numbers aren't the most useful form. BMI of 32.5 is less human-readable than "obese".
> We used WHO standard cut-offs to bucket each person's BMI into a clinical category.
> This is called **feature engineering** — using domain knowledge to create a more meaningful column.
> The new `bmi_category` column was then also one-hot encoded (same as region).

---

## 7. Feature Scaling — StandardScaler

```python
cols = ['age', 'bmi', 'children']
scaler = StandardScaler()
df_cleaned[cols] = scaler.fit_transform(df_cleaned[cols])
```

> Imagine comparing age (18–64) with charges (1000–60000). Age looks tiny and charges looks huge, even if both are equally important.
> **Scaling** rescales every column so they all live on the same playing field — average of 0 and spread of 1.
> This doesn't change the _relationship_ between values, just their scale.
> It prevents models from thinking "bigger number = more important."

---

## 8. Pearson Correlation — How Strongly is Each Feature Linked to Charges?

> Pearson correlation measures the **linear relationship** between two columns.
>
> - Score of **+1** = as one goes up, the other always goes up too
> - Score of **-1** = as one goes up, the other always goes down
> - Score of **0** = no relationship at all

```
is_smoker  →  0.787  (very strong — smokers pay far more)
age        →  0.298  (moderate — older = higher charges)
bmi        →  0.196  (moderate — higher BMI = slightly higher charges)
is_female  → -0.058  (near zero — gender barely matters)
```

> Think of it like asking: _"If I know a person smokes, how well can I guess their charges?"_ A score of 0.787 says — quite well!

---

## 9. Chi-Square Test — Does This Category Matter?

> Pearson works for numbers. For categorical (yes/no, 0/1) features, we use the **Chi-square test**.
> It answers: _"Is this category completely random with respect to charges, or does it actually influence charges?"_

### Null Hypothesis

> The default assumption we're testing against: _"This feature has NO relationship with charges."_

### p-value

> The probability that any pattern we see happened purely by chance.
>
> - **p < 0.05** → The pattern is real → Reject the null → **Keep the feature**
> - **p ≥ 0.05** → Could be random noise → Accept the null → **Drop the feature**

```
is_smoker          p ≈ 0.000  → Keep ✅  (extremely significant)
region_southeast   p ≈ 0.001  → Keep ✅
is_female          p ≈ 0.016  → Keep ✅
bmi_category_obese p ≈ 0.036  → Keep ✅
region_southwest   p ≈ 0.165  → Drop ❌  (not significant enough)
region_northwest   p ≈ 0.769  → Drop ❌  (basically random)
```

---

## 10. Final Feature Set

After running both tests, we kept only the features that are statistically important:

```
age | is_female | bmi | children | is_smoker | region_southeast | bmi_category_obese | charges
```

> This is the clean, trimmed dataset ready to hand off to a machine learning model in the next phase of the project.

---

## Key Takeaways

| Concept             | One-liner                                                                      |
| ------------------- | ------------------------------------------------------------------------------ |
| EDA                 | Look before you leap — always explore data visually first                      |
| Encoding            | Models speak numbers, not words — translate before feeding in                  |
| Feature Engineering | Use domain knowledge to create smarter columns                                 |
| Scaling             | Put all features on the same scale so none bully the others                    |
| Pearson Correlation | Measures linear relationship between a feature and the target                  |
| Chi-Square Test     | Tests whether a categorical feature is statistically independent of the target |
| p-value             | Probability the result is due to chance — lower = more trustworthy             |
| Feature Selection   | Keep only what matters — less noise, better models                             |

---

_These notes are a companion to `insurance_ml_project.py`. Written for beginners — no prior statistics or ML background assumed._
