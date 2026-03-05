# Insurance ML Project — Notes

Notes I made while going through this project so I can look back and remember what I did and why.

---

## Libraries used

- `pandas` — reads the CSV, lets you work with the data like a table
- `numpy` — maths on arrays, didn't use it much directly here
- `seaborn` / `matplotlib` — for plotting charts
- `scikit-learn` — used for scaling the data
- `scipy` — used for Pearson correlation and Chi-square tests

`warnings.filterwarnings('ignore')` — just stops annoying warning messages from printing, nothing important.

---

## Loading and checking the data

Read the CSV with `pd.read_csv`. After that I just ran a few things to see what the data looks like:

- `df.head()` — first 5 rows
- `df.info()` — column names, data types, any nulls
- `df.describe()` — average, min, max etc for number columns
- `df.isnull().sum()` — check for blank cells (we had none)

---

## EDA (just looking at the data visually)

Before doing anything I plotted the data to see what it looks like. The plot code is commented out in the script — uncomment to view.

- **Distribution plot** — shows how spread out the values are, e.g. are most people aged 20–30 or is it spread out evenly
- **Boxplot** — shows the middle range of values and any outliers (dots that are way outside the normal range)
- **Count plot** — simple bar chart, e.g. how many smokers vs non-smokers
- **Heatmap** — shows which columns are related to each other, darker = stronger relationship

---

## Cleaning the data

- Removed duplicate rows — had 1338, dropped to 1337
- No missing values so nothing to fill in
- Worked on a copy (`df_cleaned = df.copy()`) so the original stays untouched

---

## Encoding (converting text to numbers)

The model only works with numbers so text columns had to be converted.

**For yes/no or male/female columns** — just mapped directly:
- male → 0, female → 1
- yes → 1, no → 0
- Renamed `sex` to `is_female` and `smoker` to `is_smoker` so 0/1 makes sense

**For region** (4 options: northwest, northeast, southwest, southeast) — used one-hot encoding.
Can't just do 1/2/3/4 because that would imply an order that doesn't exist.
Instead creates a separate column for each region with 0 or 1.
`drop_first=True` removes one column since it's redundant — if all three are 0, it must be northeast.

---

## Feature Engineering (creating a new column)

Added a `bmi_category` column by grouping BMI values:
- Under 18.5 → underweight
- 18.5–24.9 → normal
- 25–29.9 → overweight
- 30+ → obese

Raw BMI as a number is harder to work with than a category. Used WHO cut-offs for this.
Then one-hot encoded it too (same as region).

---

## Scaling

Ran `StandardScaler` on `age`, `bmi`, `children`.

Reason: age is 18–64, charges is 1000–60000. Without scaling the model might treat bigger numbers as more important. Scaling brings everything to the same range (mean 0, std 1) without changing the actual relationship between values.

---

## Pearson Correlation

Checked how strongly each column is linearly related to `charges`.

- +1 means as one goes up the other goes up too
- -1 means opposite direction
- 0 means no relationship

```
is_smoker  →  0.787  (strongest — smokers pay a lot more)
age        →  0.298  (older = higher charges)
bmi        →  0.196  (higher bmi = slightly higher charges)
is_female  → -0.058  (barely any relationship)
```

---

## Chi-Square Test

Pearson only works for number columns. For yes/no columns I used chi-square.

It checks: is this column just random noise or does it actually relate to charges?

- p-value below 0.05 → keep the column
- p-value above 0.05 → drop it

```
is_smoker          p ≈ 0.000  → keep
region_southeast   p ≈ 0.001  → keep
is_female          p ≈ 0.016  → keep
bmi_category_obese p ≈ 0.036  → keep
region_southwest   p ≈ 0.165  → drop
bmi_category_normal   p ≈ 0.295  → drop
region_northwest   p ≈ 0.769  → drop
```

---

## Final columns kept

```
age, is_female, bmi, children, is_smoker, region_southeast, bmi_category_obese, charges
```

This is the final cleaned dataset. Next step will be to train a model on this.
