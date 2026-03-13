# Heart Attack Project — Notes

Writing this down so when I come back after a few weeks I don't have to re-read all the code to remember what I was thinking.

---

## The data

Kaggle dataset, 918 patient records. Target is `HeartDisease` — 1 means they have it, 0 means they don't. 508 positive, 410 negative, so reasonably balanced, didn't have to do anything about it.

Did the usual checks upfront — head, info, describe, isnull, duplicated. No missing values, no duplicate rows. Honestly cleaner than most datasets I've seen.

**One thing to set up before any of that** — when a dataframe has many columns, pandas shrinks the output in the terminal and replaces the middle columns with `...`. You only see the first few and last few, which defeats the point of checking the data. Fix it by adding these two lines right after imports:

```python
pd.set_option('display.max_columns', None)   # show all columns
pd.set_option('display.width', None)         # avoid wrapping
```

After this, `print(df.head())` and `print(df.describe())` print the full table without anything being cut off.

---

## Libraries

- `pandas` — loading and working with the data
- `numpy` — mostly used behind the scenes
- `seaborn` and `matplotlib` — all the charts
- `scikit-learn` — just StandardScaler here, nothing else yet

`warnings.filterwarnings('ignore')` is in there to stop sklearn throwing deprecation warnings every run. Nothing important.

---

## EDA

Before touching anything I just plotted a bunch of stuff to get a feel for the data. All the plotting code is commented out — uncomment whatever you want to see.

What I used and why:

- **histplot** — to see the shape of each column's values. Age is clustered around 50–60, not spread evenly.
- **boxplot** — quick way to see outliers. The dots outside the whiskers are the suspicious values.
- **countplot** — for the category columns like Sex and ChestPainType. Just want to know how many of each.
- **violinplot** — used it for age vs heart disease. Boxplot felt too flat for that comparison.
- **heatmap** — threw all numeric columns in there to see if anything jumps out as correlated with HeartDisease.

---

## Data issues

While plotting I noticed Cholesterol and RestingBP had zeros in them. That can't be right — zero cholesterol or zero blood pressure in a living patient is not a real reading, someone just didn't fill it in and it defaulted to 0.

Fixed it by taking the mean of all the non-zero values and replacing the zeros with that.

```
cholesterol mean (non-zero rows) = 244.64
restingBP mean (non-zero rows)   = 132.56
```

I could have used median instead but mean was fine here — checked the distribution after the fix and it looked sensible.

---

## Encoding

Models need numbers, not strings. Used `pd.get_dummies(df, drop_first=True)` to convert all the text columns.

What `drop_first=True` does: say Sex has M and F. You only need one column — if the value is 0 it's F, if it's 1 it's M. Having two columns (one for M, one for F) is redundant, they'd always add up to 1.

Then `.astype(int)` to turn the True/False output of get_dummies into actual 0s and 1s.

Columns this touched: Sex, ChestPainType, RestingECG, ExerciseAngina, ST_Slope.

---

## Scaling

Used StandardScaler on Age, RestingBP, Cholesterol, MaxHR, Oldpeak.

Why: Age tops out around 77. Cholesterol can be 500+. If you leave them as-is, the model sees 500 and 77 and might weigh Cholesterol more just because the number is bigger — not because it's actually more predictive. Scaling centers everything around 0 with a standard deviation of 1, so the model treats them on equal footing.

---

## What's left

- Actually train a model — probably start with Logistic Regression, then try Random Forest
- For medical data, recall matters more than raw accuracy. A false negative (missed heart disease case) is worse than a false alarm.
- Look at feature importance once the model is trained
- Maybe do cross-validation to check the results aren't just luck of the split
