# Titanic — who survived and who didn't

Picked the Titanic dataset for this one. The job is to predict whether a passenger survived or not — so the answer is always just yes or no, 0 or 1. That's different from the previous projects where we were guessing a number like charges or car price. Different problem means a different model and a different way of checking if it worked.

---

## What the dataset looks like

Comes built into seaborn, no CSV to download. 891 passengers, 15 columns. After cleaning it up 9 columns stayed.

| Column   | What it is                                                    |
| -------- | ------------------------------------------------------------- |
| survived | 0 = died, 1 = survived — this is what we're trying to guess   |
| pclass   | ticket class — 1st, 2nd or 3rd                                |
| sex      | male or female                                                |
| age      | how old the passenger was                                     |
| sibsp    | siblings or spouse also on the ship                           |
| parch    | parents or kids also on the ship                              |
| fare     | what they paid for the ticket                                 |
| embarked | which port they got on — Southampton, Cherbourg or Queenstown |
| alone    | were they travelling alone or not                             |

---

## What I did

**Step 1 — looked at the data first**

Ran `df.info()` and `df.isnull().sum()` to see what columns exist, what type of data is in them, and which ones have gaps.

**Step 2 — dropped 6 columns**

- `deck` — 688 out of 891 rows had no value. If you fill 688 blanks with an average you're basically making data up. Dropped it.
- `alive` — says "yes" or "no" for whether the person survived. That's the exact answer we're predicting. Keeping it during training would be cheating — score would look perfect but mean nothing.
- `class` and `embark_town` — same data as `pclass` and `embarked`, just written differently. No point having both.
- `who` and `adult_male` — already covered by `sex` and `age`.

**Step 3 — fixed missing values**

- `age` had 177 blanks. Filled them with the average age (~29.7). Not perfect but workable for 20% missing.
- `embarked` had 2 blanks. Dropped those 2 rows — 891 → 889, barely noticeable.

**Step 4 — converted text to numbers**

Model can't work with words:

- `sex` → female=0, male=1
- `embarked` → Cherbourg=0, Queenstown=1, Southampton=2
- `alone` was True/False — ran `astype(int)` to flip everything to 0s and 1s

**Step 5 — trained the model**

80% of rows for training, 20% held back for testing. Trained Logistic Regression.

**Step 6 — checked how it did**

R² doesn't work here since we're predicting a category not a number. Used confusion matrix and classification report.

---

## Why Logistic Regression and not Linear Regression

Linear Regression can spit out any number — 1.7, -0.3, 500. That makes no sense when the only valid answers are 0 or 1.

Logistic Regression runs the result through the sigmoid formula, which clamps the output between 0 and 1 no matter what:

$$P = \frac{1}{1 + e^{-z}}$$

$z$ is the same weighted column sum as Linear Regression. The sigmoid just turns it into a probability. Above 0.5 → predict 1 (survived). Below 0.5 → predict 0 (didn't survive).

---

## Checking if it worked — confusion matrix

When you're predicting a category, R² is useless. The question is just: how often did you pick the right box. Confusion matrix shows that as a 2x2 table:

|                              | Predicted: didn't survive      | Predicted: survived            |
| ---------------------------- | ------------------------------ | ------------------------------ |
| **Actually: didn't survive** | True Negative (TN) — correct ✓ | False Positive (FP) — wrong ✗  |
| **Actually: survived**       | False Negative (FN) — wrong ✗  | True Positive (TP) — correct ✓ |

Our numbers:

```
[[90  19]
 [16  53]]
```

- 90 didn't survive, model said no — correct
- 53 survived, model said yes — correct
- 19 didn't survive but model said yes — wrong (false alarm)
- 16 survived but model said no — wrong (missed them)

From those 4 numbers you get 4 scores:

**Accuracy** — what fraction of all guesses were right

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

**Precision** — of all the people the model said survived, how many actually did

$$\text{Precision} = \frac{TP}{TP + FP}$$

**Recall** — of all the people who actually survived, how many did the model catch

$$\text{Recall} = \frac{TP}{TP + FN}$$

**F1** — one number that covers both Precision and Recall together

$$\text{F1} = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

---

## Results

|           | Didn't survive | Survived |
| --------- | -------------- | -------- |
| Precision | 0.85           | 0.74     |
| Recall    | 0.83           | 0.77     |
| F1        | 0.84           | 0.75     |

**Overall accuracy: 80.3%**

Did better on non-survivors than survivors. Test set had 109 non-survivors vs 69 survivors — more non-survivor examples to learn from during training, so naturally better on that side.

---

## Libraries used

- pandas, numpy — working with the data
- seaborn — dataset comes from here
- scikit-learn — LabelEncoder, train_test_split, LogisticRegression, accuracy_score, confusion_matrix, classification_report

---

## How to run

1. No CSV needed — `sns.load_dataset('titanic')` pulls it automatically
2. Run `logistical_regression.py`
3. Uncomment `print()` lines to see individual step outputs

---

_First classification model. Titanic from seaborn._
