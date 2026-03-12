# KNN — K-Nearest Neighbours

**Titanic Survival Prediction using the K-Nearest Neighbours Algorithm**

---

## What this project does

This project predicts whether a Titanic passenger survived or died — using only facts we'd know before the ship sank: their age, gender, ticket class, how many family members they had on board, how much they paid, and which port they boarded from.

It builds and compares two classifiers:

1. **Logistic Regression** — run first as a baseline
2. **KNN (K-Nearest Neighbours)** — the main algorithm, with feature scaling applied

---

## The dataset

- **Source:** Built into seaborn — `sns.load_dataset('titanic')`, no CSV needed
- **Rows:** 891 passengers
- **Original columns:** 15
- **Columns used after cleaning:** 8 features + 1 target (`survived`)

### Features used to train the model

| Column     | What it means                              |
| ---------- | ------------------------------------------ |
| `pclass`   | Ticket class — 1st, 2nd, or 3rd            |
| `sex`      | Gender (encoded: female=0, male=1)         |
| `age`      | Age in years                               |
| `sibsp`    | Number of siblings/spouses aboard          |
| `parch`    | Number of parents/children aboard          |
| `fare`     | Ticket price paid                          |
| `embarked` | Port of boarding (C=0, Q=1, S=2)           |
| `alone`    | Whether they were travelling solo (0 or 1) |

### Target

| Column     | What it means          |
| ---------- | ---------------------- |
| `survived` | 1 = survived, 0 = died |

---

## How KNN works (plain English)

Forget the maths for a second. Here's the idea:

> You walk into a new school. You don't know which lunch table to sit at. So you look around, find the 5 kids who seem most similar to you — same hobbies, similar style — and see which table most of them sit at. You sit there too.

That's K-Nearest Neighbours.

- "K" is how many similar passengers you look at — in this project, K=5
- "Nearest" means most similar in terms of numbers (age, fare, class, etc.)
- "Neighbours" are those similar passengers from the training set

When the model sees a new passenger, it doesn't run an equation. It finds the 5 most similar passengers from training, checks how many of them survived, and predicts accordingly.

---

## Why scaling was necessary here

The `fare` column has values from 0 to 500. The `pclass` column only goes from 1 to 3. If you calculate distance without scaling, the fare column dominates everything. The model effectively ignores class, sex, and age.

We used `StandardScaler` to bring all columns to the same scale:

- Every column centred around 0
- Spread adjusted so 1 unit = 1 standard deviation

This is mandatory for KNN. Without it, the results mean nothing.

Logistic Regression doesn't have this problem because it doesn't measure distances — it finds coefficients. So the baseline ran without scaling, and KNN ran with it.

---

## Results

### Logistic Regression (baseline, no scaling)

```
Accuracy: 80.3%

Confusion Matrix:
          Predicted Died   Predicted Survived
Actual Died        90              19
Actual Survived    16              53
```

### KNN — K=5 (with StandardScaler)

```
Accuracy: 78.1%

Confusion Matrix:
          Predicted Died   Predicted Survived
Actual Died        90              19
Actual Survived    20              49
```

### Side-by-side

|                                | Logistic Regression | KNN (K=5) |
| ------------------------------ | ------------------- | --------- |
| Accuracy                       | **80.3%**           | 78.1%     |
| Correct “died” predictions     | 90                  | 90        |
| Correct “survived” predictions | 53                  | 49        |
| Missed survivors               | 16                  | 20        |

Logistic Regression won by a small margin on this dataset. That's not unusual — KNN tends to do better when data has natural clusters or non-linear patterns. On tabular survey-style data like this, simple linear models are hard to beat.

---

## How to run it

**Requirements:**

```
python >= 3.8
pandas
numpy
seaborn
matplotlib
scikit-learn
```

**Install dependencies:**

```bash
pip install pandas numpy seaborn matplotlib scikit-learn
```

**Run:**

```bash
python knn_algorithm.py
```

The results are printed inside the script via commented-out `print()` statements — uncomment whichever output you want to see.

---

## File structure

```
KNN_Algorithm/
├── knn_algorithm.py    ← main script
├── NOTES.md            ← detailed learning notes
└── README.md           ← this file
```

---

## What I learned

- KNN is one of the most intuitive algorithms to understand but one of the easiest to use wrong
- Forgetting to scale your data before running KNN gives misleading results
- Having a baseline (like Logistic Regression) before comparing algorithms is important—otherwise you don't know if your result is good or not
- Odd values of K (3, 5, 7) avoid ties in binary classification
- More complex doesn't always mean more accurate

---

_Part of the Algorithms/Classification series._
