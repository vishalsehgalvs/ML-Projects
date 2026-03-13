# 📝 Model Tuning — Study Notes

---

## The Big Picture

Model tuning is about two things:

1. **Properly evaluating** your model (so you know its real performance, not a lucky number)
2. **Improving** your model through better data prep, parameter choices, and validation techniques

This file focuses on the evaluation side — specifically **K-Fold Cross Validation** — which is the central concept in this project.

---

## Why Your First Accuracy Number Can Lie to You

When you build your first model, you split data 80/20, train, test, and get "84% accuracy!" — and feel great.

But here's the problem: that number depends entirely on which rows landed in the test set. If easy examples were in the test set, you get a high score. If hard ones were, you get a low score. You tested on ONE random slice and called it done.

```
╔══════════════════════════════════════════════════════════╗
║  "My model is 84% accurate"                              ║
║                                                          ║
║  Is it?                                                  ║
║  Or did you just get lucky with that one 20% slice?     ║
║                                                          ║
║  You don't know. That's the problem.                    ║
╚══════════════════════════════════════════════════════════╝
```

---

## What is K-Fold Cross Validation?

### The idea in one sentence

Split your data into K equal pieces (folds), train/test K times — each time using a different fold as the test set — then average all K scores.

### Why "K"?

K is just a number you choose. The most common choice is **K=5** or **K=10**.

### Walkthrough with K=5

```
Full Dataset (889 rows)
│
└── Split into 5 equal folds (~178 rows each)
    [Fold 1] [Fold 2] [Fold 3] [Fold 4] [Fold 5]


Iteration 1:  [TEST  ] [Train ] [Train ] [Train ] [Train ] → Accuracy: 83.1%
Iteration 2:  [Train ] [TEST  ] [Train ] [Train ] [Train ] → Accuracy: 82.0%
Iteration 3:  [Train ] [Train ] [TEST  ] [Train ] [Train ] → Accuracy: 81.5%
Iteration 4:  [Train ] [Train ] [Train ] [TEST  ] [Train ] → Accuracy: 80.9%
Iteration 5:  [Train ] [Train ] [Train ] [Train ] [TEST  ] → Accuracy: 86.4%

Final Score = (83.1 + 82.0 + 81.5 + 80.9 + 86.4) / 5 = 82.8%
```

**Key property:** Every single row gets used as TEST data exactly once. Nothing is wasted. Nothing is over-sampled.

### In code

```python
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = SVC()
scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')
# cv=5 means 5 folds
# scoring='accuracy' means we want accuracy as the metric

print(scores)         # [0.831, 0.820, 0.814, 0.808, 0.864]
print(scores.mean())  # 0.8279 → 82.8%
```

> Important: `cross_val_score` handles the train/test splitting internally. You pass it the full X and y — it does the rest.

---

## What the Individual Fold Scores Tell You

Looking at SVM's fold scores: `[83.1, 82.0, 81.5, 80.9, 86.4]`

- The range is **80.9% to 86.4%** — about 5.5% spread
- This tells you the model is **reasonably consistent**
- A huge spread (e.g. 60% to 95%) would mean the model is **unstable** — heavily dependent on which data it sees

Looking at KNN's fold scores: `[78.7, 76.4, 82.6, 81.5, 80.2]`

- Range: **76.4% to 82.6%** — about 6.2% spread
- More variable than SVM
- Mean (79.9%) is also lower

```
╔═══════════════════════════════════════════════════════════╗
║  High mean + low spread = good, stable, trustworthy model ║
║  High mean + high spread = good but unreliable            ║
║  Low mean + low spread = bad but at least consistently bad║
╚═══════════════════════════════════════════════════════════╝
```

---

## StandardScaler — Why Scaling Matters Before KNN and SVM

### The problem

Different features have wildly different scales:

- `fare`: 0 to 512
- `pclass`: 1, 2, or 3
- `sex`: 0 or 1
- `age`: 0 to 80

KNN uses **distance** to find similar passengers. If `fare` ranges 0-512 and `sex` ranges 0-1, the distance calculation is dominated entirely by `fare` — `sex` barely matters.

SVM also uses distance when finding the optimal separating boundary.

### What StandardScaler does

For each feature, it transforms the values so:

- **Mean becomes 0**
- **Standard deviation becomes 1**

Formula: `scaled = (value - mean) / std_deviation`

```
Before:  fare = 71.28  →  After:  fare = 1.21
Before:  age  = 38.0   →  After:  age  = 0.86
Before:  pclass = 1    →  After:  pclass = -1.04
```

Now every feature is on equal footing — a step in `fare` is as meaningful as a step in `pclass`.

### In code

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)   # fit = learn mean/std, transform = apply it
```

> Note: Always `fit` on training data only. If you fit on test data too, you're leaking future information into the model. With `cross_val_score`, this is handled correctly internally.

---

## Data Cleaning Notes

### Missing values — 3 strategies used here

| Strategy            | When to use                                                       | Used for                                |
| ------------------- | ----------------------------------------------------------------- | --------------------------------------- |
| **Fill with mean**  | Numeric column, not too many missing, roughly normal distribution | `age` (177 missing → filled with ~29.7) |
| **Drop the rows**   | Very few rows missing (losing them is negligible)                 | `embarked` (only 2 rows → dropped)      |
| **Drop the column** | So many values missing the column is useless                      | `deck` (688/891 missing → dropped)      |

### Encoding — turning text into numbers

ML models only understand numbers. Text columns must be encoded.

**LabelEncoder** — assigns a number to each category alphabetically:

```
sex:      female=0, male=1
embarked: C=0, Q=1, S=2
alone:    False=0, True=1  (boolean — converted via astype(int))
```

**When LabelEncoder can be a problem:**
LabelEncoder implies order — it makes the model think `S=2` is "more embarked" than `C=0`. For tree-based models this doesn't matter much. For distance-based models (KNN, SVM), it can cause issues. The alternative is **One-Hot Encoding** — but for this dataset with the scaling applied, results are acceptable.

---

## The Models Used

### SVM — Support Vector Machine

**The idea:** Find the line (or plane) that best separates survivors from non-survivors, with the **maximum gap (margin)** between the two groups.

```
                    ✗ ✗
              ✗                          ← Non-survivors
         ✗         |  ←  decision boundary
              ✗    |
─────────────────────────────────────────
                   |
              ✓    |
         ✓         |  ←  Survivors
              ✓ ✓
```

The SVM tries to maximise the distance from the boundary to the nearest points of each class.

**Why SVM works well on Titanic:**

- The data has clear patterns (1st class women survived at very high rates, 3rd class men died at very high rates)
- SVM handles these kinds of structured separations well

**Cross Validation result:**

- Scores: `[83.1%, 82.0%, 81.5%, 80.9%, 86.4%]`
- Mean: **82.8%**

---

### KNN — K-Nearest Neighbours

**The idea:** To classify a new passenger, find the K most similar passengers in the training data and take a majority vote.

```
New Passenger: male, age 30, 3rd class, alone

K=5 nearest neighbours found:
  Passenger A: male, age 28, 3rd class, alone → Died
  Passenger B: male, age 32, 3rd class, alone → Died
  Passenger C: male, age 29, 3rd class, not alone → Died
  Passenger D: male, age 31, 2nd class, alone → Died
  Passenger E: male, age 30, 3rd class, alone → Survived

Vote: 4 Died vs 1 Survived → Prediction: Died
```

**Why KNN needs scaling:**
KNN measures similarity using Euclidean distance. Without scaling, a passenger with `fare=500` would appear "far away" from one with `fare=7` even if everything else is identical. Scaling puts all features on equal footing.

**Cross Validation result:**

- Scores: `[78.7%, 76.4%, 82.6%, 81.5%, 80.2%]`
- Mean: **79.9%**

---

## Evaluation Metrics — Understanding the Numbers

### Accuracy

```
Accuracy = (Correct predictions) / (Total predictions)
         = (True Positives + True Negatives) / Total
```

Easy to understand but can be misleading. If 90% of passengers died, a model that predicts "died" for everyone gets 90% accuracy — but it's useless.

### Confusion Matrix

The confusion matrix tells you WHERE your model is wrong, not just HOW OFTEN.

```
                    Model says: DIED    Model says: SURVIVED
Actually DIED           88  ✓                21  ✗
Actually SURVIVED       20  ✗                49  ✓
```

- **88 TN** (True Negative) — correctly predicted death
- **49 TP** (True Positive) — correctly predicted survival
- **21 FP** (False Positive) — predicted survived, actually died (false alarm)
- **20 FN** (False Negative) — predicted died, actually survived (missed!)

### Precision

"Of everyone I labelled as survived — what % actually did?"

```
Precision = TP / (TP + FP) = 49 / (49 + 21) = 70%
```

High precision means: few false alarms. When the model says "survived", it's usually right.

### Recall (Sensitivity)

"Of all the people who actually survived — what % did I catch?"

```
Recall = TP / (TP + FN) = 49 / (49 + 20) = 71%
```

High recall means: few misses. The model doesn't miss many actual survivors.

### F1 Score

Balances Precision and Recall into one number. Useful when both matter.

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
   = 2 × (0.70 × 0.71) / (0.70 + 0.71)
   = 0.71
```

### Precision vs Recall trade-off

```
╔══════════════════════════════════════════════════════════════════════╗
║  SCENARIO            CARE MORE ABOUT    WHY                         ║
╠══════════════════════════════════════════════════════════════════════╣
║  Cancer diagnosis    Recall             Don't miss any real cases    ║
║                                         (false negative = deadly)   ║
║  Spam filter         Precision          Don't block real emails      ║
║                                         (false positive = annoying) ║
║  Fraud detection     Recall             Don't miss fraud            ║
║  General purpose     F1                 Balanced                    ║
╚══════════════════════════════════════════════════════════════════════╝
```

---

## Things to Try Next (Model Tuning Extensions)

### 1. GridSearchCV — find the best hyperparameters

```python
from sklearn.model_selection import GridSearchCV

param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_scaled, y)
print(grid_search.best_params_)   # {'C': 1, 'kernel': 'rbf'} for example
print(grid_search.best_score_)    # best CV accuracy found
```

GridSearchCV trains the model for every combination of parameters you specify and tells you which combo works best.

### 2. Try different values of K in KNN

```python
for k in range(1, 20):
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_scaled, y, cv=5, scoring='accuracy')
    print(f"K={k}: {scores.mean():.4f}")
```

### 3. Random Forest instead of a single Decision Tree

A Random Forest builds 100 decision trees and takes a majority vote — almost always better than one tree.

```python
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100)
scores = cross_val_score(model, X_scaled, y, cv=5)
print(scores.mean())
```

---

## Concept Summary

| Concept                     | In One Line                                                        |
| --------------------------- | ------------------------------------------------------------------ |
| **K-Fold Cross Validation** | Test the model K times on K different slices; average the scores   |
| **cv=5**                    | 5 folds — the most common choice                                   |
| **cross_val_score**         | sklearn function that runs the whole K-fold loop for you           |
| **StandardScaler**          | Rescales each feature to mean=0, std=1 — essential for KNN and SVM |
| **SVM**                     | Draws a boundary that maximises the gap between classes            |
| **KNN**                     | Classifies by majority vote of K nearest neighbours                |
| **Accuracy**                | % of correct predictions — simple but can mislead                  |
| **Confusion Matrix**        | Breaks down correct/wrong predictions into 4 buckets               |
| **Precision**               | Of predicted positives, how many were actually positive?           |
| **Recall**                  | Of actual positives, how many did we catch?                        |
| **F1 Score**                | Balanced score between precision and recall                        |

---

_Part of the Algorithms/Unsupervised/Model_Tuning series._
