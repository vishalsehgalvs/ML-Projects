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

This is now done in `Grid_Search_CV.py` — see the section below.

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

# Grid Search CV — Notes

> File: `Grid_Search_CV.py` | Dataset: Iris

---

## What are Hyperparameters?

Hyperparameters are settings you choose **before** training starts. The model doesn't learn them from data — you set them manually.

Examples:

- In KNN: how many neighbours to check (`n_neighbors`)
- In SVM: how strict the penalty for mistakes is (`C`), and the shape of the boundary (`kernel`)

The problem is that different values give different accuracy scores. And there's no formula to tell you the best value upfront — you just have to try.

---

## Why Manual Tuning is a Bad Idea

In `Grid_Search_CV.py`, the first thing done is manually changing values and checking the score:

```python
model_KNN = KNeighborsClassifier(n_neighbors=5)
# score = 0.98

model_KNN = KNeighborsClassifier(n_neighbors=13)
# score = 1.0  ← looks amazing but is suspicious
```

A score of 1.0 (100%) on a test set is almost always **overfitting** — the model memorised the test examples rather than learning the real pattern. Change the test set and the score drops.

The deeper issue: you're choosing settings by peeking at the test score. That's not honest evaluation — the test set is supposed to be completely untouched until the very end.

**Manual tuning problems:**

- You don't know which combinations to even try
- There are too many to check by hand
- You end up unintentionally overfitting to your test set

---

## How GridSearchCV Works

You give it:

1. A model
2. A dictionary of hyperparameter options to try
3. How many folds for cross-validation (`cv`)

It tries every combination, runs K-Fold CV on each one, and returns a full results table.

```python
from sklearn.model_selection import GridSearchCV

classifier = GridSearchCV(model_SVM, param_grid={
    'C': [1, 10, 20, 30],
    'kernel': ['rbf', 'linear']
}, cv=5, return_train_score=False)

classifier.fit(X, y)   # pass the full dataset — GridSearchCV handles the splits
```

Notice: you pass the full `X` and `y` here, not a train/test split. GridSearchCV does all the splitting internally through cross-validation.

---

## Reading the Results Table

`classifier.cv_results_` returns a messy dictionary. Converting it to a DataFrame makes it readable:

```python
result_df = pd.DataFrame(classifier.cv_results_)
pd.set_option('display.max_columns', None)   # show all columns
pd.set_option('display.width', None)         # avoid wrapping
print(result_df)
```

The columns that actually matter:

| Column            | What it tells you                      |
| ----------------- | -------------------------------------- |
| `param_C`         | Which value of C was used for this row |
| `param_kernel`    | Which kernel was used                  |
| `mean_test_score` | Average accuracy across all 5 folds    |
| `rank_test_score` | 1 = best combo, higher = worse         |

To get a clean summary:

```python
clean = result_df[['param_C', 'param_kernel', 'mean_test_score', 'rank_test_score']]
print(clean.sort_values('rank_test_score'))
```

---

## SVM Grid Search — What We Found

Searched: `C` in [1, 10, 20, 30] and `kernel` in ['rbf', 'linear'] — 8 combinations total.

```
C=1,  rbf    → 98.0%   ← rank 1
C=1,  linear → 98.0%   ← rank 1
C=10, rbf    → 98.0%   ← rank 1
C=10, linear → 97.3%
C=20, rbf    → 96.7%
C=20, linear → 96.7%
C=30, rbf    → 96.0%
C=30, linear → 96.0%
```

**Why does lower C win?**

`C` controls how much the model punishes itself for misclassifying a training point.

- **High C** = very strict = tries to get every training point right = boundary becomes jagged and specific = overfitting
- **Low C** = more relaxed = allows a few training mistakes = boundary stays smooth and general = better on unseen data

For the Iris dataset (which is pretty clean and well-separated), `C=1` is relaxed enough to get a smooth boundary that generalises well.

---

## KNN Grid Search — What We Found

Searched 28 combinations (`n_neighbors` × `weights` × `metric`).

Best result:

```
n_neighbors=11, weights='distance', metric='minkowski'  → 98.67%   rank 1
n_neighbors=11, weights='distance', metric='euclidean'  → 98.67%   rank 1
```

**Why `weights='distance'`?**

With `weights='uniform'`, every neighbour gets an equal vote regardless of how close they are. A neighbour 0.01 away counts the same as one 5.0 away.

With `weights='distance'`, closer neighbours count more. A flower that's almost identical to the new one has more say than a flower that's only vaguely similar.

For the Iris dataset, distance weighting helped because the three species form fairly tight clusters — nearby neighbours are very reliable signals.

**Why not n_neighbors=13 which scored 1.0 earlier?**

That 1.0 was tested on a fixed 33% test split without cross-validation — that split happened to be easy. GridSearchCV's 5-fold CV tested on multiple different slices and gave a more honest score of 97.3%. The slightly lower, more honest number is trustworthy. The 1.0 was not.

---

## The Iris Dataset

Iris is a classic benchmark dataset with 150 rows of flower measurements. Unlike Titanic, it needs no cleaning — no missing values, no text columns, no duplicate rows.

```
   sepal_length  sepal_width  petal_length  petal_width species
0           5.1          3.5           1.4          0.2  setosa
1           4.9          3.0           1.4          0.2  setosa
...
```

Three classes: `setosa`, `versicolor`, `virginica`. This is a **multi-class** problem (3 categories), unlike Titanic which was binary (survived: yes or no). Both SVM and KNN handle multi-class natively.

---

## Concept Summary

| Concept              | In One Line                                                                  |
| -------------------- | ---------------------------------------------------------------------------- |
| **Hyperparameter**   | A model setting you choose before training — not learned from data           |
| **GridSearchCV**     | Tries every combo of hyperparameters using CV and returns the best one       |
| **param_grid**       | The dictionary of options you want to test                                   |
| **cv=5**             | Each combo is tested with 5-fold cross validation                            |
| **mean_test_score**  | The average accuracy across all 5 folds for that combo                       |
| **rank_test_score**  | 1 = best; use this to sort and find the winner                               |
| **C (in SVM)**       | Strictness — lower C = smoother boundary = less overfitting                  |
| **weights=distance** | Closer neighbours vote more — useful when clusters are tight                 |
| **Overfitting sign** | Score of 1.0 on a simple test split with n_neighbors=13 — doesn't hold in CV |

---

# Randomized Search CV — Notes

> File: `random_search.py` | Dataset: Iris

---

## The Problem with Grid Search at Scale

Grid Search is great but it always tries every combination. If your param grid has:

- 10 values of C
- 5 kernels
- 3 gamma values

That's 10 × 5 × 3 = **150 combinations**, each run 5 times with cross-validation = **750 model fits**.

For small grids, fine. For large grids, it can take a very long time.

Randomized Search solves this with one extra parameter: `n_iter`.

---

## How RandomizedSearchCV Works

```python
from sklearn.model_selection import RandomizedSearchCV

classifier = RandomizedSearchCV(model_SVM, {
    'C': [1, 10, 20, 30],
    'kernel': ['rbf', 'linear']
}, cv=5, return_train_score=False, n_iter=4)

classifier.fit(X, y)
```

`n_iter=4` means: instead of testing all 8 combinations, randomly pick 4.

Everything else — splitting, fitting, scoring, results format — is identical to GridSearchCV.

---

## Why Random Sampling Works

You might ask: "what if the randomly chosen combos miss the best one?"

In practice this rarely matters because:

1. The best combo usually isn't dramatically better than the second or third best
2. With a reasonable `n_iter` (20-50% of total combos), you almost always get close to the top
3. The time saved by skipping the rest usually outweighs the tiny accuracy difference

In this experiment: Grid Search tried all 8 SVM combos and found 98.0%. Randomized Search tried 4 and also found 98.0%. Same result, half the work.

---

## SVM — What We Found

`n_iter=4` — 4 random combos picked from 8.

```
C=30, linear  → 96.0%   rank 4
C=20, linear  → 96.7%   rank 3
C=20, rbf     → 98.0%   rank 1
C=1,  linear  → 98.0%   rank 1
```

Best found: C=20 rbf or C=1 linear at 98.0%. Note: the exact picks change each run because they're random. The winner might be slightly different on your machine.

---

## KNN — What We Found

`n_iter=5` — 5 random combos picked from 28.

```
n_neighbors=13, distance, minkowski  → 98.0%   rank 2
n_neighbors=3,  distance, euclidean  → 96.7%   rank 5
n_neighbors=11, distance, euclidean  → 98.67%  rank 1
n_neighbors=9,  distance, euclidean  → 97.3%   rank 3
n_neighbors=13, uniform,  minkowski  → 97.3%   rank 3
```

Best: `n_neighbors=11, weights=distance, metric=euclidean` at 98.67%. Same as the full Grid Search result.

---

## When to Use Which

| Situation                             | Use                                                         |
| ------------------------------------- | ----------------------------------------------------------- |
| Fewer than ~20 combinations           | GridSearchCV                                                |
| Hundreds of combinations              | RandomizedSearchCV                                          |
| You want guaranteed exhaustive search | GridSearchCV                                                |
| Quick exploration of a large space    | RandomizedSearchCV                                          |
| First pass, then narrow down          | RandomizedSearchCV first, then Grid Search on the shortlist |

---

## Concept Summary

| Concept                | In One Line                                                                  |
| ---------------------- | ---------------------------------------------------------------------------- |
| **RandomizedSearchCV** | Same as GridSearchCV but only tests n_iter randomly chosen combos            |
| **n_iter**             | How many random combinations to try — lower = faster, higher = more thorough |
| **Trade-off**          | Speed vs exhaustiveness — usually worth it for large grids                   |
| **Results format**     | Same as GridSearchCV — `cv_results_`, `mean_test_score`, `rank_test_score`   |
| **Unpredictability**   | Picks are random so results may vary slightly between runs                   |

---

---

# Ensemble Methods — Notes

> Files: `ensemble_learning.py`, `ensemble_methods.py` | Dataset: Iris

---

## The Core Idea

One model is like one person's opinion. It might be right most of the time, but it has blind spots. A group of models is like a panel of experts — their individual errors tend to cancel each other out, and the consensus is more reliable.

The three ways to build an ensemble:

1. **Stacking** — different models, a second model combines their answers
2. **Bagging** — same model type, many copies, each trained on a random data slice, majority vote
3. **Boosting** — same model type, trained in sequence, each one correcting the last one's mistakes

---

## Stacking in Depth

### How it works

```
Step 1: Train base learners on training data
        DecisionTree  → predictions on training folds
        SVM           → predictions on training folds
        LogisticReg   → predictions on training folds

Step 2: Use those predictions as INPUT to the meta-learner
        Meta-learner learns: when Tree says X and SVM says Y, the answer is usually Z

Step 3: At test time:
        Base learners predict → meta-learner makes final call
```

### Why cv=5 in StackingClassifier?

The base learners can't train and then predict on the same data — they'd just memorise it, making the meta-learner's job meaningless. So sklearn uses cross-validation internally: train each base learner on 4 folds, predict on the 5th, rotate, and the meta-learner gets trained on predictions the base models never saw.

### In code

```python
base_learners = [
    ('dt', DecisionTreeClassifier(random_state=42)),
    ('svc', SVC(probability=True, kernel='rbf', random_state=42)),
    ('lr', LogisticRegression(max_iter=1000))
]
meta_learner = LogisticRegression(max_iter=1000)

stacking = StackingClassifier(estimators=base_learners, final_estimator=meta_learner, cv=5)
stacking.fit(X_train, y_train)
```

`probability=True` on SVC makes it output a probability (e.g. 0.87 for setosa) rather than just a label. The meta-learner can use those probabilities as richer input.

---

## Bagging — Random Forest

### The idea

Random Forest builds 100 (or however many you set with `n_estimators`) decision trees. Each one gets:
- A random sample of rows (with replacement — some rows may appear twice, some not at all)
- A random subset of features to consider at each split

Then it takes a majority vote.

### Why this helps

A single decision tree is prone to overfitting — it memorises the training data and then does badly on new data. But when you average 100 slightly different trees, the overfitting in each tree cancels out. The average is much more stable.

### In code

```python
rf = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)
rf.fit(X_train, y_train)
```

`max_depth=None` means trees grow until they're pure — fine for Random Forest because the ensemble effect contains the overfitting.

---

## Boosting — AdaBoost, Gradient Boosting, XGBoost

### AdaBoost

Stands for Adaptive Boosting.

- Start with equal weight on all rows
- Train model, check which rows it got wrong
- Increase the weight on wrong rows, decrease weight on correct rows
- Next model sees the reweighted data and focuses more on the hard cases
- Repeat 100 times, final answer = weighted vote of all rounds

Result on Iris: 93.3% — decent but not the best.

### Gradient Boosting

- Same sequence idea, but uses a smarter mechanism: each new tree tries to predict the **residual error** (the gap between what the model predicted and the actual answer)
- `learning_rate=0.1` means each new tree contributes 10% of its correction — small steps, more careful improvement
- After 100 rounds of 10% corrections, the cumulative error shrinks significantly

Result on Iris: 100%.

### XGBoost

Extreme Gradient Boosting — same concept as Gradient Boosting but:
- Handles missing values automatically
- Built-in regularisation to prevent overfitting
- Runs much faster (parallel processing)
- The industry standard for tabular data

```python
xgb = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3,
                    use_label_encoder=False, eval_metric='mlogloss')
```

**Important:** XGBoost requires numeric labels. If your target column has strings (like 'setosa'), use `LabelEncoder` to convert them to integers (0, 1, 2) before passing to XGBoost. Sklearn models accept strings, XGBoost does not.

Result on Iris: 100%.

---

## Model Comparison on Iris

| Method             | Type     | Result  | Notes                                          |
| ------------------ | -------- | ------- | ---------------------------------------------- |
| Stacking           | Stacking | —      | Most flexible, combines different model types  |
| Random Forest      | Bagging  | —      | Stable, rarely overfits, great default choice  |
| AdaBoost           | Boosting | 93.3%   | Older, simpler boosting                        |
| Gradient Boosting  | Boosting | 100%    | Strong, slower to train                        |
| XGBoost            | Boosting | 100%    | Fastest, most widely used in practice          |

---

## When to Use Which

| Situation | Recommended |
| --- | --- |
| Quick strong baseline | Random Forest |
| Best possible accuracy on tabular data | XGBoost |
| You have very different model types and want to combine them | Stacking |
| Data is noisy, variance is the problem | Bagging (Random Forest) |
| Bias is the problem (model too simple) | Boosting |

---

## Concept Summary

| Concept | In One Line |
| --- | --- |
| **Ensemble** | Combine multiple models to get better results than any one alone |
| **Bagging** | Same model type, parallel, random subsets, majority vote |
| **Boosting** | Same model type, sequential, each corrects the last one's errors |
| **Stacking** | Different model types, a meta-model combines their outputs |
| **Random Forest** | 100+ decision trees via bagging — the bagging gold standard |
| **AdaBoost** | Original boosting — reweights wrong examples |
| **Gradient Boosting** | Corrects residual errors at each step |
| **XGBoost** | Gradient boosting, faster, more powerful, industry standard |
| **Meta-learner** | The second-level model in stacking that learns from base model outputs |
| **probability=True** | Makes SVC output probabilities instead of hard labels — useful for stacking |

---

_Part of the Algorithms/Model_Tuning series._
