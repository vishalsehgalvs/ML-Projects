# Notes — KNN (K-Nearest Neighbours) on the Titanic Dataset

---

## What we're doing here in one sentence

We're teaching a computer to guess whether a Titanic passenger survived or died — using only information like their age, gender, ticket class, and how many family members were on board.

---

## The dataset

The Titanic dataset is built into the seaborn library — no CSV file needed, just `sns.load_dataset('titanic')` and it loads right in.

There were 891 passengers in the dataset and 15 columns to start with. Not all of them were useful.

---

## Step 1 — Getting rid of the rubbish columns

Before training anything, we dropped 6 columns:

| Column        | Reason for dropping                                                                                                                            |
| ------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| `deck`        | 688 out of 891 values were blank. Filling them with a made-up average would just be lying to the model.                                        |
| `alive`       | This is literally just the `survived` column written as "yes" or "no". Keeping it would be cheating — the model would already know the answer. |
| `class`       | Same thing as `pclass`, just written differently. Duplicate.                                                                                   |
| `embark_town` | Same thing as `embarked`. Duplicate.                                                                                                           |
| `who`         | Already covered by `sex` and `age` combined. Adds nothing new.                                                                                 |
| `adult_male`  | Same — already captured by `sex` and `age`. Redundant.                                                                                         |

A good rule of thumb: if two columns are saying the exact same thing, keep one and throw the other away. And if a column is more than 70% empty, it's usually more noise than signal.

---

## Step 2 — Fixing the missing values

After dropping those columns, we had two remaining issues:

**Age — 177 missing values**
We filled the blanks with the mean age (around 29.7). This isn't perfect but it's the standard move when you have too many blanks to throw away the rows but don't want to make up wild numbers.

**Embarked — 2 missing values**
Only 2 rows, so we just dropped them entirely. Going from 891 to 889 rows is not a meaningful loss.

---

## Step 3 — Turning words into numbers

Machine learning models can't read words. They only understand numbers. So we converted:

- `sex` — female → 0, male → 1
- `embarked` — Cherbourg → 0, Queenstown → 1, Southampton → 2

This is done using `LabelEncoder` from sklearn. It just scans the unique values and assigns them numbers in alphabetical order.

Also, the `alone` column was stored as `True/False` (boolean). We converted the entire dataframe to `int` so everything is consistently numeric, with `True → 1` and `False → 0`.

---

## Baseline — Logistic Regression first

Before even touching KNN, we ran Logistic Regression on the same data. This gives us a baseline — a score to beat, or at least compare against.

**Logistic Regression results (no scaling needed):**

```
Accuracy: 80.3%

Confusion Matrix:
[[90  19]
 [16  53]]

Classification Report:
              precision    recall    f1
   Died (0)     0.85       0.83     0.84
Survived (1)   0.74       0.77     0.75
```

So Logistic Regression got 80.3% right on the test set. That's our bar.

---

## How KNN actually works — the plain English version

Imagine you're at a party and you know nothing about an incoming guest. But you can see which existing guests they're most similar to — same age range, same background, same interests. You look at the 5 people most like them, check what they do for work, and say "most of the similar people are teachers, so this new person is probably a teacher too."

That's KNN.

1. Take a new passenger we haven't seen before
2. Measure how similar (or rather how "close" in number-space) they are to every row in the training data
3. Pick the 5 nearest ones (because we set K=5)
4. Check how many of those 5 survived vs died
5. Whatever the majority says — that's the prediction

There's no learning happening in the traditional sense. The model doesn't build equations or find patterns. It literally just stores all the training data and does the comparison at prediction time.

---

## How distance is measured

By default, KNN uses Euclidean distance — the straight-line distance between two points. In school you'd call it the Pythagorean theorem:

$$d = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2}$$

With 8 columns, it's the same formula but with 8 terms under the square root instead of 2. One term per column.

There's also Manhattan distance — instead of a straight line, you walk along the grid:

$$d = |x_2 - x_1| + |y_2 - y_1|$$

Think of driving in a city. You can't cut diagonally through buildings — you go along streets. That's Manhattan distance.

For most standard problems, the default (Euclidean) works fine.

---

## The most important step for KNN — Feature Scaling

This is where most beginners get burned.

Look at the data after cleaning:

- `age` → values like 22, 38, 26 (range roughly 1–80)
- `fare` → values like 7, 71, 53 (range roughly 0–500)
- `pclass` → values like 1, 2, 3 (range 1–3)

When KNN calculates distance, a passenger with age=22 and fare=7 vs another with age=23 and fare=71 — the fare difference of 64 completely dominates. The age difference of 1 barely registers. The model essentially ignores age and pclass and weights everything by fare.

Scaling fixes this. We used `StandardScaler`, which transforms each column so it has:

- Mean = 0
- Standard deviation = 1

So instead of fare being 7 or 500, it becomes -0.48 or 2.4. Instead of age being 22 or 80, it becomes -0.61 or 3.1. Everything is on the same playing field.

**Important:** Fit the scaler on training data, transform both train and test. Don't leak test data into the scaler fitting.

---

## Why K=5 was used

K=5 is the most common default starting point. Here's the intuition:

- **K=1** — Only looks at the single closest neighbour. One weird outlier in your data can completely throw off a prediction. Too sensitive.
- **K=3** — Bit more stable but still can be noisy.
- **K=5** — Good balance. Stable enough to ignore individual outliers, small enough to still be local.
- **K very large** — Starts to just predict whatever the majority class is in the whole dataset. Loses its "local neighbourhood" advantage.

Always use odd numbers for binary classification to avoid ties. No one wins with a 2-2 split.

---

## KNN Results

```
Accuracy: 78.1%

Confusion Matrix:
[[90  19]
 [20  49]]

Classification Report:
              precision    recall    f1
   Died (0)     0.82       0.83     0.82
Survived (1)   0.72       0.71     0.72
```

---

## Reading the Confusion Matrix

The confusion matrix has 4 cells:

```
                  Predicted: Died    Predicted: Survived
Actual: Died           90                  19
Actual: Survived       20                  49
```

- **90** — We said died, they actually died. Correct. (True Negative)
- **49** — We said survived, they actually survived. Correct. (True Positive)
- **19** — We said survived, they actually died. False alarm. (False Positive)
- **20** — We said died, they actually survived. Missed them. (False Negative)

In real life, the cost of these two wrong types is different. Wrongly predicting someone survived when they died is a very different kind of mistake from predicting they died when they actually made it. Depending on the use case, you'd optimise for one over the other.

For this exercise, we're just tracking overall accuracy.

---

## KNN vs Logistic Regression — Head to Head

| Metric          | Logistic Regression | KNN (K=5) |
| --------------- | ------------------- | --------- |
| Accuracy        | **80.3%**           | 78.1%     |
| Died Precision  | 0.85                | 0.82      |
| Survived Recall | 0.77                | 0.71      |
| Died Recall     | 0.83                | 0.83      |

Logistic Regression edges out KNN here by about 2%. This is actually pretty common on tabular datasets — KNN shines more when the data has a natural geometric structure. On messy real-world data like this, simple linear models often hold their own.

KNN is not worse — just a different tool. For visual data, clusters, or patterns that aren't linear, KNN can outperform Logistic Regression quite easily.

---

## What's good and bad about KNN

**Good:**

- The simplest algorithm to explain to anyone — "find the most similar ones and take a vote"
- No training step — just stores the data (fast to set up)
- Works for multi-class problems (3+ categories) without any modification
- Makes zero assumptions about the shape of the data — non-linear patterns are fine

**Bad:**

- Slow at prediction time — for every new point, it has to measure distance to every training row
- Memory hungry — keeps the entire training dataset around forever
- Falls apart with lots of columns — the "curse of dimensionality" means everything starts looking equally far from everything else
- Useless if you forget to scale — the results will be garbage

---

## Key takeaways from this implementation

1. Always drop duplicate columns first — having two versions of the same feature doesn't help
2. Fill or drop missing values before anything else
3. LabelEncoder turns text categories into numbers but assumes no ordering between them
4. For KNN specifically, StandardScaler is not optional — it's mandatory
5. K=5 is a safe default for binary classification
6. Logistic Regression doesn't need scaling the same way — linear models care about coefficients, not distances
7. Always have a baseline before claiming a model is good or bad

---

_Part of the Algorithms/Classification series._
