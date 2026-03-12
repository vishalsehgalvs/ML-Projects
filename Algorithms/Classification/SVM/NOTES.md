# Notes — Support Vector Machine (SVM)

Same dataset, same cleaning. This one felt the most abstract to understand at first, but once it clicked it actually makes a lot of sense.

---

## The core idea

Every other algorithm we've done tries to find a pattern by looking at the whole dataset at once — drawing a line, calculating probabilities, building a tree. SVM does something different.

SVM looks for a **boundary line** (or in higher dimensions, a boundary surface) that separates the two classes — survived vs died. But it doesn't just find _any_ boundary. It finds the one with the **biggest gap on either side**.

Imagine drawing a line between two groups of coloured dots on a piece of paper. You could draw it a hundred different ways and still separate them. SVM says: draw the line so that the dots on either side are as far away from the line as possible. The wider that gap, the more confident the model is when it sees a new point.

The dots sitting right at the edge of that gap — the ones closest to the boundary — are called **support vectors**. They're the ones that actually define where the boundary sits. Move them and the boundary moves. Move any other dot and nothing changes. That's where the name comes from.

---

## The margin

The distance between the boundary line and the nearest point on each side is called the **margin**. SVM maximises this margin.

Why does that matter? A wider margin means the model has more breathing room. A new passenger who sits somewhere in between the two groups is more likely to land on the correct side if the boundary was drawn conservatively with space on both sides, rather than squeezed tightly against the training data.

Think of it like navigating a narrow mountain road vs a wide highway. You're less likely to go off the edge on the highway even if you wander a bit.

---

## What is the RBF kernel and why does it matter

The data isn't always neatly separable with a straight line. Survived and died passengers are mixed in ways that no single line can cleanly separate.

The **kernel trick** is SVM's way of handling this. Instead of trying to draw a line in the original space, it mathematically transforms the data into a higher-dimensional space where a clean boundary _does_ exist, finds the boundary there, then maps it back.

You never actually see this transformation — sklearn handles it internally. You just pick the kernel type.

We used `kernel="rbf"` — **Radial Basis Function**. This is the most commonly used kernel for non-linear data. It effectively draws curved, circular boundaries around clusters of points rather than straight lines.

Think of it like wrapping the survived group in a bubble that curves around their cluster rather than cutting across the data with a ruler.

Other kernel options:

- `linear` — straight line boundary. Good when the classes are roughly linearly separable.
- `poly` — polynomial curve. More flexible than linear, less flexible than rbf.
- `rbf` — curves in all directions. Default and usually a good starting point.

---

## Why scaling is mandatory for SVM

SVM measures distances to find the margin. It's the same problem as KNN — if fare goes up to 500 and pclass only goes 1–3, the fare column completely dominates the distance calculation. The boundary ends up being almost entirely about fare.

Scaling brings everything to the same range, so all features contribute equally to where the boundary sits. Both training and test data are scaled before fitting.

---

## Results

```
Accuracy: 82.6%

Confusion Matrix:
[[96  13]
 [18  51]]
```

Out of 178 test passengers:

- 96 — said died, actually died. correct.
- 51 — said survived, actually survived. correct.
- 13 — said survived, actually died. false alarm.
- 18 — said died, actually survived. missed.

```
Classification Report:
              precision    recall    f1
   Died (0)     0.84       0.88     0.86
Survived (1)   0.80       0.74     0.77
```

---

## All five algorithms compared

|                 | Logistic Regression | KNN K=5 | Naive Bayes | Decision Tree | SVM RBF   |
| --------------- | ------------------- | ------- | ----------- | ------------- | --------- |
| Accuracy        | 80.3%               | 78.1%   | 77.5%       | 77.0%         | **82.6%** |
| Died recall     | 0.83                | 0.83    | 0.77        | 0.81          | **0.88**  |
| Survived recall | 0.77                | 0.71    | 0.78        | 0.71          | 0.74      |

SVM wins on accuracy and on died recall. It missed fewer of the actual deaths (only 13 false alarms, vs 16–25 for the others). The tradeoff is that survivor recall dropped slightly — it missed 18 actual survivors vs Logistic Regression's 16.

The big jump from Logistic Regression (80.3%) to SVM (82.6%) is meaningful for such a small dataset. The RBF kernel is picking up non-linear patterns in the data that a straight-line classifier can't capture.

---

## Why SVM tends to perform well on small datasets

The support vectors — the handful of points right at the edge of the margin — are doing all the work. The rest of the training data doesn't even affect the boundary. This makes SVM surprisingly effective when you don't have thousands of rows, because it focuses on the most informative examples rather than averaging across everything.

It also doesn't overfit as easily as a Decision Tree, because the margin constraint acts as a natural brake on how tightly the boundary hugs the training data.

---

## Where SVM struggles

- **Slow on large datasets** — finding the optimal boundary involves solving a quadratic optimisation problem. That's fine for 700 rows but painful at millions.
- **Hard to interpret** — you can't look at an SVM model and explain in plain words why it made a particular prediction. The kernel transformation makes it a black box.
- **Sensitive to kernel and parameter choice** — RBF with default settings worked well here, but in general you'd want to tune the `C` parameter (controls how strict the margin is) and `gamma` (controls how far the influence of a single point reaches). We didn't tune either here.

---

## Things to remember

- SVM finds the boundary with the biggest gap between classes — the margin.
- The support vectors (points closest to the boundary) are the only ones that define where it sits.
- RBF kernel handles non-linear data by curving the boundary. Usually the safest default.
- Scaling is required — SVM is distance-based just like KNN.
- Best accuracy of all five algorithms on this dataset (82.6%).
- Not great for very large datasets or situations where you need to explain the prediction.

---

_Part of the Algorithms/Classification series._
