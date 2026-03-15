# Unsupervised Learning

So far everything we've done had labels. We gave the model data and told it the right answer — this passenger survived, this person has heart disease, this insurance charge is X. The model learned by checking its guesses against those answers.

Unsupervised learning is different. There are no labels. No right answers. You just give the model a bunch of data and say "find whatever's interesting in here." It has to figure out patterns, groups, and structure completely on its own.

A simple way to think about it: imagine someone hands you 500 random photos of animals and asks you to sort them into groups, but doesn't tell you what the groups should be. You'd probably end up grouping cats together, dogs together, birds together — not because you were told to, but because those things just look similar. That's essentially what unsupervised learning does with data.

---

## What's the Goal?

There's no single goal — it depends on what you're trying to do:

- **Find groups in your data** — maybe your customers naturally fall into a few types and you want to know what they are
- **Shrink the data down** — if you have 1000 columns, maybe 50 of them capture most of what matters. Less data, same insight, faster training
- **Spot the weird ones** — find transactions, readings, or records that don't look like anything else. Useful for fraud, defects, anything that shouldn't be there
- **Understand your data before building a model** — sometimes you don't even know what you want to predict yet. Unsupervised learning helps you explore first

---

## How is it Different from What We've Done Before?

|                            | What we've done (supervised)         | Unsupervised                              |
| -------------------------- | ------------------------------------ | ----------------------------------------- |
| Does the data have labels? | Yes — every row has a correct answer | No — just raw data, no answers            |
| What does the model learn? | How to predict that label            | What patterns or groups naturally exist   |
| What do you need?          | A labelled dataset                   | Just the raw data                         |
| What comes out?            | A prediction                         | Groups, a compressed version, or outliers |
| Examples                   | SVM, KNN, Linear Regression          | K-Means, PCA                              |

---

## The Main Techniques

### Clustering — finding groups

The model looks at all the data and puts similar things together into groups (called clusters), with no instruction on what the groups should be.

Practical example: a company has 50,000 customers. No one's manually tagged them. Run clustering and you might get back 4 groups — people who buy once and never return, people who browse a lot but rarely buy, loyal weekly shoppers, and bulk buyers. The model found those groups from the data alone.

Will cover: K-Means, DBSCAN, Hierarchical Clustering

---

### Dimensionality Reduction — shrinking without losing what matters

Some datasets have hundreds or thousands of columns. A lot of those columns are saying similar things — they overlap. Dimensionality reduction compresses them into fewer columns that still carry most of the original information.

Practical example: a medical dataset has 800 test results per patient. Many of those tests are correlated. You can compress them down to 40 "summary" columns that capture 95% of the same information. Smaller, faster, often just as accurate.

Will cover: PCA, t-SNE

---

### Anomaly Detection — spotting what doesn't belong

Train on normal data, then flag anything that looks unusual.

Practical example: a model trained on millions of normal bank transactions learns what "normal" looks like. When a transaction comes in that doesn't match those patterns — a $5,000 purchase at 3am in a country the account has never been used in — it flags it. No one told it what fraud looks like. It just knows what's out of place.

---

## When Would You Actually Use This?

- You got a new dataset and don't know what to predict yet — explore it first with clustering
- Your dataset has too many columns and training is slow — compress it with PCA first
- You want to catch unusual things in live data — fraud, defects, network attacks
- You want to segment your users or customers without manually tagging thousands of records

---

## What's Coming in This Folder

| Folder                                                     | What it covers                                       | Status |
| ---------------------------------------------------------- | ---------------------------------------------------- | ------ |
| `Clustering_Algorithm/K_Mean_Clustering/`                  | K-Means — centroid-based clustering, elbow method    | Done   |
| `Clustering_Algorithm/DBSCAN_Algorithm/`                   | DBSCAN — density-based clustering, handles any shape | Done   |
| `Clustering_Algorithm/Dimensionality_Reduction_Algorithm/` | PCA, t-SNE — compressing high-dimensional data       | Done   |

> Model Tuning stuff (cross-validation, grid search, ensemble methods) is in `Algorithms/Model_Tuning/` — that applies to any kind of model so it lives one level up, not here.

---

_Part of the Algorithms/ series._

we have to initialise k -vvlaye
outliers will move the centre points
only fir for centroid based data

DBSCAN:
density based clustering of application with noise

-non -parametric algo.
epeselon distance is calculated
on trial basis the algorithm calculates the ditance and everything within the imaginary distance/circle is considered as cluster
everything inside epsielon distance is considered as as closter

remember to compare dbscan cluster with k mean cluster and take the cluster image from images folder please

dimensionality reduction algorithm
