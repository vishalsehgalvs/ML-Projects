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

Covered here: K-Means, DBSCAN

---

### Dimensionality Reduction — shrinking without losing what matters

Some datasets have hundreds or thousands of columns. A lot of those columns are saying similar things — they overlap. Dimensionality reduction compresses them into fewer columns that still carry most of the original information.

Practical example: a medical dataset has 800 test results per patient. Many of those tests are correlated. You can compress them down to 40 "summary" columns that capture 95% of the same information. Smaller, faster, often just as accurate.

Covered here: PCA, t-SNE

---

## When Would You Actually Use This?

- You got a new dataset and don't know what to predict yet — explore it first with clustering
- Your dataset has too many columns and training is slow — compress it with PCA first
- You want to catch unusual things in live data — fraud, defects, network attacks
- You want to segment your users or customers without manually tagging thousands of records

---

## What's in This Folder

| Folder                                                     | What it covers                                       | Status                            |
| ---------------------------------------------------------- | ---------------------------------------------------- | --------------------------------- |
| `Clustering_Algorithm/K_Mean_Clustering/`                  | K-Means — centroid-based clustering, elbow method    | Done                              |
| `Clustering_Algorithm/DBSCAN_Algorithm/`                   | DBSCAN — density-based clustering, handles any shape | Done                              |
| `Clustering_Algorithm/Dimensionality_Reduction_Algorithm/` | PCA, t-SNE — compressing high-dimensional data       | Notes & README done — code coming |

> Model Tuning stuff (cross-validation, grid search, ensemble methods) is in `Algorithms/Model_Tuning/` — that applies to any kind of model so it lives one level up, not here.

---

## K-Means vs DBSCAN — Side by Side

Both plots below use the exact same crescent moon dataset. The only thing that changes is the algorithm used to group the data.

|                                                                  K-Means — fails on this shape                                                                   |                                     DBSCAN — handles it correctly                                      |
| :--------------------------------------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------: |
| ![K-Means Failure](Clustering_Algorithm/K_Mean_Clustering/k_mean_clustering_images/failure%20of%20k%20mean%20clustering%20when%20data%20is%20not%20circulat.png) |         ![DBSCAN Clusters](Clustering_Algorithm/DBSCAN_Algorithm/images/dbscan%20clusters.png)         |
|                                  Draws a straight line between two centroids. Points from both crescents end up mixed together.                                  | Follows the density trail. Each crescent becomes its own cluster. Isolated points are marked as noise. |

**Why does K-Means fail here?**
K-Means assigns every point to its nearest centroid and draws straight-line boundaries between them. For crescent-shaped data, both centroids end up floating in the gap between the curves — not actually inside either crescent. So it slices the data down the middle and misclassifies half the points.

**Why does DBSCAN handle it?**
DBSCAN has no centroids. It just looks at each point and asks: are there enough nearby points to call this a dense region? It then follows the density from point to point — and that trail naturally follows the curve of the crescent, picking up the whole shape correctly.

This is the single clearest example of why choosing the right algorithm for your data shape matters.

---

## Which Algorithm Should I Use?

A quick cheat sheet for picking the right one:

| Situation                                                                                       | Go with                                                                                              |
| ----------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| Your data forms round blob-shaped groups and you have a rough idea of how many groups           | K-Means                                                                                              |
| Your data has irregular shapes — curves, rings, crescents — or you don't know how many clusters | DBSCAN                                                                                               |
| You have too many features and want to reduce them before training a model                      | PCA                                                                                                  |
| You just want to visualise the structure of your data in 2D                                     | t-SNE                                                                                                |
| You want to find outliers or unusual records                                                    | DBSCAN — noise points (label = -1) are the outliers                                                  |
| You're not sure — exploring the data for the first time                                         | Start with K-Means to see if any obvious groups emerge, then try DBSCAN if the shapes look irregular |

---

_Part of the Algorithms/ series._
