# Dimensionality Reduction — PCA

Imagine you have a spreadsheet with 500 rows and 5 columns. Each column is a piece of information about each row — a feature. That's 5 dimensions.

Now imagine you had 500 columns instead. It gets very hard for any algorithm to find patterns in that much data. Things get slow, noisy, and the model starts to struggle.

Dimensionality reduction is the idea of taking all those columns and squashing them down into fewer columns — while keeping as much of the useful information as possible. You lose a little, but you keep the important structure.

This folder covers **PCA — Principal Component Analysis**, which is the most common way to do this.

---

## The Core Idea

PCA looks at all your features and asks: which direction in this data has the most spread? That direction becomes the first new column — called **PC1** (Principal Component 1).

Then it asks: what's the next direction with the most spread, that doesn't overlap with the first? That becomes **PC2**.

So instead of your original 5 features, you now have 2 new ones — PC1 and PC2 — that together capture most of what was going on in the original 5.

**Why does this work?** Because in real datasets, many features are correlated — they move together. PCA finds those shared patterns and summarises them into fewer, cleaner columns.

---

## What the Code Does

1. Generates 500 data points with 5 features and 3 clusters using `make_blobs`
2. Scales the data with `StandardScaler` — this is important because PCA is sensitive to scale
3. Applies PCA to compress those 5 features down to just 2
4. Plots the result as a scatter plot — PC1 on the x-axis, PC2 on the y-axis, coloured by cluster

The result below shows that even after squashing 5 dimensions into 2, the 3 clusters are still cleanly separated:

![PCA Scatter Plot](images/principle%20component%20analysis%20scater%20plot.png)

This is the whole point — you've thrown away 3 out of 5 features, but the structure that matters (the 3 groups) is still clearly visible.

---

## Why Scale Before PCA?

PCA works by measuring how much each feature varies. If one feature ranges from 0 to 10,000 and another ranges from 0 to 1, the first one will completely dominate just because of its size — not because it's more important.

`StandardScaler` fixes this by pulling every feature to the same scale before PCA runs, so all features get a fair say.

---

## PCA in Real Life

- You have 200 features in a dataset and training is slow — run PCA and keep the top 20 components that explain 95% of the variance
- You want to visualise whether natural clusters exist in your data — compress to 2 dimensions and plot it
- Many of your features are correlated — PCA merges overlapping information into cleaner, independent components

---

## Files in This Folder

| File                                    | What it does                                                                   |
| --------------------------------------- | ------------------------------------------------------------------------------ |
| `dimensionality_reduction_algorithm.py` | PCA on a 5-feature dataset — compresses to 2 components and plots the clusters |

---

_Part of the Unsupervised Learning series in `Algorithms/Unsupervised/`._
