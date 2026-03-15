# Dimensionality Reduction

Most real datasets have a lot of columns. Sometimes hundreds. Sometimes thousands. Each column is a "dimension" — a different piece of information about each row.

The problem: the more dimensions you have, the harder it gets for a model to find patterns. Data gets sparse, training gets slow, and the model can start to overfit. This is sometimes called the **curse of dimensionality**.

Dimensionality reduction is the technique of taking a high-dimensional dataset and compressing it into fewer dimensions — while keeping as much of the useful information as possible.

---

## Why Would You Do This?

- Your dataset has 500 columns but many are correlated — they're basically saying the same thing in different words
- Training is too slow because of the sheer size of the data
- You want to visualise data — you can't plot 100 dimensions, but you can plot 2 or 3
- You want to reduce noise — keeping only the most important variation in the data

Think of it like taking a 4K photo and compressing it to a much smaller file size while keeping it visually recognisable. Some information is lost, but the essential structure is preserved.

---

## The Key Idea — Keep What Matters, Drop What Doesn't

In a dataset, not all variation is useful. Some features move together (correlated), some barely vary at all, some are just noise. Dimensionality reduction finds the directions where the data varies the most, and builds a new (smaller) set of features that captures those directions.

You go from many original features → a few **principal components** (or similar) that summarise the most important structure.

---

## PCA — Principal Component Analysis

PCA is the most widely used dimensionality reduction technique.

**How it works in plain English:**

1. Centre the data (subtract the mean from each feature)
2. Find the direction in which the data spreads the most — this is the first principal component
3. Find the next direction of maximum spread that is perpendicular to the first — this is the second principal component
4. Repeat for as many components as you want
5. Project all your data onto these new axes

The result: instead of your original 100 columns, you might end up with 5 principal components that capture 95% of the variance in the data.

**Explained variance** is the key metric — it tells you how much of the original information each component keeps. You keep enough components to explain, say, 95% of the variance and drop the rest.

---

## t-SNE — For Visualisation

t-SNE (t-distributed Stochastic Neighbour Embedding) is a different technique specifically designed for visualisation. It's great at taking high-dimensional data and mapping it to 2D or 3D in a way that preserves the neighbourhood structure — points that were close together in the original space end up close together in the 2D map.

Use t-SNE when you want to see what's going on in your data visually. It's not suitable for feeding data into a model (it doesn't work on new data), but it's excellent for exploration.

---

## PCA vs t-SNE

| Feature                     | PCA                                   | t-SNE                                   |
| --------------------------- | ------------------------------------- | --------------------------------------- |
| Purpose                     | Preprocessing, feature reduction      | Visualisation only                      |
| Works on new data?          | Yes — apply the same transformation   | No — must rerun on full dataset         |
| Preserves global structure? | Yes                                   | No — only local neighbourhood structure |
| Interpretable components?   | Somewhat (loadings show contribution) | No                                      |
| Speed on large datasets?    | Fast                                  | Slow                                    |
| When to use                 | Before training a model               | Exploring and visualising clusters      |

---

## When Should You Use Dimensionality Reduction?

- You have more features than samples (common in medical/genomics data)
- Many of your features are highly correlated
- Your model is training too slowly
- You want to visualise clusters or structure in high-dimensional data
- You want to remove noise from the data before training

---

## Files in This Folder

| File                                    | What it does       |
| --------------------------------------- | ------------------ |
| `dimensionality_reduction_algorithm.py` | Code not added yet |

---

_Part of the Unsupervised Learning series in `Algorithms/Unsupervised/`._
