# Dimensionality Reduction — Notes

---

## What Are Dimensions?

Every feature (column) in your dataset is a dimension. A dataset with 5 columns lives in 5-dimensional space. You can't visualise that — humans max out at 3. But you can compress it down to 2 dimensions and plot it, which is exactly what PCA lets you do.

---

## The Curse of Dimensionality

As you add more features, the data becomes increasingly sparse. In high-dimensional space, all points start to look equally far apart from each other. This breaks distance-based algorithms like K-Means and KNN because distance stops being a meaningful signal when everything is equally far.

More features also means more parameters, more overfitting risk, and slower training. PCA is one of the main tools for dealing with this before you build a model.

---

## PCA — How It Actually Works

PCA finds new axes that capture the most spread (variance) in the data.

- **PC1** — the direction where the data varies the most
- **PC2** — the direction with the second most variance, at a right angle to PC1
- And so on for however many components you want

Each principal component is a combination of the original features. The components don't overlap — they're independent of each other.

In the code, 5 original features get compressed into just PC1 and PC2. The scatter plot shows those 2 new axes, coloured by cluster label.

---

## Why Scaling Matters

PCA measures variance. A feature ranging 0-10,000 will look far more important than a feature ranging 0-1, purely because of its size. That's not right.

StandardScaler brings every feature to mean=0, standard deviation=1 before PCA runs. This makes sure each feature gets judged on its actual pattern, not its raw scale.

Rule: always scale before PCA.

---

## What the Code Produces

| Step                     | What happens                                                   |
| ------------------------ | -------------------------------------------------------------- |
| make_blobs(n_features=5) | Creates 500 points with 5 features and 3 known clusters        |
| StandardScaler           | Scales all 5 features to the same range                        |
| PCA(n_components=2)      | Compresses 5 features down to 2 principal components           |
| sns.scatterplot          | Plots PC1 vs PC2, coloured by cluster — 3 clean groups visible |

The output shows that even though 3 out of 5 features have been dropped, the 3 clusters are still clearly separated. PCA kept the important structure.

---

## Quick Reference

| Term                    | Plain English                                                                 |
| ----------------------- | ----------------------------------------------------------------------------- |
| Dimension               | A feature / column in the dataset                                             |
| PCA                     | Technique that compresses many features into fewer while keeping key patterns |
| Principal component     | A new axis PCA creates that captures maximum variance                         |
| PC1, PC2                | The first and second new axes — the ones with the most spread                 |
| Variance                | How spread out a feature is — PCA prioritises the most spread directions      |
| Curse of dimensionality | Too many features means data gets sparse and models struggle                  |
| StandardScaler          | Makes all features equally scaled before PCA so no one feature dominates      |

---

_Part of the Unsupervised Learning series — see parent folder for full overview._
