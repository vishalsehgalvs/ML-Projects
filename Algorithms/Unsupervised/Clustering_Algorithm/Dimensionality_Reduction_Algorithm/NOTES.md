# Dimensionality Reduction — Notes

---

## What Are Dimensions?

Every feature (column) in your dataset is a dimension. A dataset with 10 columns exists in 10-dimensional space. Humans can visualise up to 3 dimensions. Models can theoretically handle thousands — but performance degrades.

---

## The Curse of Dimensionality

As you add more dimensions, the data becomes increasingly sparse. In high-dimensional space, all points start to look equally far apart from each other. This breaks distance-based algorithms (like K-Means, KNN) because distance stops being a meaningful signal.

Also: more features = more parameters = more overfitting risk = slower training.

Dimensionality reduction is one of the main tools for dealing with this.

---

## PCA — Principal Component Analysis

### What It Actually Does

PCA finds new axes (principal components) that capture the maximum variance in the data.

- **First principal component (PC1):** the axis along which the data varies the most
- **Second principal component (PC2):** perpendicular to PC1, captures the second most variance
- And so on...

Each component is a linear combination of the original features. The components are uncorrelated with each other (orthogonal).

### Explained Variance Ratio

Each principal component has an **explained variance ratio** — the percentage of the total variance in the data it captures.

```
PC1: 72%
PC2: 18%
PC3: 5%
...
```

You'd pick enough components to hit your threshold (usually 95% or 99%). In this example, PC1 + PC2 already explain 90%, and adding PC3 gets you to 95% — so you'd keep 3 components and drop the rest.

### The Practical Pattern in Code

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Always scale before PCA — variance is sensitive to scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Option 1: Specify number of components
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X_scaled)

# Option 2: Keep enough components to explain 95% of variance
pca = PCA(n_components=0.95)
X_reduced = pca.fit_transform(X_scaled)

# See how much variance each component explains
print(pca.explained_variance_ratio_)
```

### Important: Always Scale Before PCA

PCA is based on variance. A feature that ranges from 0–10,000 will have massively more variance than a feature ranging from 0–1, even if the 0–1 feature is more informative. StandardScaler puts everything on equal footing first.

---

## t-SNE — t-distributed Stochastic Neighbour Embedding

### What It Does

t-SNE maps high-dimensional data to 2D or 3D for visualisation. It works by:

1. Calculating similarities between every pair of points in high-dimensional space
2. Trying to reproduce those same similarities in 2D space
3. It's optimised so close neighbours in high-D stay close in 2D

### Key Difference from PCA

PCA preserves **global structure** — it finds the overall directions of maximum variance.

t-SNE preserves **local structure** — it keeps nearby points together, but doesn't care about the relative positions of distant clusters.

### When to Use t-SNE

- Visualising clusters in high-dimensional data
- Checking if your data naturally separates into groups before modelling
- Exploring embeddings (text vectors, image features, etc.)

### When NOT to Use t-SNE

- As preprocessing before training a model (use PCA for that)
- When you need to apply the same transform to new data (t-SNE can't)
- On very large datasets without sampling first (it's slow)

---

## PCA vs t-SNE — Decision Guide

**Use PCA when:**

- You need to reduce dimensions before training a model
- You want to speed up training without losing much information
- You need to apply the reduction to new data later
- You want to understand which features contribute most to variance

**Use t-SNE when:**

- You want to visualise the structure of your data in 2D
- You want to see if natural clusters exist before running K-Means
- You're exploring embeddings from a neural network or NLP model

---

## Quick Reference

| Term                     | Plain English                                                          |
| ------------------------ | ---------------------------------------------------------------------- |
| Dimension                | A feature / column in the dataset                                      |
| Dimensionality reduction | Compressing many columns into fewer columns while keeping key patterns |
| Principal component      | A new axis PCA creates that captures maximum variance                  |
| Explained variance ratio | What percentage of the data's variation this component represents      |
| Orthogonal               | Perpendicular — principal components don't overlap/correlate           |
| t-SNE                    | Technique for squashing high-D data into 2D for visualisation          |
| Curse of dimensionality  | Too many features → data gets sparse → models struggle                 |

---

_Part of the Unsupervised Learning series — see parent folder for full overview._
