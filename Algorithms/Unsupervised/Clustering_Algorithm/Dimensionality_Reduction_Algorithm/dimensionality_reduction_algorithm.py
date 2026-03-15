# ============================================================
# DIMENSIONALITY REDUCTION — PCA AND t-SNE
# ============================================================
# Real datasets often have many features (columns). More features
# means slower training, sparseness, and overfitting risk.
#
# This file demonstrates two techniques:
#   PCA  — compresses features while keeping the most important
#           variation. Use before training a model.
#   t-SNE — maps data to 2D purely for visualisation. Not for
#            feeding into a model.
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris           # 4-feature dataset — good for reduction demo
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# -------------------------------------------------------
# STEP 1 — Load the Iris dataset
# 150 flowers, 4 measurements each:
#   sepal length, sepal width, petal length, petal width
# 3 species: setosa, versicolor, virginica
# We'll reduce 4 dimensions down to 2 so we can plot it
# -------------------------------------------------------
iris = load_iris()
X = iris.data          # shape (150, 4) — 150 samples, 4 features
y = iris.target        # 0, 1, or 2 — the species label
feature_names = iris.feature_names
target_names = iris.target_names

df = pd.DataFrame(X, columns=feature_names)
df['species'] = [target_names[label] for label in y]

# -------------------------------------------------------
# STEP 2 — Scale the features before PCA
# PCA is based on variance. A feature ranging 0-10,000 would
# completely overshadow a feature ranging 0-1.
# StandardScaler puts everything on the same scale first.
# -------------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[feature_names])

# -------------------------------------------------------
# STEP 3A — PCA: reduce 4 features down to 2 components
# The two principal components capture the two directions
# where the data varies the most.
# -------------------------------------------------------
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# How much of the original information does each component keep?
print("Variance explained by each component:")
print(f"  PC1: {pca.explained_variance_ratio_[0]*100:.1f}%")
print(f"  PC2: {pca.explained_variance_ratio_[1]*100:.1f}%")
print(f"  Total: {sum(pca.explained_variance_ratio_)*100:.1f}%")
# Expected output:
#   PC1: 72.9%
#   PC2: 22.8%
#   Total: 95.7%
# So just 2 components already capture 95.7% of all the information

df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
df_pca['species'] = df['species']

# -------------------------------------------------------
# STEP 3B — PCA with auto-select (keep 95% of variance)
# Instead of picking 2 manually, tell PCA to keep however
# many components it takes to explain 95% of the variance.
# -------------------------------------------------------
pca_auto = PCA(n_components=0.95, random_state=42)
X_pca_auto = pca_auto.fit_transform(X_scaled)
print(f"\nComponents needed to explain 95% variance: {pca_auto.n_components_}")

# -------------------------------------------------------
# STEP 4 — Plot the PCA result
# We can now visualise 4 dimensions in a 2D scatter plot
# by plotting PC1 vs PC2
# -------------------------------------------------------
sns.scatterplot(x=df_pca['PC1'],
                y=df_pca['PC2'],
                hue=df_pca['species'],
                palette='tab10')
plt.title('PCA — Iris dataset reduced to 2 dimensions')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

# -------------------------------------------------------
# STEP 5 — t-SNE: reduce 4 features to 2D for visualisation
# t-SNE shuffles points around in 2D to keep neighbouring
# points close together. Great for seeing cluster structure.
# Note: t-SNE results change with perplexity and random_state.
# Do not use t-SNE output as input to a model — for viz only.
# -------------------------------------------------------
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

df_tsne = pd.DataFrame(X_tsne, columns=['TSNE_1', 'TSNE_2'])
df_tsne['species'] = df['species']

sns.scatterplot(x=df_tsne['TSNE_1'],
                y=df_tsne['TSNE_2'],
                hue=df_tsne['species'],
                palette='tab10')
plt.title('t-SNE — Iris dataset visualised in 2D')
plt.xlabel('t-SNE dimension 1')
plt.ylabel('t-SNE dimension 2')
plt.show()
