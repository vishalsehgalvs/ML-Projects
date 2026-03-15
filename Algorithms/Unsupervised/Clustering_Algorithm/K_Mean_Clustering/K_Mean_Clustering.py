# ============================================================
# K-MEANS CLUSTERING — CENTROID-BASED CLUSTERING ALGORITHM
# ============================================================
# K-Means groups data points into K clusters by assigning each
# point to its nearest centroid, then iteratively moving each
# centroid to the average of its assigned points until stable.
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs  # generates synthetic blob-shaped cluster data
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# -------------------------------------------------------
# STEP 1 — Generate synthetic data with 3 known clusters
# make_blobs creates 500 data points grouped around 3 centres
# We use this so we already know the correct answer is K=3
# -------------------------------------------------------
X, y_true = make_blobs(n_samples=500, centers=3, cluster_std=0.60, random_state=42)
# X is a 2D numpy array of shape (500, 2) — 500 points, each with 2 features
# Example output:
# [[-6.1900632  -7.30201545]
#  [ 3.02174685  1.94059276]
#  [ 5.9537606   1.48819071]
#  [-2.74446251  8.13617716] ...]

# Convert to a DataFrame for easier handling
df = pd.DataFrame(X, columns=['Feature_1', 'Feature_2'])
# Example:
#      Feature_1  Feature_2
# 0    -6.190063  -7.302015
# 1     3.021747   1.940593
# 2     5.953761   1.488191
# 3    -2.744463   8.136177

# -------------------------------------------------------
# STEP 2 — Scale the features using StandardScaler
# K-Means relies on Euclidean distance, so unscaled features
# with different ranges will distort the distance calculation.
# StandardScaler transforms each feature to mean=0, std=1.
# -------------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)
# Example scaled output:
# [[-9.69818350e-01 -1.33993820e+00]
#  [ 9.63343617e-01  8.22475346e-02]
#  [ 1.57864695e+00  1.26351772e-02]
#  [-2.46735247e-01  1.03557927e+00] ...]

# -------------------------------------------------------
# STEP 3 — Elbow Method to find the optimal K
# Run K-Means for K=1 through K=10 and record the inertia
# (WCSS — Within Cluster Sum of Squares) at each K value.
# Plot and look for the "elbow" — where improvement flattens.
# -------------------------------------------------------
inertia = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Inertia values for K=1 to K=10:
# [1000.0, 297.9, 11.6, 9.75, 8.26, 6.92, 6.33, 5.70, 5.06, 4.76]
# The sharp drop from K=2 to K=3 and flatness after K=3 confirms K=3 is optimal

# Uncomment to visualise the elbow curve:
# plt.plot(K_range, inertia, marker='o')
# plt.show()

# -------------------------------------------------------
# STEP 4 — Fit final K-Means model with optimal K=3
# fit_predict() fits the model and returns the cluster label
# for every data point in a single step.
# -------------------------------------------------------
kmeans_final = KMeans(n_clusters=3, random_state=42)
cluster_label = kmeans_final.fit_predict(X_scaled)
# cluster_label is an array of integers (0, 1, or 2) — one per data point
# Each number indicates which cluster that point was assigned to
# Example: [1, 0, 0, 2, 0, 0, 1, 0, 0, 1, 0, 2, ...]

df['cluster'] = cluster_label

# -------------------------------------------------------
# STEP 5 — Visualise the clusters
# Colour each data point by its assigned cluster label.
# -------------------------------------------------------
sns.scatterplot(x=df['Feature_1'],
                y=df['Feature_2'],
                hue=df['cluster'],
                palette='viridis')
plt.show()