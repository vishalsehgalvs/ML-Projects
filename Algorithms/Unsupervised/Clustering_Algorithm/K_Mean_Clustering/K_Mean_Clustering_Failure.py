# ============================================================
# K-MEANS FAILURE CASE — NON-CIRCULAR / CRESCENT-SHAPED DATA
# ============================================================
# K-Means assumes clusters are roughly circular (blob-shaped)
# because it uses Euclidean distance to a centroid. When the
# actual cluster shapes are curved (like crescents), K-Means
# draws incorrect straight-line boundaries and misclassifies.
#
# This file uses the make_moons dataset to demonstrate the
# failure. For a direct comparison, see dbscan_algorithm.py
# which correctly separates the same data.
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_moons  # generates two crescent-moon shaped clusters
from sklearn.cluster import KMeans, DBSCAN  # both imported for side-by-side comparison
from sklearn.preprocessing import StandardScaler

# -------------------------------------------------------
# STEP 1 — Generate crescent-shaped data
# make_moons creates two interleaving half-circles.
# This shape is impossible for K-Means to separate correctly.
# -------------------------------------------------------
X, y = make_moons(n_samples=500, noise=0.05, random_state=42)
df = pd.DataFrame(X, columns=['Feature_1', 'Feature_2'])

# -------------------------------------------------------
# STEP 2 — Scale before clustering
# K-Means uses Euclidean distance — scaling ensures both
# features contribute equally to the distance calculation.
# -------------------------------------------------------
scaler = StandardScaler()
x_scaled = scaler.fit_transform(df)

# -------------------------------------------------------
# STEP 3 — Fit K-Means with K=2
# We know there are 2 real classes (two crescents).
# K-Means will attempt to split them — and fail.
# It draws a straight boundary cutting across both crescents
# instead of following the curved shape of each cluster.
# -------------------------------------------------------
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans_labels = kmeans.fit_predict(x_scaled)

df['kmeans_clusters'] = kmeans_labels

# -------------------------------------------------------
# STEP 4 — Visualise the (incorrect) K-Means result
# Notice how the two colours don't align with the crescent
# shapes — K-Means has incorrectly split both crescents.
# Uncomment plt.show() to display.
# -------------------------------------------------------
sns.scatterplot(x=df['Feature_1'],
                y=df['Feature_2'],
                hue=df['kmeans_clusters'],
                palette='tab10')
# plt.show()

