# ============================================================
# DBSCAN — DENSITY-BASED SPATIAL CLUSTERING OF APPLICATIONS
#          WITH NOISE
# ============================================================
# DBSCAN groups points that are close together in dense regions
# and marks isolated points as noise. Unlike K-Means, it does
# not require specifying K upfront and handles non-circular,
# irregularly-shaped clusters naturally.
#
# This file uses the same make_moons dataset as the K-Means
# failure case, so you can directly compare the results.
# K-Means fails on this shape. DBSCAN handles it correctly.
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_moons  # generates two crescent-moon shaped clusters
from sklearn.cluster import KMeans, DBSCAN  # both imported for direct comparison
from sklearn.preprocessing import StandardScaler

# -------------------------------------------------------
# STEP 1 — Generate the same crescent-shaped data as
# the K-Means failure case (same random_state for a fair
# side-by-side comparison)
# -------------------------------------------------------
X, y = make_moons(n_samples=500, noise=0.05, random_state=42)
df = pd.DataFrame(X, columns=['Feature_1', 'Feature_2'])

# -------------------------------------------------------
# STEP 2 — Scale the data
# DBSCAN uses epsilon as a distance threshold, so scaling
# is critical — unscaled features would make epsilon
# meaningless across features with different ranges.
# -------------------------------------------------------
scaler = StandardScaler()
x_scaled = scaler.fit_transform(df)

# -------------------------------------------------------
# STEP 3 — Fit DBSCAN
# eps=0.3   — the radius of each point's neighbourhood circle.
#             Any point within 0.3 units is a neighbour.
# min_samples=5 — a point needs at least 5 neighbours inside
#             its eps circle to be a core point.
# Points that can't reach any core point get label -1 (noise).
# -------------------------------------------------------
dbscan = DBSCAN(eps=0.3, min_samples=5)
dbscan_labels = dbscan.fit_predict(x_scaled)

df['dbscan_clusters'] = dbscan_labels

# -------------------------------------------------------
# STEP 4 — Visualise the DBSCAN result
# Compare this scatter plot with the K-Means version.
# DBSCAN correctly follows the crescent curves.
# Label -1 (noise points) will appear as a separate colour.
# -------------------------------------------------------
sns.scatterplot(x=df['Feature_1'],
                y=df['Feature_2'],
                hue=df['dbscan_clusters'],
                palette='tab10')
plt.show()