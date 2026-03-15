# K-Means Clustering — Notes

---

## What Kind of Algorithm Is This?

K-Means is a **centroid-based clustering algorithm**. Every cluster is defined by its centroid — the average position of all the points in that cluster. Points are assigned to whichever centroid they're closest to.

---

## WCSS — Within Cluster Sum of Squares

WCSS is the metric K-Means minimises. For each cluster, you measure the distance between every point and its centroid, square it (so big distances count more), then add it all up across every cluster.

In plain words: **WCSS = sum of (distance from each point to its cluster centroid)²**

- **k** — the number of clusters you chose
- **each cluster** contributes its own sum of squared distances
- the centroid is the average position of all points in that cluster

The goal of K-Means is to find the centroid positions that make this total as small as possible.

**As WCSS decreases** → clusters are tighter → better clustering.

---

## The Elbow Method — Step by Step

1. Run K-Means for K = 1 through K = 10 (or more)
2. Record the WCSS (called `inertia` in scikit-learn) for each K value
3. Plot K on the x-axis versus WCSS on the y-axis
4. Find the "elbow" — the point where the curve bends and the improvement slows down sharply
5. That K is your optimal number of clusters

Why does WCSS always decrease as K increases? Because with more clusters, every point can be closer to its own centroid. With K equal to the number of data points, WCSS = 0 (every point is its own cluster). But that's useless — you want the natural groups, not one per point.

---

## Data Preprocessing — Why We Scale First

Before running K-Means, the data is always standardised using `StandardScaler`. This transforms each feature to have mean=0 and standard deviation=1.

Why this matters: K-Means uses Euclidean distance. If Feature_1 ranges from 0 to 10,000 and Feature_2 ranges from 0 to 1, then Feature_1 will completely dominate the distance calculation. Scaling puts both features on equal footing.

---

## K-Means Failure — When the Data Isn't Circular

K-Means assumes clusters are spherical (circular in 2D). It draws its decision boundaries as straight lines midway between centroids. This means it can't separate clusters that wrap around each other.

**Classic failure case — make_moons dataset:**
The data is shaped like two interlocking crescent moons. K-Means with K=2 will cut the data straight down the middle, misclassifying half the points because it's physically unable to draw a curved boundary.

The `K_Mean_Clustering_Failure.py` file demonstrates this using `make_moons`. It imports both `KMeans` and `DBSCAN` side-by-side so you can see the direct comparison — K-Means fails, DBSCAN handles it correctly.

---

## Key Parameters in scikit-learn

| Parameter      | What it does                                                                                  |
| -------------- | --------------------------------------------------------------------------------------------- |
| `n_clusters`   | Number of clusters (K) — the only required parameter                                          |
| `random_state` | Seeds the random centroid initialisation for reproducibility                                  |
| `n_init`       | How many times to run with different starting centroids and pick the best result (default 10) |
| `max_iter`     | Maximum number of iterations before stopping (default 300)                                    |

`kmeans.inertia_` — the WCSS value after fitting. This is what gets plotted in the elbow method.

`fit_predict()` — fits the model and returns the cluster label for every data point in one step.

---

## Quick Reference

| Concept            | Short Answer                                                   |
| ------------------ | -------------------------------------------------------------- |
| What is K?         | The number of clusters — you pick this manually                |
| What is WCSS?      | Total squared distance from each point to its cluster centroid |
| Elbow method       | Plot WCSS vs K, find the bend — that's your best K             |
| What is inertia?   | scikit-learn's name for WCSS after fitting                     |
| Why scale?         | So no single feature dominates the distance calculation        |
| When does it fail? | When clusters are not circular/blob-shaped                     |
| What fixes that?   | DBSCAN — it groups by density, not distance to a centroid      |

---

_Part of the Unsupervised Learning series — see parent folder for full overview._
