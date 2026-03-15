# K-Means Clustering

K-Means is one of the most popular clustering algorithms. You give it a bunch of data points and tell it how many groups (clusters) you want — and it figures out which points belong together, completely on its own.

The key idea: points that are close to each other probably belong to the same group.

---

## How Does It Work?

Think of it like this — you walk into a room of 500 strangers and have to split them into 3 groups by how similar they look, with no pre-set rules. You'd probably pick 3 people as your starting "representatives" and then assign everyone else to whoever their representative looks most like. Then you'd recalculate who the best representative for each group is, and reshuffle. Repeat until nothing changes. That's K-Means.

**The actual steps:**

1. **Decide K** — You pick how many clusters you want. This is the only thing you have to decide upfront.
2. **Place centroids randomly** — The algorithm picks K random points in the data as starting cluster centres (called centroids).
3. **Assign every point to the nearest centroid** — It uses Euclidean distance (straight-line distance) to figure out which centroid each point is closest to.
4. **Move the centroid to the middle** — Each centroid moves to the average position of all the points assigned to it.
5. **Repeat** — Reassign all points again based on the new centroid positions, move the centroids again, and keep going until no point changes its cluster.

---

## Scatter Plot — What the Clusters Look Like

This is the output when K-Means correctly groups the data into 3 clusters:

![Scatter Plot for Clusters](k_mean_clustering_images/scatter%20plot%20for%20clusters.png)

Each colour is a different cluster. The algorithm found these groupings on its own — no labels, no instruction on what the groups should be.

---

## How do You Pick the Right K? — The Elbow Method

K-Means needs you to tell it how many clusters to make. But what if you don't know? That's where the **Elbow Method** comes in.

**WCSS — Within Cluster Sum of Squares:**
For each cluster, you measure how far apart the points inside it are. Add all that up across every cluster — that's your WCSS. The smaller the WCSS, the tighter and better your clusters are.

The trick: you run K-Means with K=1, K=2, K=3... all the way to K=10 and plot the WCSS for each. As K increases, WCSS always goes down (more clusters = tighter groups). But at some point the improvement slows down dramatically — that's the "elbow". The elbow is your optimal K.

![Elbow Plot](k_mean_clustering_images/elbow%20plot.png)

In this example, the WCSS drops sharply from K=1 to K=3, then flattens out. The elbow is at **K=3** — that's the right number of clusters.

---

## When K-Means Fails — Non-Circular Data

K-Means has a hard limitation: it only works when clusters are roughly circular (or "blob-shaped"). This is because it uses Euclidean distance and assumes clusters form around a single central point.

When your data has a different shape — like two crescent moons facing each other — K-Means gets confused. It draws straight boundaries and splits clusters incorrectly.

![K-Means Failure on Non-Circular Data](k_mean_clustering_images/failure%20of%20k%20mean%20clustering%20when%20data%20is%20not%20circular.png)

This is exactly why DBSCAN was invented. DBSCAN handles these irregular shapes easily — see `../DBSCAN_Algorithm/` for a direct comparison.

---

## Advantages

- Simple to understand and implement
- Fast even on large datasets
- Works great when clusters are roughly circular and well-separated

## Disadvantages

- **You must choose K manually** — there's no automatic way to know the right number of clusters
- **Sensitive to outliers** — one extreme point can drag a centroid away from the real centre
- **Only works for circular/blob-shaped clusters** — fails on crescent, ring, or irregular shapes (demonstrated above)
- **Different runs can give different results** — because it starts with random centroids

---

## Files in This Folder

| File                           | What it does                                                             |
| ------------------------------ | ------------------------------------------------------------------------ |
| `K_Mean_Clustering.py`         | Main implementation — elbow method, fitting K-Means, scatter plot output |
| `K_Mean_Clustering_Failure.py` | Demonstrates where K-Means breaks down (non-circular data)               |

---

_Part of the Unsupervised Learning series in `Algorithms/Unsupervised/`._
