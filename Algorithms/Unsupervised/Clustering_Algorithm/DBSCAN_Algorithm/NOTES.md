# DBSCAN — Notes

---

## Full Name

**D**ensity-**B**ased **S**patial **C**lustering of **A**pplications with **N**oise

It is a non-parametric algorithm — meaning it doesn't assume your data follows any particular statistical distribution (like a bell curve). It just looks at where points are dense.

---

## The Two Parameters You Need to Set

### epsilon (eps)

The radius of the neighbourhood around each point. Think of it as drawing a circle around a point — `eps` is the size of that circle.

- Too small → almost every point is isolated → mostly noise
- Too large → everything merges into one big cluster → useless
- Right value → clusters emerge naturally from the dense regions

### min_samples

The minimum number of neighbouring points that must be inside a point's `eps` circle for it to be a core point.

- Higher `min_samples` → stricter definition of "dense" → smaller, tighter clusters, more noise points
- Lower `min_samples` → looser definition → more points get absorbed into clusters

---

## Point Classifications

When DBSCAN runs, every point ends up classified as one of three types:

**Core Point** — Has at least `min_samples` neighbours within distance `eps`. This point is "in the heart" of a dense region. Clusters grow outward from core points.

**Border Point** — Has fewer than `min_samples` neighbours within `eps`, but is within the `eps` distance of at least one core point. It gets included in the cluster but doesn't expand it further.

**Noise Point** — Not a core point, not reachable from any core point. These get label -1 in scikit-learn. They're the outliers.

---

## How Cluster Expansion Works (Reachability)

DBSCAN uses the concept of **density-reachability**:

- Point B is **directly density-reachable** from point A if: B is within `eps` of A, and A is a core point
- Point C is **density-reachable** from A if there's a chain of points A → B → C where each step is directly density-reachable
- Two points are **density-connected** if there's a mutual core point that can reach both

A cluster is a maximal set of density-connected points. This is why DBSCAN can find arbitrarily shaped clusters — it follows the density trail wherever it leads, without assuming any particular shape.

---

## Why DBSCAN Beats K-Means on Crescent/Ring Data

K-Means decides cluster membership by distance to a centroid. For a crescent shape, the centroid ends up in the empty space between the two tips — not actually inside the crescent. So K-Means misclassifies points on the "wrong" side of the straight boundary.

DBSCAN doesn't care about centroids. It just checks: is this point in a dense neighbourhood? If yes, it follows the density trail. The trail naturally follows the curve of the crescent, correctly grouping both halves of each moon.

---

## The Code Pattern

```python
# 1. Scale the data first (distances matter)
scaler = StandardScaler()
x_scaled = scaler.fit_transform(df)

# 2. Fit DBSCAN
dbscan = DBSCAN(eps=0.3, min_samples=5)
dbscan_labels = dbscan.fit_predict(x_scaled)
# Note: there is no separate .fit() and .predict() — DBSCAN doesn't generalise to new data
# fit_predict() does everything in one step on the training data

# 3. Labels: 0, 1, 2... are cluster IDs. -1 means noise.
df['dbscan_clusters'] = dbscan_labels
```

Unlike K-Means, DBSCAN doesn't support predicting cluster membership for new unseen points. It only labels the data it was trained on.

---

## Choosing eps — The k-Distance Graph Method

A useful heuristic for choosing `eps`:

1. For each point, calculate the distance to its k-th nearest neighbour (where k = `min_samples`)
2. Sort and plot these distances
3. Find the "elbow" in the graph — that's a good starting value for `eps`

This is the DBSCAN equivalent of the elbow method from K-Means.

---

## Quick Reference

| Parameter/Concept | What it means                                                          |
| ----------------- | ---------------------------------------------------------------------- |
| `eps`             | Neighbourhood radius — how close two points must be to be "neighbours" |
| `min_samples`     | Minimum neighbours to be a core point                                  |
| Core point        | Dense — has enough neighbours. Clusters grow from here                 |
| Border point      | Near a core point but not dense enough to be one                       |
| Noise point       | Isolated — label = -1                                                  |
| Non-parametric    | Makes no assumptions about data distribution                           |
| Doesn't need K    | Number of clusters emerges from the data automatically                 |

---

_Part of the Unsupervised Learning series — see parent folder for full overview._
