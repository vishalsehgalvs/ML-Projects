# DBSCAN — Density-Based Spatial Clustering of Applications with Noise

K-Means is great when clusters are circular blobs. But real-world data rarely looks that clean. DBSCAN was built specifically to handle the messy, irregular shapes that K-Means can't.

The core idea: points that are close together in a dense region belong to the same cluster. Points that are isolated, with no neighbours nearby, are noise.

---

## How Is It Different from K-Means?

| Feature                            | K-Means                                 | DBSCAN                                    |
| ---------------------------------- | --------------------------------------- | ----------------------------------------- |
| You choose the number of clusters? | Yes — required upfront                  | No — it figures it out automatically      |
| Works on circular shapes?          | Yes, great                              | Yes, great                                |
| Works on irregular shapes?         | No — struggles badly                    | Yes — handles crescents, rings, any shape |
| Handles outliers/noise?            | No — outliers get pulled into a cluster | Yes — marks them as noise (label = -1)    |
| Based on what?                     | Distance to a centroid                  | Density of points in a neighbourhood      |

---

## How Does It Work?

DBSCAN doesn't need you to pick K. Instead, you give it two things:

- **epsilon (eps)** — the radius of the imaginary circle drawn around each point. Any point that falls inside this circle is a "neighbour."
- **min_samples** — the minimum number of neighbours a point needs inside its epsilon circle to be considered a "core point."

**The process:**

1. Draw an imaginary circle of radius `eps` around a point.
2. Count how many other points are inside that circle.
3. If there are at least `min_samples` points inside — this is a **core point**. Start a cluster here.
4. Expand the cluster by doing the same for every neighbour of that core point.
5. Keep expanding until no more points can be reached.
6. Any point that isn't a core point and isn't close enough to any core point is marked as **noise** (gets label -1).

---

## The Clusters — Side by Side with K-Means

This is what DBSCAN produces on the same crescent moon dataset that K-Means completely fails on:

![DBSCAN Clusters](images/dbscan%20clusters.png)

DBSCAN correctly identifies the two crescent shapes as separate clusters, and handles any outlier points by labelling them as noise. K-Means would just cut the data in half with a straight line and call it done.

---

## Parameters Used in the Code

```python
dbscan = DBSCAN(eps=0.3, min_samples=5)
```

- `eps=0.3` — the radius of the neighbourhood circle. Points within 0.3 units of each other are considered neighbours.
- `min_samples=5` — a point needs at least 5 neighbours in its neighbourhood to be treated as a core point and start a cluster.

These values are found by trial and error for each dataset — there's no perfect formula. Too large an `eps` merges clusters together. Too small and everything gets flagged as noise.

---

## Anatomy of a DBSCAN Cluster

Every point in a DBSCAN run gets one of three labels:

| Point Type       | What it means                                                                    |
| ---------------- | -------------------------------------------------------------------------------- |
| **Core point**   | Has at least `min_samples` neighbours within `eps` — the cluster grows from here |
| **Border point** | Fewer than `min_samples` neighbours, but sits within the `eps` of a core point   |
| **Noise point**  | Doesn't belong to any cluster — too isolated. Gets label -1                      |

---

## Advantages of DBSCAN

- No need to specify the number of clusters beforehand
- Handles clusters of any shape — not just blobs
- Naturally identifies and ignores outliers/noise
- Works well when clusters have similar density

## Disadvantages of DBSCAN

- Choosing the right `eps` and `min_samples` requires some trial and error
- Struggles when clusters have very different densities
- Doesn't scale as efficiently as K-Means on very large datasets

---

## Files in This Folder

| File                  | What it does                                                                       |
| --------------------- | ---------------------------------------------------------------------------------- |
| `dbscan_algorithm.py` | DBSCAN on the make_moons dataset — directly comparable to the K-Means failure case |

---

_Part of the Unsupervised Learning series in `Algorithms/Unsupervised/`._
