# DBSCAN — Density-Based Clustering of Personality Profiles

## Overview

This notebook applies **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)** to the Big Five personality traits, Organizational Citizenship Behavior (OCB), and Counterproductive Work Behavior (CWB).

Unlike K-Means, DBSCAN:
- Does **not** require specifying K in advance
- Identifies **outliers** explicitly (labeled as noise, −1)
- Finds clusters of **arbitrary shape**
- Defines clusters as dense regions separated by sparse regions

The primary goals of this analysis are to:
1. Determine whether density-based personality clusters exist in this sample
2. Identify **personality outliers** — participants whose profiles fall outside any dense region
3. Contextualize findings within the broader unsupervised analysis (PCA → K-Means → DBSCAN)

---

## Features Used

| Variable | Description | Scale |
|---|---|---|
| Extraversion | Sociability, assertiveness, positive affect | 1–5 |
| Agreeableness | Cooperation, trust, prosocial orientation | 1–5 |
| Conscientiousness | Organization, self-discipline, goal-directedness | 1–5 |
| Neuroticism | Emotional instability, anxiety, stress-proneness | 1–5 |
| Openness | Curiosity, creativity, intellectual engagement | 1–5 |
| OCB | Organizational Citizenship Behavior (mean of OCB1–OCB10) | 1–5 |
| CWB | Counterproductive Work Behavior (mean of CWB1–CWB10) | 1–5 |

All features were standardized (mean = 0, SD = 1) prior to clustering.

---

## Methodology

### Key Parameters
| Parameter | Role |
|---|---|
| `eps` | Neighborhood radius — how close two points must be to be considered neighbors |
| `min_samples` | Minimum neighbors required to be a core point — higher = stricter, fewer clusters |

### Point Classification
Each participant is classified as:
- **Core point:** has ≥ `min_samples` neighbors within radius `eps`
- **Border point:** within `eps` of a core point but not itself a core point
- **Noise point (−1):** not within `eps` of any core point → personality outlier

### Parameter Selection
Two-stage grid search:

1. **Stage 1:** Rule-of-thumb `min_samples = 2 × n_features − 1 = 13`, eps ∈ [0.5, 3.25] → all combinations produced either 100% noise or 1 cluster
2. **Stage 2:** Expanded search across eps ∈ [1.0, 2.5] and min_samples ∈ [3, 5, 7, 10] → identified one meaningful combination

### Final Parameters
```python
DBSCAN(eps=1.5, min_samples=3)
```

---

## Results

### Parameter Tuning Summary

| eps | min_samples | n_clusters | noise_% | silhouette |
|---|---|---|---|---|
| 1.5 | 3 | 4 | 17.8% | 0.0534 |

This was the only parameter combination producing ≥ 2 clusters with noise < 30%. The near-zero silhouette score (0.0534) indicates minimal separation between clusters.

### Cluster Output
- **Clusters found:** 4
- **Noise points (outliers):** ~89 participants (17.8%)
- **Core participants:** ~411 participants (82.2%)

### Key Finding
The requirement for permissive parameters (min_samples=3, below the 7-dimensional rule of thumb) and the near-zero silhouette score confirm that **personality variation in this sample is continuously distributed** with no strong density-based structure. The 4 clusters represent locally dense regions within a continuous distribution rather than discrete personality types.

### Outlier Profile
The ~17.8% of participants classified as noise represent individuals with atypical personality combinations — the most psychologically interesting output of the DBSCAN analysis. Outlier profiling examines how their mean trait scores differ from the core group.

---

## Contextualizing the Findings

Across all three unsupervised analyses, a consistent picture emerges:

| Method | Result | Interpretation |
|---|---|---|
| PCA | Diffuse elliptical scatter, no gaps in biplot | Continuous personality variation |
| K-Means | K=2, silhouette=0.237 | Broad, weakly separated groupings |
| DBSCAN | Needs min_samples=3, silhouette≈0 | No strong density structure |

These findings consistently support a **dimensional rather than categorical** view of personality in this sample, in line with dominant models in personality psychology (McCrae & Costa, 1997; Haslam et al., 2012).

---

## Visualizations

The notebook produces the following plots:

1. **K-distance plot** — guides eps selection
2. **Parameter tuning table** — all eps × min_samples combinations
3. **Cluster scatter in PCA 2D space** — clusters + noise points marked as ×
4. **Cluster profile heatmap** — mean trait scores per cluster
5. **Outlier vs core comparison** — bar chart of mean trait differences

---

## Limitations

- `min_samples=3` is below the recommended rule of thumb for 7-dimensional data — clusters may reflect local density fluctuations rather than true subgroups
- Near-zero silhouette score indicates substantial cluster overlap
- Results are sensitive to `eps` — small parameter changes produce very different solutions
- DBSCAN assumes clusters of similar density, which may not hold for personality data
- Sample size (n=500) limits the density required for robust DBSCAN performance in 7 dimensions

---

## Dependencies

```
pandas
numpy
matplotlib
seaborn
scikit-learn
pathlib
itertools
```

Install with:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

## Usage

1. Run `pca.ipynb` first to generate `pca_coords.csv` (needed for visualization)
2. Open `Unsupervised Learning/DBSCAN/dbscan.ipynb`
3. Run all cells in order

```python
# Loads pca_coords.csv for visualization coordinates
# Clusters on raw standardized Big Five + OCB + CWB features
```

---

## References

- Ester, M., Kriegel, H. P., Sander, J., & Xu, X. (1996). A density-based algorithm for discovering clusters in large spatial databases with noise. *KDD-96 Proceedings*, 226–231.
- Haslam, N., Holland, E., & Kuppens, P. (2012). Categories versus dimensions in personality and psychopathology. *Psychological Medicine, 42*(5), 903–916.
- Schubert, E., Sander, J., Ester, M., Kriegel, H. P., & Xu, X. (2017). DBSCAN revisited, revisited. *ACM Transactions on Database Systems, 42*(3), 1–21.
