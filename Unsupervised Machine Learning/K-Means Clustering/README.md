# K-Means Clustering — Personality Profiles in BFI-2 Life Narrative Data

## Overview

This notebook applies **K-Means clustering** to identify natural personality profiles in the BFI-2 Life Narrative dataset. K-Means partitions participants into K groups by minimizing the within-cluster sum of squares (WCSS) — the total distance between each point and its cluster centroid.

Clustering was performed on the standardized raw feature space (Big Five traits + OCB + CWB). As a robustness check, clustering was also performed on the four-component PCA-reduced space (PC1–PC4, 79.2% variance retained); both approaches yielded the same optimal K, confirming result stability across feature representations.

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

### Choosing K
Two complementary methods were used to select the optimal number of clusters:

- **Elbow method:** plots WCSS against K — the elbow point indicates diminishing returns from adding more clusters
- **Silhouette score:** measures cohesion and separation of clusters (range: −1 to 1, higher is better)

The silhouette score peaked at **K=2** (score = 0.237). The elbow plot showed a smooth, gradual decline without a sharp inflection, consistent with continuous rather than discrete personality variation in this sample.

### Model
```python
KMeans(n_clusters=2, random_state=42, n_init=20)
```
`n_init=20` reruns K-Means with 20 different random initializations and retains the best result, reducing sensitivity to initialization.

### Cross-validation
A **K=3 sensitivity analysis** was also conducted to explore whether a three-cluster solution offered greater psychological interpretability, following established personality typology research (Asendorpf et al., 2001).

---

## Results

### Cluster Sizes
| Cluster | n | % |
|---|---|---|
| 0 — Stressed/Disengaged | 315 | 63% |
| 1 — Adjusted/Prosocial | 185 | 37% |

### Cluster Profiles

| Trait | Cluster 0 | Cluster 1 |
|---|---|---|
| Extraversion | 2.88 | 3.69 |
| Agreeableness | 3.57 | 4.16 |
| Conscientiousness | 3.31 | 4.21 |
| Neuroticism | 3.23 | 2.27 |
| Openness | 3.66 | 4.12 |
| OCB | 2.62 | 3.02 |
| CWB | 1.68 | 1.40 |

**Cluster 0 — "Stressed/Disengaged"**  
Lower scores on all positive Big Five traits, elevated Neuroticism (3.23) and CWB (1.68), lower OCB (2.62). Mean age 25.1 years, 56.8% women.

**Cluster 1 — "Adjusted/Prosocial"**  
Higher Conscientiousness (4.21), Agreeableness (4.16), and Extraversion (3.69), markedly lower Neuroticism (2.27), higher OCB (3.02), lower CWB (1.40). Mean age 30.5 years, more gender-balanced (48.6% women).

### Age & Gender Differences
- Cluster 1 participants were on average **5.4 years older** than Cluster 0, consistent with the **maturity principle** in personality development — Conscientiousness and Agreeableness tend to increase while Neuroticism decreases across adulthood (Roberts et al., 2006)
- Cluster 0 skewed more female (56.8% vs 48.6%), consistent with well-replicated small gender differences in Neuroticism

---

## Visualizations

The notebook produces the following plots:

1. **Elbow + Silhouette plots** — choosing K
2. **Cluster scatter in PCA 2D space** — participants colored by cluster, centroids marked
3. **Cluster profile heatmap** — mean trait scores per cluster
4. **Radar chart** — personality type fingerprints
5. **Gender composition bar chart** — per cluster
6. **Age boxplot + mean bar chart** — per cluster

---

## Interpretation

The two-cluster solution captures a broad **adjusted vs. stress-prone** distinction that aligns with the PC1 axis identified in PCA (positive adjustment dimension). The low silhouette score (0.237) confirms that personality variation in this sample is largely **continuous** rather than organized into discrete types.

---

## Dependencies

```
pandas
numpy
matplotlib
seaborn
scikit-learn
pathlib
```

Install with:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

## Usage

1. Run `pca.ipynb` first to generate `pca_coords.csv` in the repo root
2. Open `Unsupervised Learning/K-Means/kmeans.ipynb`
3. Run all cells in order

```python
# The notebook loads data from pca_coords.csv automatically
# No additional setup required
```

---

## References

- MacQueen, J. (1967). Some methods for classification and analysis of multivariate observations. *Proceedings of the 5th Berkeley Symposium on Mathematical Statistics and Probability.*
- Roberts, B. W., Walton, K. E., & Viechtbauer, W. (2006). Patterns of mean-level change in personality traits across the life course. *Psychological Bulletin, 132*(1), 1–25.
- Asendorpf, J. B., Borkenau, P., Ostendorf, F., & Van Aken, M. A. (2001). Carving personality description at its joints. *European Journal of Personality, 15*(3), 169–198.
