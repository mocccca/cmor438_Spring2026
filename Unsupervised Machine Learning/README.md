# Unsupervised Machine Learning

Unsupervised learning is a type of machine learning where the model learns from **unlabeled data** — meaning, there are no predefined outputs or categories. Instead, the algorithm finds hidden structure, patterns, or groupings in the data on its own.

The workflow of unsupervised machine learning is usually:
- Collect and preprocess data: Standardize features so no variable dominates due to scale differences
- Choose a method: Based on the goal — dimensionality reduction, clustering, or outlier detection
- Fit the model: Let the algorithm find structure without any labels
- Evaluate and interpret: Use metrics (silhouette score, explained variance) and domain knowledge to assess meaningfulness
- Contextualize findings: Compare across methods to build a coherent picture

Below are the unsupervised machine learning methods applied in this project:
- Principal Component Analysis (PCA)
- K-Means Clustering
- DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

All three methods were applied to the same feature set: the **Big Five personality traits (Extraversion, Agreeableness, Conscientiousness, Neuroticism, Openness), Organizational Citizenship Behavior (OCB), and Counterproductive Work Behavior (CWB)** — 7 variables in total, all standardized to mean = 0, SD = 1.

---

## Research Question: Is Personality Variation in This Sample Continuous or Categorical?

A longstanding debate in personality psychology concerns whether personality is best represented as **continuous dimensions** (trait-based models, e.g., McCrae & Costa, 1997) or **discrete categories** (type-based models, e.g., Asendorpf et al., 2001). The unsupervised analyses in this project use data-driven methods to examine which view better characterizes this sample.

---

## Principal Component Analysis (PCA)

PCA was applied first to reveal the **underlying structure** of the 7-variable feature set and reduce dimensionality for downstream clustering.

### Key Results

| Component | Variance Explained | Cumulative |
|---|---|---|
| PC1 | 33.88% | 33.88% |
| PC2 | 20.94% | 54.82% |
| PC3 | 14.01% | 68.84% |
| PC4 | 10.33% | 79.17% |

**4 components were retained**, capturing 79.2% of total variance — just under the conventional 80% threshold. These components were saved as inputs for K-Means and DBSCAN.

### Component Interpretations

**PC1 — Positive Adjustment (33.9%)**  
Contrasts adaptive traits (Conscientiousness, Extraversion, Agreeableness) against stress-prone/counterproductive tendencies (Neuroticism, CWB). The dominant dimension in the data, reflecting a broad well-adjusted vs. stress-prone distinction consistent with Big Five theory.

**PC2 — Behavioral Engagement (20.9%)**  
Dominated by both OCB (0.596) and CWB (0.641) loading in the same positive direction — a notable finding suggesting that behavioral activation at work (whether prosocial or counterproductive) shares common variance, possibly reflecting extraversion-driven engagement.

**PC3 — Openness/Emotional Reactivity (14.0%)**  
Primarily driven by Openness (0.740) and Neuroticism (0.500), capturing intellectual curiosity and emotional reactivity independently of the first two components.

**PC4 — Agreeableness/OCB (10.3%)**  
Contrasts Agreeableness and OCB against Openness and Extraversion, potentially distinguishing cooperative/compliant individuals from independently minded ones.

### Biplot Finding
The PCA biplot showed a **diffuse, elliptical participant scatter** with no visible gaps or discrete groupings — an early indication that personality variation in this sample is continuously distributed rather than type-based.

---

## K-Means Clustering

K-Means was applied to the standardized raw features (7 variables) to identify natural personality groupings. As a robustness check, clustering was also performed on the 4-component PCA-reduced space; both approaches yielded the same optimal K, confirming result stability.

### Choosing K

| Method | Result |
|---|---|
| Elbow method | Smooth decline — no sharp inflection point |
| Silhouette score | Peaked at **K=2** (score = 0.237) |

The low silhouette score and absence of a clear elbow are consistent with the PCA biplot finding — no strongly separated natural clusters exist in this data.

### Cluster Profiles (K=2)

| Trait | Cluster 0 — "Stressed/Disengaged" (n=315) | Cluster 1 — "Adjusted/Prosocial" (n=185) |
|---|---|---|
| Extraversion | 2.88 | 3.69 |
| Agreeableness | 3.57 | 4.16 |
| Conscientiousness | 3.31 | 4.21 |
| Neuroticism | 3.23 | 2.27 |
| Openness | 3.66 | 4.12 |
| OCB | 2.62 | 3.02 |
| CWB | 1.68 | 1.40 |

**Cluster 0 — "Stressed/Disengaged" (n=315, 63%)**  
Lower scores on all positive Big Five traits, elevated Neuroticism (3.23) and CWB (1.68), lower OCB (2.62). Younger on average (M=25.1 years), 56.8% women.

**Cluster 1 — "Adjusted/Prosocial" (n=185, 37%)**  
Higher Conscientiousness (4.21), Agreeableness (4.16), and Extraversion (3.69), markedly lower Neuroticism (2.27), higher OCB (3.02), lower CWB (1.40). Older on average (M=30.5 years), more gender-balanced.

### Age & Gender Differences
The 5.4-year age gap between clusters is consistent with the **maturity principle** in personality development — Conscientiousness and Agreeableness tend to increase while Neuroticism decreases across adulthood (Roberts et al., 2006). Gender differences were modest and should be interpreted cautiously given the correlational design.

---

## DBSCAN

DBSCAN was applied to identify density-based clusters and personality outliers without requiring a pre-specified K.

### Parameter Selection
A two-stage grid search was conducted:
1. **Stage 1** (min_samples = 13, rule of thumb): All combinations produced either 100% noise or 1 cluster
2. **Stage 2** (min_samples ∈ [3, 5, 7, 10], eps ∈ [1.0, 2.5]): One meaningful combination found

**Final parameters:** `eps=1.5, min_samples=3`  
**Silhouette score:** −0.009 (near zero)

### Cluster Output

| Label | n | % | Profile |
|---|---|---|---|
| Cluster 0 — "Typical" | 399 | 79.8% | Moderate scores across all traits |
| Cluster 1 — "Exemplary" | 3 | 0.6% | Conscientiousness=4.92, Neuroticism=1.11, OCB=4.37 |
| Cluster 2 — "Distressed" | 5 | 1.0% | Conscientiousness=2.42, Neuroticism=3.77, Agreeableness=2.53 |
| Cluster 3 — "Behaviorally Activated" | 4 | 0.8% | CWB=2.85, OCB=3.60 — highest behavioral engagement |
| Noise (outliers) | 89 | 17.8% | Atypical personality combinations |

### Critical Assessment
Clusters 1–3 contain only 3–5 participants each and are better understood as **extreme outlier pockets** than meaningful personality subgroups. The min_samples=3 threshold, while the only combination producing multiple clusters, is below the recommended rule of thumb for 7-dimensional data. The near-zero silhouette score confirms minimal separation.

The most meaningful DBSCAN output is the **89 noise points (17.8%)** — participants whose personality profiles are sufficiently atypical to fall outside any dense neighborhood in the feature space.

---

## Synthesis: Continuous vs. Categorical Personality

All three methods converge on the same conclusion:

| Method | Key Result | Implication |
|---|---|---|
| PCA | Diffuse elliptical scatter, no biplot gaps | Continuous variation |
| K-Means | K=2, silhouette=0.237 (weak) | Broad, overlapping groupings |
| DBSCAN | Needs min_samples=3, silhouette≈0, 3 tiny clusters | No density-based structure |

Together, these findings strongly support a **dimensional rather than categorical** view of personality in this sample. The two broad K-Means clusters (Stressed/Disengaged vs. Adjusted/Prosocial) likely reflect the same PC1 "positive adjustment" dimension identified in PCA, suggesting that what appears as a personality "type" distinction is better understood as the upper and lower halves of a continuous trait dimension.


---

## Connection to Supervised Learning Findings

The unsupervised findings complement the supervised learning results in meaningful ways:

- The **PC1 adjustment dimension** (Conscientiousness vs. Neuroticism) aligns with the supervised finding that CWB is predicted by low Conscientiousness, low Agreeableness, and high Neuroticism
- The **PC2 behavioral engagement finding** (OCB and CWB sharing variance) is consistent with the supervised finding that personality alone predicts OCB less well than CWB — if OCB reflects situational engagement rather than purely dispositional factors, it would appear less structured in unsupervised analyses too
- The **K-Means age difference** (Cluster 1 older by 5.4 years) suggests that the adjusted/prosocial personality profile associated with better workplace behavior (higher OCB, lower CWB) becomes more common with age — a developmental pattern worth exploring in future research

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

## Notebook Order

Run notebooks in this order — PCA must run first as it generates `pca_coords.csv` required by K-Means and DBSCAN:

```
1. Unsupervised Learning/PCA/pca.ipynb
        ↓ generates pca_coords.csv
2. Unsupervised Learning/K-Means/kmeans.ipynb
3. Unsupervised Learning/DBSCAN/dbscan.ipynb
```

---

## References

- Jolliffe, I. T. (2002). *Principal Component Analysis* (2nd ed.). Springer.
- MacQueen, J. (1967). Some methods for classification and analysis of multivariate observations. *KDD-96 Proceedings*, 226–231.
- Ester, M., Kriegel, H. P., Sander, J., & Xu, X. (1996). A density-based algorithm for discovering clusters in large spatial databases with noise. *KDD-96 Proceedings*, 226–231.
- McCrae, R. R., & Costa, P. T. (1997). Personality trait structure as a human universal. *American Psychologist, 52*(5), 509–516.
- Roberts, B. W., Walton, K. E., & Viechtbauer, W. (2006). Patterns of mean-level change in personality traits across the life course. *Psychological Bulletin, 132*(1), 1–25.
- Asendorpf, J. B., Borkenau, P., Ostendorf, F., & Van Aken, M. A. (2001). Carving personality description at its joints. *European Journal of Personality, 15*(3), 169–198.
- Haslam, N., Holland, E., & Kuppens, P. (2012). Categories versus dimensions in personality and psychopathology. *Psychological Medicine, 42*(5), 903–916.
- Spector, P. E., & Fox, S. (2010). Counterproductive work behavior and organisational citizenship behavior. *Human Resource Management Review, 20*(1), 72–81.
