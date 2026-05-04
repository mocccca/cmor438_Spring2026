# Principal Component Analysis (PCA) — Personality Structure in BFI-2 Life Narrative Data


## Overview

This notebook applies **Principal Component Analysis (PCA)** to the Big Five personality traits (Extraversion, Agreeableness, Conscientiousness, Neuroticism, Openness) along with Organizational Citizenship Behavior (OCB) and Counterproductive Work Behavior (CWB). 

PCA is an unsupervised dimensionality reduction technique that finds new axes (principal components) capturing the **maximum variance** in the data as linear combinations of the original features. Each component is orthogonal (uncorrelated) to the others and ordered by explained variance.

The primary goals of this analysis are to:
1. Reveal the **underlying structure** of personality and workplace behavior in this sample
2. Reduce 7 correlated variables into interpretable components
3. Produce a 2D visualization of participants in "personality space"
4. Generate reduced-dimension coordinates for downstream K-Means and DBSCAN clustering

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

All features were standardized (mean = 0, SD = 1) prior to PCA.

---

## Methodology

### Standardization
All variables were z-score standardized before PCA. This ensures that no variable dominates the components simply due to having higher variance — especially important when combining personality trait scores with behavioral composites (OCB, CWB).

### Component Selection
PCA was first fit retaining all 7 components. A **scree plot** with cumulative variance was used to identify the number of components to retain, using the conventional **80% variance threshold**.

### Outputs
- Full 7-component PCA for scree plot and loadings analysis
- 4-component PCA (PC1–PC4) for clustering inputs
- 2-component PCA (PC1–PC2) for 2D visualization and biplot

---

## Results

### Variance Explained

| Component | Individual (%) | Cumulative (%) |
|---|---|---|
| PC1 | 33.88 | 33.88 |
| PC2 | 20.94 | 54.82 |
| PC3 | 14.01 | 68.84 |
| **PC4** | **10.33** | **79.17** |
| PC5 | 8.05 | 87.22 |
| PC6 | 7.31 | 94.53 |
| PC7 | 5.47 | 100.00 |

**4 components were retained**, capturing 79.2% of total variance — just under the 80% threshold. These were used as input features for K-Means and DBSCAN clustering.

### PCA Loadings (First 4 Components)

| Variable | PC1 | PC2 | PC3 | PC4 |
|---|---|---|---|---|
| Extraversion | 0.439 | 0.330 | −0.238 | −0.335 |
| Agreeableness | 0.410 | −0.164 | 0.310 | 0.666 |
| Conscientiousness | 0.495 | −0.153 | −0.085 | −0.074 |
| Neuroticism | −0.469 | 0.149 | 0.500 | 0.113 |
| Openness | 0.283 | 0.228 | 0.740 | −0.465 |
| OCB | 0.227 | 0.596 | 0.009 | 0.456 |
| CWB | −0.206 | 0.641 | −0.206 | 0.040 |

### Component Interpretations

**PC1 — Positive Adjustment (33.9%)**  
Contrasts adaptive traits (Conscientiousness, Extraversion, Agreeableness) against stress-prone/counterproductive ones (Neuroticism, CWB). Reflects a broad well-adjusted vs. stress-prone dimension consistent with Big Five theory.

**PC2 — Behavioral Engagement (20.9%)**  
Dominated by OCB (0.596) and CWB (0.641) loading in the same direction, with moderate contributions from Extraversion and Openness. Captures behavioral activation at work regardless of valence — people high on PC2 engage strongly in both prosocial and counterproductive behaviors.

**PC3 — Openness/Emotional Reactivity (14.0%)**  
Primarily driven by Openness (0.740) and Neuroticism (0.500). Captures intellectual curiosity and emotional reactivity independently of the first two components.

**PC4 — Agreeableness/OCB vs. Openness (10.3%)**  
Contrasts Agreeableness (0.666) and OCB (0.456) against Openness (−0.465) and Extraversion (−0.335). May reflect a cooperative/compliant vs. independently minded distinction.

### Biplot Findings

The PCA biplot (PC1 × PC2) confirmed the above structure visually:
- Conscientiousness and Agreeableness arrows point right (+PC1); Neuroticism and CWB point left (−PC1)
- **OCB and CWB arrows both project upward (+PC2)** — a notable finding suggesting shared behavioral engagement variance
- OCB and Conscientiousness form ~90° angle, indicating near-zero correlation in this sample
- Participant scatter is diffuse and elliptical with no discrete groupings — consistent with continuous personality variation

---

## Visualizations

The notebook produces the following plots:

1. **Scree plot** — individual and cumulative explained variance with 80% threshold line
2. **2D participant scatter** — colored by OCB score and CWB score separately
3. **PCA biplot** — participant cloud with variable loading arrows
4. **Loadings heatmap** — component composition across first 4 PCs

---

## Output File

The notebook saves `pca_coords.csv` to the repo root containing:

| Column | Description |
|---|---|
| PC1–PC4 | 4-component PCA coordinates for clustering |
| PC1_vis, PC2_vis | 2-component PCA coordinates for visualization |
| Extraversion–CWB | Original feature values for cluster profiling |
| Gender, Age | Demographic variables for downstream analysis |

This file is required by `kmeans.ipynb` and `dbscan.ipynb`.

---

## Key Findings

- **Personality variation is continuous** — the diffuse biplot scatter and 4-component structure suggest no discrete personality types in this sample, consistent with dimensional models of personality (McCrae & Costa, 1997)
- **OCB and CWB share common variance on PC2** — both load positively on the behavioral engagement dimension, suggesting high-activation individuals engage in more workplace behavior of all kinds (Spector & Fox, 2010)
- **Conscientiousness is the strongest single contributor to PC1** (loading = 0.495), confirming its role as the primary adaptive personality trait
- **Neuroticism opposes Conscientiousness on PC1** (loading = −0.469), consistent with their well-established negative correlation in the Big Five literature

---

## Limitations

- PC1 and PC2 together explain only 54.8% of variance — 2D visualization omits substantial information
- PCA assumes **linear relationships** among variables; non-linear structure would not be captured
- Loadings reflect this specific sample and may not generalize to other populations
- The OCB/CWB behavioral engagement finding on PC2 warrants replication with larger, more diverse samples

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

1. Run `data_preprocessing.ipynb` (or include the preprocessing block) to generate `df`
2. Open `Unsupervised Learning/PCA/pca.ipynb`
3. Run all cells in order — `pca_coords.csv` will be saved to the repo root automatically

```python
# pca_coords.csv is required by downstream notebooks:
# - Unsupervised Learning/K-Means/kmeans.ipynb
# - Unsupervised Learning/DBSCAN/dbscan.ipynb
```

---

## References

- Jolliffe, I. T. (2002). *Principal Component Analysis* (2nd ed.). Springer.
- Spector, P. E., & Fox, S. (2010). Counterproductive work behavior and organisational citizenship behavior. *Human Resource Management Review, 20*(1), 72–81.
- Organ, D. W., & Ryan, K. (1995). A meta-analytic review of attitudinal and dispositional predictors of OCB. *Personnel Psychology, 48*(4), 775–802.
