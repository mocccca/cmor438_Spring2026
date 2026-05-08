## Rice - Spring 2026 - CMOR 438 Machine Learning Final Project

This repository is created by Xinyi Li, a current junior at Rice, to showcase all the projects for CMOR 438 (Instructor: Dr. Randy R. Davila) during Spring 2026. 

This repository implements machine learning algorithms to predict Big Five personality with narrative text data and analyze its relationships with organizational outcomes (e.g., organizational citizenship behaviors and counterproductive work behaviors). Contains both supervised and unsupervised approaches for exploratory analysis and prediction.

In addition to conducting analyses on the dataset, this repository also includes reusable Python machine learning packages developed within `src/ml_models/`, allowing machine learning workflows to be modularized outside of notebooks in a scikit-learn-style structure.

# Repository Structure

```text

cmor438_Spring2026/

│
├── src/
│   └── ml_models/
│       ├── linear_regression.py
│       ├── logistic_regression.py
│       ├── perceptron.py
│       ├── multi_perceptron.py
│       ├── knn.py
│       ├── svr.py
│       ├── decision_tree.py
│       ├── random_forest.py
│       ├── gradient_boosting.py
│       ├── pca.py
│       ├── k_means.py
│       └── dbscan.py
│

├── Supervised Machine Learning/
│   ├── Linear Regression/
│   ├── Logistic Regression/
│   ├── perceptron.py
│   ├── multi_perceptron.py
│   ├── K-Nearest Neighbors/
│   ├── Support Vector Regression/
│   ├── Decision Trees/
│   ├── Random Forests/
│   └── Gradient Boosting/
│

├── Unsupervised Machine Learning/
│   ├── PCA/
│   └── K-Means/
│   └── DBSCAN/
│

├── datasets/
├── README.md
└── requirements.txt
```

# Reusable Python Packages (src/ml_models/)

The src/ml_models/ directory contains reusable machine learning packages created for this repository. Rather than placing all code directly inside notebooks, reusable workflows were modularized into standalone Python files.

These packages are designed to:

* mimic scikit-learn-style workflows
* improve code organization and reproducibility
* separate reusable logic from analysis notebooks
* make workflows reusable across datasets

Most packages include reusable methods such as:

* fit()
* predict()
* transform()
* fit_transform()
* evaluate()
* cross_validate()

depending on the machine learning method.


# Supervised Learning Methods

The repository implements several supervised machine learning approaches.

Regression & Classification

* Linear Regression
* Logistic Regression

Text-Based Personality Prediction

Narrative text responses are transformed using TF-IDF vectorization and used to predict Big Five personality traits.

Implemented models include:

* K-Nearest Neighbors (KNN)
* Support Vector Regression (SVR)
* Decision Trees
* Random Forests
* Gradient Boosting

These analyses include:

* hyperparameter tuning
* cross-validation
* model comparison
* feature importance analysis
* evaluation using:
    * MAE
    * RMSE
    * R²
    * Pearson correlation
    * accuracy / F1 (classification tasks)


# Unsupervised Learning Methods

### Principal Component Analysis (PCA)

PCA was used to:

* reduce dimensionality
* examine latent personality/workplace behavior structures
* analyze explained variance
* interpret component loadings
* visualize behavioral profiles through biplots and heatmaps

### K-Means Clustering

K-Means clustering was used to:

* identify behavioral/personality profiles
* compare cluster structures
* evaluate silhouette scores and elbow plots
* generate cluster heatmaps and demographic comparisons

Clusters were then examined across:

* personality traits
* OCB/CWB
* age
* gender

### DBSCAN Clustering

DBSCAN clustering was used to:
* identify naturally occurring groups and outliers
* evaluate cluster structures using k-distance plots and parameter tuning
* compare clustering solutions across different parameters
* examine cluster patterns and demographic differences

# Workflow

The general workflow throughout the repository is:

1. Data preprocessing and cleaning
2. Feature engineering / vectorization
3. Application of reusable machine learning packages
4. Model fitting and evaluation
5. Visualization and interpretation
6. Comparison across machine learning approaches

The notebooks primarily focus on:

* applying models
* interpreting outputs
* visualizing results
* discussing behavioral implications

while reusable code is modularized inside src/ml_models/.


# Installation

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Install the repository in editable mode:
```bash
pip install -e .
```

# Example Imports
```python
from ml_models.random_forest import RandomForestPersonality
from ml_models.svr import SVRPersonality
from ml_models.pca import PCAAnalysis
from ml_models.k_means import KMeansAnalysis
```

**Dataset**

All data are from a dataset on life narratives and personlity assessments, which contains responses from 500 participants. Features include Big Five personality measures, organizational behaviors (organizational citizenship behaviors, counterwork behaviors) and personal narrative data.

⚠️ *Confidentiality*: This data is protected for academic use only. Unauthorized redistribution or commercial use is prohibited.

**Libraries**
Numpy, Pandas, Scikit-learn, Matplotlib, TensorFlow, SciPy, etc.
