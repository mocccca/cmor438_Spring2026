# src/ — Reusable Machine Learning Packages

This directory contains reusable Python packages developed for the machine learning analyses in this repository. Rather than implementing all workflows directly inside notebooks, reusable components were modularized into standalone Python files following a scikit-learn-style workflow.

The goal of this structure is to improve:
- code organization
- reproducibility
- modularity
- readability
- reusability across notebooks and datasets

These packages are designed primarily for coursework, experimentation, and applied machine learning analyses rather than production deployment.


# Design Philosophy

The packages in `src/ml_models/` are designed to:

- Follow reusable scikit-learn-style workflows
- Separate reusable machine learning logic from notebook analysis
- Keep notebooks focused on:
  - data cleaning
  - interpretation
  - visualization
  - model comparison
- Support both supervised and unsupervised learning workflows
- Make analyses easier to reproduce across datasets

Most classes follow familiar machine learning interfaces such as:

- `fit()`
- `predict()`
- `transform()`
- `fit_transform()`
- `evaluate()`
- `cross_validate()`

depending on the algorithm type.


# Package Structure

```text
src/
└── ml_models/
    ├── linear_regression.py
    ├── logistic_regression.py
    ├── perceptron.py
    ├── multi_perceptron.py
    ├── knn.py
    ├── svr.py
    ├── decision_tree.py
    ├── random_forest.py
    ├── gradient_boosting.py
    ├── pca.py
    ├── k_means.py
    └── dbscan.py
```

Each file contains reusable functions or classes for a specific machine learning method.

# Supervised Learning Packages

The following packages are used for supervised machine learning tasks:

# Regression & Classification

* linear_regression.py
    * reusable linear regression workflows
    * regression metrics and coefficient extraction
* logistic_regression.py
    * logistic regression classification workflows
    * tertile-based class creation
    * classification metrics and confusion matrices
* perceptron.py
    * Perceptron classification workflows
    * tertile-based class creation
    * standardized predictor variables
    * classification metrics and confusion matrices
* multi_perceptron.py
    * Multi-Layer Perceptron regression workflows
    * neural network regression models
    * customizable hidden layer architecture
    * regression metrics for continuous outcomes

# Text-Based Personality Prediction

These models primarily use TF-IDF vectorized narrative text data:

* knn.py
    * K-Nearest Neighbors regression
    * optional dimensionality reduction
    * hyperparameter tuning
* svr.py
    * linear and non-linear Support Vector Regression
    * optional Truncated SVD dimensionality reduction
* decision_tree.py
    * Decision Tree regression models
* random_forest.py
    * Random Forest regression models
    * feature importance visualization
* gradient_boosting.py
    * Gradient Boosting regression models

Most supervised learning packages include:

* training workflows
* prediction functions
* evaluation metrics
* cross-validation procedures

# Unsupervised Learning Packages

* pca.py
    * dimensionality reduction
    * explained variance computation
    * component loadings
    * PCA score extraction
    * biplots and visualization support
* k_means.py
    * K-Means clustering workflows
    * elbow plots
    * silhouette score optimization
    * cluster profiling
    * cluster visualization
* dbscan.py
    * density-based clustering workflows
    * k-distance plots for selecting `eps`
    * parameter search across `eps` and `min_samples`
    * noise point detection
    * cluster profiling and visualization

# Installation & Usage

From the project root:
```bash
pip install -e .
```

Example import:
```python
from ml_models.random_forest import RandomForestPersonality
from ml_models.pca import PCAAnalysis
from ml_models.k_means import KMeansAnalysis
```
Notebook analyses then apply these reusable packages to specific datasets and research questions.

# Intended Use

These packages were developed for:

* machine learning coursework
* applied behavioral data analysis
* personality prediction research
* experimentation with supervised and unsupervised learning methods
* improving reproducibility and organization of notebook-based analyses

The implementations prioritize clarity, modularity, and interpretability over production-scale optimization.