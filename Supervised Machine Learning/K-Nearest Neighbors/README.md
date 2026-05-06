# K-Nearest Neighbors (KNN)

## Objective

The objective of this project is to apply K-Nearest Neighbors (KNN) regression to predict Big Five personality traits from free-text narrative responses. This project explores whether linguistic patterns extracted from narrative text can be used to estimate personality scores.


## Task

This project performs:

- Text preprocessing using TF-IDF vectorization
- Dimensionality reduction using Truncated SVD
- KNN regression for personality prediction
- Hyperparameter tuning for selecting optimal *k*
- Model evaluation using:
  - Mean Absolute Error (MAE)
  - R²
  - Pearson correlation (*r*)
  - 5-fold cross-validation

The following Big Five traits were predicted:

- Extraversion
- Agreeableness
- Conscientiousness
- Neuroticism
- Openness

## Dataset

The dataset contains:

- Free-text narrative responses
- Big Five personality scores

Narrative responses were transformed into numerical representations using TF-IDF vectorization before model training.

## Packages

### Custom Python Package

Reusable KNN implementation was defined in:

```text
src/ml_models/knn.py
```

The package includes reusable methods for:

- model building
- training
- prediction
- evaluation
- cross-validation
- hyperparameter tuning


### Imported Libraries

This project also uses:

- pandas
- numpy
- scikit-learn
    - KNeighborsRegressor
    - TfidfVectorizer
    - TruncatedSVD
    - Pipeline
    - GridSearchCV
    - KFold
- matplotlib


### References

- Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., … & Duchesnay, E. (2011). *Scikit-learn: Machine learning in Python*. Journal of Machine Learning Research, 12, 2825–2830.
- Cover, T., & Hart, P. (1967). *Nearest neighbor pattern classification*. IEEE Transactions on Information Theory, 13(1), 21–27.
- Goldberg, L. R. (1990). *An alternative “description of personality”*: The Big-Five factor structure. Journal of Personality and Social Psychology, 59(6), 1216–1229.