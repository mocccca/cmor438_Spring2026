# Support Vector Regression (SVR)

## Objective

The objective of this project is to apply Support Vector Regression (SVR) models to predict Big Five personality traits from free-text narrative responses. This project compares both linear and non-linear SVR approaches to examine whether more complex kernels improve personality prediction performance.

## Task

This project performs:

- Text preprocessing using TF-IDF vectorization
- Optional dimensionality reduction using Truncated SVD
- Linear SVR and non-linear RBF SVR modeling
- Personality prediction for:
  - Extraversion
  - Agreeableness
  - Conscientiousness
  - Neuroticism
  - Openness
- Model evaluation using:
  - Mean Absolute Error (MAE)
  - R²
  - Pearson correlation (*r*)
  - 5-fold cross-validation

The project also compares linear and non-linear SVR performance across traits.


## Dataset

The dataset contains:

- Free-text narrative responses
- Big Five personality scores

Narrative text responses were transformed into numerical representations using TF-IDF vectorization before model training.


## Packages

### Custom Python Package

Reusable SVR implementation was defined in:

```text
src/ml_models/svr.py
```

The package includes reusable methods for:

- pipeline construction
- model training
- prediction
- evaluation
- cross-validation

Both linear and RBF kernel SVR models were implemented within the package.

### Imported Libraries

This project also uses:

- pandas
- numpy
- scikit-learn
    - SVR
    - MultiOutputRegressor
    - TfidfVectorizer
    - TruncatedSVD
    - Pipeline
    - KFold
- matplotlib


## References

- Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Grisel, O., … & Duchesnay, E. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12, 2825–2830.
- Drucker, H., Burges, C. J. C., Kaufman, L., Smola, A., & Vapnik, V. (1997). Support vector regression machines. Advances in Neural Information Processing Systems, 9, 155–161.