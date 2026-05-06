# Gradient Boosting

## Objective

The objective of this project is to apply Gradient Boosting regression models to predict Big Five personality traits from free-text narrative responses. This project explores whether sequential ensemble learning methods can improve personality prediction from linguistic features.


## Task

This project performs:

- Text preprocessing using TF-IDF vectorization
- Gradient Boosting regression for personality prediction
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

The project also compares Gradient Boosting performance with other tree-based approaches such as Decision Trees and Random Forests.


## Dataset

The dataset contains:

- Free-text narrative responses
- Big Five personality trait scores

Narrative text responses were transformed into numerical representations using TF-IDF vectorization before model training.


## Packages

### Custom Python Package

Reusable Gradient Boosting implementation was defined in:

```text
src/ml_models/gradient_boosting.py
```
The package includes reusable methods for:

* pipeline construction
* model training
* prediction
* evaluation
* cross-validation

### Imported Libraries

This project also uses:

* pandas
* numpy
* scikit-learn
    * GradientBoostingRegressor
    * TfidfVectorizer
    * Pipeline
    * KFold
* matplotlib


## References

* Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Grisel, O., … & Duchesnay, E. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12, 2825–2830.
* Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. Annals of Statistics, 29(5), 1189–1232.