# Random Forests

## Objective

The objective of this project is to apply Random Forest regression models to predict Big Five personality traits from free-text narrative responses. This project examines whether ensemble-based machine learning methods can identify meaningful linguistic patterns associated with personality.

## Task

This project performs:

- Text preprocessing using TF-IDF vectorization
- Random Forest regression for personality prediction
- Hyperparameter tuning for Random Forest models
- Feature importance analysis
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

The project also visualizes the most important linguistic features contributing to personality prediction.

## Dataset

The dataset contains:

- Free-text narrative responses
- Big Five personality trait scores

Narrative responses were transformed into numerical features using TF-IDF vectorization before model training.

## Packages

### Custom Python Package

Reusable Random Forest implementation was defined in:

```text
src/ml_models/random_forest.py
```

The package includes reusable methods for:

- pipeline construction
- hyperparameter tuning
- model training
- prediction
- evaluation
- cross-validation
- feature importance visualization

### Imported Libraries

This project also uses:

* pandas
* numpy
* scikit-learn
    * RandomForestRegressor
    * TfidfVectorizer
    * Pipeline
    * GridSearchCV
    * KFold
* matplotlib

⸻

## References

* Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Grisel, O., … & Duchesnay, E. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12, 2825–2830.
* Breiman, L. (2001). Random forests. Machine Learning, 45(1), 5–32.
* Goldberg, L. R. (1990). An alternative “description of personality”: The Big-Five factor structure. Journal of Personality and Social Psychology, 59(6), 1216–1229.