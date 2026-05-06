# Decision / Regression Trees

## Objective

The objective of this project is to apply Decision Tree regression models to predict Big Five personality traits from free-text narrative responses. This project explores whether interpretable tree-based machine learning methods can identify linguistic patterns associated with personality traits.


## Task

This project performs:

- Text preprocessing using TF-IDF vectorization
- Decision Tree regression for personality prediction
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

The project also compares Decision Tree performance with ensemble tree-based methods such as Random Forests and Gradient Boosting.


## Dataset

The dataset contains:

- Free-text narrative responses
- Big Five personality trait scores

Narrative responses were transformed into numerical representations using TF-IDF vectorization before model training.


## Packages

### Custom Python Package

Reusable Decision Tree implementation was defined in:

```text
src/ml_models/decision_tree.py
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
    * DecisionTreeRegressor
    * TfidfVectorizer
    * Pipeline
    * KFold
* matplotlib


## References

* Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Grisel, O., … & Duchesnay, E. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12, 2825–2830.
* Breiman, L., Friedman, J., Olshen, R., & Stone, C. (1984). Classification and regression trees. Wadsworth International Group.