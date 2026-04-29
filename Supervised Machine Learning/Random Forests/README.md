# Random Forest Implementation
## Overview

A Random Forest model was used to predict each Big Five personality trait from participants' narrative responses. Performance was evaluated using **5-fold cross-validation**, meaning the dataset was repeatedly split into training and test sets to estimate how well the model generalizes to new data.

The evaluation metrics reported are:

- **R²**: Proportion of variance in the trait explained by the model
- **MAE**: Mean Absolute Error
- **Pearson's r**: Correlation between predicted scores and actual scores

---

# What Pearson's r Tells Us

Pearson's r measures how strongly the predicted scores align with the real scores.

- **r = 1.00** → perfect positive prediction
- **r = 0.00** → no linear relationship
- **r = -1.00** → perfect negative relationship

In this context:

- Higher **r** means the model correctly ranks people higher or lower on the trait.
- Even if predictions are not exact, a positive **r** means the model captures meaningful signal.

For personality prediction tasks, modest correlations are common because personality is complex and difficult to infer perfectly from text.