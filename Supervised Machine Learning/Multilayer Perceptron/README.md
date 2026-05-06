# Multilayer Perceptron Regression Notebook

## Overview

This notebook predicts workplace behavior using a neural network model called a **Multilayer Perceptron (MLP)**.

The predictors are the Big Five personality traits:

- Extraversion
- Agreeableness
- Conscientiousness
- Neuroticism
- Openness

The outcomes are:

- CWB
- OCB

Because the outcomes are continuous scores, this is a regression task.

---

## What is a Multilayer Perceptron?

A Multilayer Perceptron is a feedforward neural network.

It consists of:

1. Input layer (the predictors)
2. Hidden layers (learn internal patterns)
3. Output layer (predicted score)

Each node applies weights and nonlinear transformations to learn relationships in the data.

Unlike linear regression, MLP can model nonlinear and interaction effects.

---

## Why It Works for This Case

MLP is useful here because personality may influence workplace behavior in more complex ways than a straight linear relationship.

Examples:

- High conscientiousness may reduce CWB only when neuroticism is low.
- Extraversion may increase OCB more strongly at certain levels of agreeableness.

A neural network can learn these nonlinear combinations automatically.

---

## Imported Libraries

* pandas
* numpy
* scikit-learn
    * train_test_split
    * StandardScaler
    * MLPRegressor
    * Pipeline
    * mean_absolute_error
    * mean_squared_error
    * r2_score

## Pipeline

The workflow used in this notebook is:

1. Load preprocessed dataset.
2. Select Big Five traits as predictors.
3. Select CWB or OCB as target.
4. Split data into training and test sets.
5. Standardize predictors.
6. Fit MLP regression model.
7. Predict on test set.
8. Evaluate using:
   - MAE
   - RMSE
   - R²
9. Compare performance against Linear Regression.

---

## Interpretation

If MLP outperforms Linear Regression, it suggests nonlinear relationships exist.

If it performs similarly or worse, simpler linear relationships may explain the data sufficiently.