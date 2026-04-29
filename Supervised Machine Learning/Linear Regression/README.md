# Linear Regression Notebook

## Overview

This notebook examines whether Big Five personality traits can predict workplace behavior using **Linear Regression**.

The predictors are:

- Extraversion
- Agreeableness
- Conscientiousness
- Neuroticism
- Openness

The outcomes are:

- CWB (Counterproductive Work Behavior)
- OCB (Organizational Citizenship Behavior)

Because CWB and OCB are continuous scores, this is a regression problem.

---

## What is Linear Regression?

Linear Regression is one of the most widely used statistical and machine learning models for predicting numeric outcomes.

It estimates the relationship between predictors and an outcome by fitting a linear equation:

```text
Outcome = intercept + b1X1 + b2X2 + ... + bnXn
```

Each coefficient represents the expected change in the outcome associated with a one-unit increase in that predictor, while holding the other predictors constant.

## Why It Works for This Case

Linear Regression is appropriate here because:

- The outcomes (CWB and OCB) are continuous.
- The predictors (Big Five traits) are numeric.
- Personality-outcome relationships are often approximately linear.
- The model is highly interpretable.

This makes it a strong baseline model for testing whether personality traits are related to workplace behavior.

---

## Pipeline

The workflow used in this notebook is:

1. Load preprocessed dataset.
2. Select predictors:
    * Extraversion
    * Agreeableness
    * Conscientiousness
    * Neuroticism
    * Openness
3. Select target:
    * CWB or OCB
4. Split data into training and test sets.
5. Fit Linear Regression model on training data.
6. Predict outcomes on test data.
7. Evaluate model using:
    * MAE
    * RMSE
    * R²
8. Inspect coefficients to understand which traits matter most.

---

## Interpretation

A positive coefficient means higher levels of that trait predict higher workplace behavior scores.

A negative coefficient means higher levels of that trait predict lower workplace behavior scores.

R² indicates how much variance in workplace behavior is explained by personality traits.