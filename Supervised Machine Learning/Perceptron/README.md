# Perceptron Classification Notebook

## Overview

This notebook classifies people into low, medium, or high levels of workplace behavior using a **Perceptron** model.

Predictors:

- Extraversion
- Agreeableness
- Conscientiousness
- Neuroticism
- Openness

Outcomes:

- Low / Medium / High CWB
- Low / Medium / High OCB

This is a classification problem.


## What is a Perceptron?

The perceptron is a simple linear classification algorithm.

It learns a weighted decision rule:

```text
prediction = sign(w1x1 + w2x2 + ... + wnxn + bias)
```

When it makes mistakes, it updates the weights to improve future predictions.

It is one of the earliest machine learning algorithms and forms the basis of neural networks.


## Why It Works for This Case

The perceptron is useful here as a baseline classifier.

It tests whether a simple linear decision boundary is enough to separate low, medium, and high workplace behavior groups using personality traits.

Because it is simple and fast, it provides a benchmark for more advanced models.


## Imported Libraries

* pandas
* scikit-learn
    * train_test_split
    * StandardScaler
    * Perceptron
    * Pipeline
    * accuracy_score
    * precision_score
    * recall_score
    * f1_score
    * classification_report
    * confusion_matrix

## Pipeline

The workflow used in this notebook is:

1. Load preprocessed dataset.
2. Convert continuous CWB/OCB into:
    * Low
    * Medium
    * High
3. Select Big Five predictors.
4. Split data into training and test sets.
5. Standardize predictors.
6. Fit Perceptron model.
7. Predict classes on test set.
8. Evaluate using:
    * Accuracy
    * Precision
    * Recall
    * Macro F1
    * Confusion Matrix


## Interpretation

If the perceptron performs well, the groups are linearly separable.

If performance is weak, more flexible models may be needed.