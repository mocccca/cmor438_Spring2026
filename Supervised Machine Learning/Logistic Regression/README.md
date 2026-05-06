# Logistic Regression Classification Notebook

# Overview

This notebook predicts whether people fall into low, medium, or high levels of workplace behavior using **Logistic Regression**.

The predictors are:

- Extraversion
- Agreeableness
- Conscientiousness
- Neuroticism
- Openness

The outcomes are categorized into:

- Low CWB / Medium CWB / High CWB
- Low OCB / Medium OCB / High OCB

This is a classification task.


# What is Logistic Regression?

Logistic Regression is a classification model that predicts probabilities of class membership.

Instead of predicting a continuous score, it estimates:

```text
P(class = Low)
P(class = Medium)
P(class = High)
```

The case is assigned to the class with the highest probability.

For more than two classes, multinomial logistic regression is used.

## Why It Works for This Case

This model is useful because some applied settings care about identifying risk groups rather than exact scores.

Examples:

* Who is high in CWB risk?
* Who is high in OCB potential?

Logistic regression works well when relationships are relatively linear and predictors are numeric.

It is also interpretable and stable on moderate sample sizes.


## Imported Libraries

* pandas
* scikit-learn
    * train_test_split
    * StandardScaler
    * LogisticRegression
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
6. Fit multinomial logistic regression model.
7. Predict class labels on test set.
8. Evaluate using:
    * Accuracy
    * Precision
    * Recall
    * Macro F1
    * Confusion Matrix


## Interpretation

Higher accuracy and macro F1 indicate better classification performance.

Confusion matrices show which groups are easiest or hardest to identify.

