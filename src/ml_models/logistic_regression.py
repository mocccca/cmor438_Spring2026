import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from sklearn.pipeline import Pipeline


def make_tertiles(series, labels=None):
    """
    Convert a continuous variable into tertile-based classes.
    """
    if labels is None:
        labels = ["Low", "Medium", "High"]

    return pd.qcut(
        series,
        q=3,
        labels=labels,
        duplicates="drop"
    )


def run_logistic_regression_classification(
    df,
    predictors,
    outcomes,
    test_size=0.2,
    random_state=42,
    class_method="tertiles"
):
    """
    Run logistic regression classification for one or more outcomes.

    Continuous outcomes are converted into classes before classification.
    """

    results = {}

    for outcome in outcomes:

        data = df[predictors + [outcome]].dropna().copy()

        if class_method == "tertiles":
            data[f"{outcome}_class"] = make_tertiles(data[outcome])
        else:
            raise ValueError("Currently only class_method='tertiles' is supported.")

        X = data[predictors]
        y = data[f"{outcome}_class"]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("logistic", LogisticRegression(
                max_iter=1000,
                random_state=random_state
            ))
        ])

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        metrics = {
            "algorithm": "Logistic Regression",
            "outcome": outcome,
            "task": "classification",
            "class_method": class_method,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision_macro": precision_score(
                y_test,
                y_pred,
                average="macro",
                zero_division=0
            ),
            "recall_macro": recall_score(
                y_test,
                y_pred,
                average="macro",
                zero_division=0
            ),
            "f1_macro": f1_score(
                y_test,
                y_pred,
                average="macro",
                zero_division=0
            ),
            "classification_report": classification_report(
                y_test,
                y_pred,
                zero_division=0
            ),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
            "model": model,
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "y_pred": y_pred,
            "n_samples": len(data)
        }

        results[outcome] = metrics

    return results