import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def run_linear_regression(
    df,
    predictors,
    outcomes,
    test_size=0.2,
    random_state=42
):
    """
    Run linear regression models for one or more continuous outcomes.

    Parameters
    ----------
    df : pandas DataFrame
        Dataset containing predictor and outcome variables.

    predictors : list
        List of predictor column names.

    outcomes : list
        List of continuous outcome column names.

    test_size : float
        Proportion of data used for testing.

    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        Dictionary containing fitted models, metrics, coefficients,
        predictions, and train/test data for each outcome.
    """

    results = {}

    for outcome in outcomes:

        data = df[predictors + [outcome]].dropna().copy()

        X = data[predictors]
        y = data[outcome]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state
        )

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        coefficients = pd.DataFrame({
            "Predictor": predictors,
            "Coefficient": model.coef_
        })

        metrics = {
            "algorithm": "Linear Regression",
            "outcome": outcome,
            "task": "regression",
            "MAE": mean_absolute_error(y_test, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
            "R2": r2_score(y_test, y_pred),
            "coefficients": coefficients,
            "intercept": model.intercept_,
            "model": model,
            "y_test": y_test,
            "y_pred": y_pred,
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "n_samples": len(data)
        }

        results[outcome] = metrics

    return results