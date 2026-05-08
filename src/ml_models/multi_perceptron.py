import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


def run_mlp_regression(
    df,
    predictors,
    outcomes,
    hidden_layer_sizes=(10, 5),
    activation="relu",
    solver="adam",
    max_iter=2000,
    test_size=0.2,
    random_state=42
):
    """
    Run Multi-Layer Perceptron (MLP) regression models
    for one or more continuous outcomes.
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

        model = Pipeline([

            ("scaler", StandardScaler()),

            ("mlp", MLPRegressor(

                hidden_layer_sizes=hidden_layer_sizes,

                activation=activation,

                solver=solver,

                max_iter=max_iter,

                random_state=random_state
            ))
        ])

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        metrics = {
            "algorithm": "MLP Regression",
            "outcome": outcome,
            "task": "regression",
            "MAE": mean_absolute_error(y_test, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
            "R2": r2_score(y_test, y_pred),
            "model": model,
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "y_pred": y_pred,
            "n_samples": len(data),
            "hidden_layer_sizes": hidden_layer_sizes,
            "activation": activation,
            "solver": solver
        }

        results[outcome] = metrics

    return results