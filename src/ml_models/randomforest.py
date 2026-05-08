import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score


class RandomForest:
    """
    Reusable Random Forest regression pipeline for text-based prediction.

    This class can be applied to any dataset where text input is used to predict
    one or more continuous outcome variables.
    """

    def __init__(
        self,
        n_estimators=200,
        max_depth=None,
        min_samples_split=5,
        max_features_tfidf=500,
        ngram_range=(1, 2),
        stop_words="english",
        random_state=42
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features_tfidf = max_features_tfidf
        self.ngram_range = ngram_range
        self.stop_words = stop_words
        self.random_state = random_state

        self.pipelines = {}
        self.best_params = {}
        self.results = {}

    def _build_pipeline(self, params=None):
        """Build a TF-IDF + Random Forest regression pipeline."""

        if params is None:
            params = {
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "min_samples_split": self.min_samples_split
            }

        return Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=self.max_features_tfidf,
                stop_words=self.stop_words,
                ngram_range=self.ngram_range
            )),
            ("rf", RandomForestRegressor(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                min_samples_split=params["min_samples_split"],
                random_state=self.random_state,
                n_jobs=-1
            ))
        ])

    def tune(self, X_train, y_train, cv=5, scoring="r2"):
        """
        Tune Random Forest hyperparameters separately for each outcome variable.
        """

        param_grid = {
            "rf__n_estimators": [100, 200],
            "rf__max_depth": [None, 10, 20],
            "rf__min_samples_split": [2, 5, 10]
        }

        base_pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=self.max_features_tfidf,
                stop_words=self.stop_words,
                ngram_range=self.ngram_range
            )),
            ("rf", RandomForestRegressor(
                random_state=self.random_state,
                n_jobs=-1
            ))
        ])

        for outcome in y_train.columns:

            grid = GridSearchCV(
                base_pipeline,
                param_grid,
                cv=cv,
                scoring=scoring,
                n_jobs=-1
            )

            grid.fit(X_train, y_train[outcome])

            self.best_params[outcome] = {
                "n_estimators": grid.best_params_["rf__n_estimators"],
                "max_depth": grid.best_params_["rf__max_depth"],
                "min_samples_split": grid.best_params_["rf__min_samples_split"]
            }

        return self.best_params

    def fit(self, X_train, y_train):
        """
        Fit one Random Forest model per continuous outcome variable.
        """

        for outcome in y_train.columns:

            params = self.best_params.get(outcome)

            pipe = self._build_pipeline(params=params)

            pipe.fit(X_train, y_train[outcome])

            self.pipelines[outcome] = pipe

        return self

    def predict(self, X):
        """
        Predict all continuous outcome variables for new text data.
        """

        preds = {}

        for outcome, pipe in self.pipelines.items():
            preds[outcome] = pipe.predict(X)

        return pd.DataFrame(preds)

    def evaluate(self, X_test, y_test):
        """
        Evaluate model using R², MAE, and Pearson correlation.
        """

        y_pred = self.predict(X_test)

        rows = []

        for outcome in y_test.columns:

            r2 = r2_score(y_test[outcome], y_pred[outcome])
            mae = mean_absolute_error(y_test[outcome], y_pred[outcome])
            r = np.corrcoef(y_test[outcome], y_pred[outcome])[0, 1]

            rows.append({
                "Outcome": outcome,
                "R²": round(r2, 4),
                "MAE": round(mae, 4),
                "Pearson r": round(r, 4)
            })

            self.results[outcome] = {
                "R²": r2,
                "MAE": mae,
                "Pearson r": r
            }

        return pd.DataFrame(rows).set_index("Outcome")

    def cross_validate(self, X, y, cv=5):
        """
        Run k-fold cross-validation for each continuous outcome variable.
        """

        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)

        kf = KFold(
            n_splits=cv,
            shuffle=True,
            random_state=self.random_state
        )

        rows = []

        for outcome in y.columns:

            params = self.best_params.get(outcome)

            pipe = self._build_pipeline(params=params)

            fold_r2 = []
            fold_mae = []
            fold_r = []

            for train_idx, test_idx in kf.split(X):

                X_train = X.iloc[train_idx]
                X_test = X.iloc[test_idx]

                y_train = y[outcome].iloc[train_idx]
                y_test = y[outcome].iloc[test_idx]

                pipe.fit(X_train, y_train)

                y_pred = pipe.predict(X_test)

                fold_r2.append(r2_score(y_test, y_pred))
                fold_mae.append(mean_absolute_error(y_test, y_pred))
                fold_r.append(np.corrcoef(y_test, y_pred)[0, 1])

            rows.append({
                "Outcome": outcome,
                "CV Mean R²": round(np.mean(fold_r2), 4),
                "CV Std R²": round(np.std(fold_r2), 4),
                "CV MAE": round(np.mean(fold_mae), 4),
                "Pearson r": round(np.mean(fold_r), 4)
            })

        return pd.DataFrame(rows).set_index("Outcome")

    def plot_feature_importance(self, outcome, top_n=15):
        """
        Plot top words/features used by the Random Forest model
        for predicting one outcome variable.
        """

        if outcome not in self.pipelines:
            raise ValueError(
                f"{outcome} has not been trained yet. "
                "Run .fit(X_train, y_train) before plotting feature importance."
            )

        pipe = self.pipelines[outcome]

        tfidf = pipe.named_steps["tfidf"]
        rf = pipe.named_steps["rf"]

        feature_names = tfidf.get_feature_names_out()
        importances = rf.feature_importances_

        top_idx = np.argsort(importances)[::-1][:top_n]

        top_words = feature_names[top_idx]
        top_scores = importances[top_idx]

        fig, ax = plt.subplots(figsize=(8, 5))

        ax.barh(top_words[::-1], top_scores[::-1])

        ax.set_xlabel("Random Forest Feature Importance")
        ax.set_title(f"Top {top_n} Features for Predicting {outcome}")

        plt.tight_layout()
        plt.show()