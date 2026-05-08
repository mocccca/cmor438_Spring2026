import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score


class GradientBoosting:
    """
    Reusable Gradient Boosting regression pipeline for text-based prediction.

    This class can be applied to any dataset where text input is used to predict
    one or more continuous outcome variables.
    """

    def __init__(
        self,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        min_samples_split=2,
        max_features_tfidf=500,
        ngram_range=(1, 2),
        stop_words="english",
        random_state=42
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features_tfidf = max_features_tfidf
        self.ngram_range = ngram_range
        self.stop_words = stop_words
        self.random_state = random_state

        self.pipelines = {}
        self.results = {}

    def _build_pipeline(self):
        """Build a TF-IDF + Gradient Boosting regression pipeline."""

        return Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=self.max_features_tfidf,
                stop_words=self.stop_words,
                ngram_range=self.ngram_range
            )),
            ("gb", GradientBoostingRegressor(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                random_state=self.random_state
            ))
        ])

    def fit(self, X_train, y_train):
        """
        Fit one Gradient Boosting model per continuous outcome variable.
        """

        for outcome in y_train.columns:

            pipe = self._build_pipeline()

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

            pipe = self._build_pipeline()

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