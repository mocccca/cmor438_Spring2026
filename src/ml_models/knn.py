import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score


class KNN:
    """
    Reusable K-Nearest Neighbors regression pipeline for text-based prediction.

    This class can be applied to any dataset where text input is used to predict
    one or more continuous outcome variables.
    """

    def __init__(
        self,
        n_neighbors=20,
        weights="distance",
        n_components=100,
        max_features_tfidf=100000,
        min_df=5,
        ngram_range=(1, 2),
        random_state=42
    ):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.n_components = n_components
        self.max_features_tfidf = max_features_tfidf
        self.min_df = min_df
        self.ngram_range = ngram_range
        self.random_state = random_state

        self.pipelines = {}
        self.best_k = {}
        self.results = {}

    def _build_pipeline(self, n_neighbors=None):
        """Build a TF-IDF + SVD + KNN regression pipeline."""

        if n_neighbors is None:
            n_neighbors = self.n_neighbors

        return Pipeline([
            ("tfidf", TfidfVectorizer(
                lowercase=True,
                ngram_range=self.ngram_range,
                min_df=self.min_df,
                max_features=self.max_features_tfidf
            )),
            ("svd", TruncatedSVD(
                n_components=self.n_components,
                random_state=self.random_state
            )),
            ("knn", KNeighborsRegressor(
                n_neighbors=n_neighbors,
                weights=self.weights
            ))
        ])

    def tune_k(self, X_train, y_train, k_values=None, cv=5, scoring="r2"):
        """
        Tune the number of neighbors separately for each outcome variable.
        """

        if k_values is None:
            k_values = [10, 15, 20, 25, 30, 35, 40, 50]

        param_grid = {"knn__n_neighbors": k_values}

        for outcome in y_train.columns:

            pipe = self._build_pipeline()

            grid = GridSearchCV(
                pipe,
                param_grid,
                cv=cv,
                scoring=scoring,
                n_jobs=-1
            )

            grid.fit(X_train, y_train[outcome])

            self.best_k[outcome] = grid.best_params_["knn__n_neighbors"]

        return self.best_k

    def fit(self, X_train, y_train):
        """
        Fit one KNN pipeline per continuous outcome variable.
        """

        for outcome in y_train.columns:

            k = self.best_k.get(outcome, self.n_neighbors)

            pipe = self._build_pipeline(n_neighbors=k)

            pipe.fit(X_train, y_train[outcome])

            self.pipelines[outcome] = pipe

        return self

    def predict(self, X):
        """
        Predict all outcome variables for new text data.
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

            k = self.best_k.get(outcome, self.n_neighbors)

            pipe = self._build_pipeline(n_neighbors=k)

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