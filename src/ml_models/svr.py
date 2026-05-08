import numpy as np
import pandas as pd

from sklearn.svm import SVR as SklearnSVR
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score


class SVR:
    """
    Reusable Support Vector Regression pipeline for text-based prediction.

    This class can be applied to any dataset where text input is used to predict
    one or more continuous outcome variables.

    Supports both linear SVR and non-linear SVR kernels such as RBF.
    """

    def __init__(
        self,
        kernel="linear",
        C=1.0,
        epsilon=0.1,
        gamma="scale",
        max_features_tfidf=100000,
        min_df=5,
        ngram_range=(1, 2),
        use_svd=False,
        n_components=300,
        random_state=42
    ):
        self.kernel = kernel
        self.C = C
        self.epsilon = epsilon
        self.gamma = gamma
        self.max_features_tfidf = max_features_tfidf
        self.min_df = min_df
        self.ngram_range = ngram_range
        self.use_svd = use_svd
        self.n_components = n_components
        self.random_state = random_state

        self.model = None
        self.results = {}
        self.outcomes = None

    def _build_pipeline(self):
        """Build a TF-IDF + optional SVD + SVR pipeline."""

        steps = [
            ("tfidf", TfidfVectorizer(
                lowercase=True,
                ngram_range=self.ngram_range,
                min_df=self.min_df,
                max_features=self.max_features_tfidf
            ))
        ]

        if self.use_svd:
            steps.append(
                ("svd", TruncatedSVD(
                    n_components=self.n_components,
                    random_state=self.random_state
                ))
            )

        steps.append(
            ("reg", MultiOutputRegressor(
                SklearnSVR(
                    kernel=self.kernel,
                    C=self.C,
                    epsilon=self.epsilon,
                    gamma=self.gamma
                )
            ))
        )

        return Pipeline(steps)

    def fit(self, X_train, y_train):
        """
        Fit SVR model to one or more continuous outcome variables.
        """

        self.outcomes = list(y_train.columns)

        self.model = self._build_pipeline()

        self.model.fit(X_train, y_train)

        return self

    def predict(self, X):
        """
        Predict all continuous outcome variables for new text data.
        """

        if self.model is None:
            raise ValueError("Model has not been fitted yet. Run .fit() first.")

        preds = self.model.predict(X)

        return pd.DataFrame(preds, columns=self.outcomes)

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
        Run k-fold cross-validation and return mean metrics by outcome variable.
        """

        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)

        kf = KFold(
            n_splits=cv,
            shuffle=True,
            random_state=self.random_state
        )

        rows = []

        for fold, (train_idx, test_idx) in enumerate(kf.split(X)):

            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]

            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]

            model = self._build_pipeline()

            model.fit(X_train, y_train)

            y_pred = pd.DataFrame(
                model.predict(X_test),
                columns=y.columns
            )

            for outcome in y.columns:

                rows.append({
                    "Fold": fold + 1,
                    "Outcome": outcome,
                    "MAE": mean_absolute_error(y_test[outcome], y_pred[outcome]),
                    "R²": r2_score(y_test[outcome], y_pred[outcome]),
                    "Pearson r": np.corrcoef(y_test[outcome], y_pred[outcome])[0, 1]
                })

        results = pd.DataFrame(rows)

        summary = results.groupby("Outcome").agg({
            "MAE": "mean",
            "R²": ["mean", "std"],
            "Pearson r": "mean"
        })

        summary.columns = [
            "CV MAE",
            "CV Mean R²",
            "CV Std R²",
            "Pearson r"
        ]

        return summary.round(4)