import numpy as np
import pandas as pd

from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score


class SVRPersonality:
    """
    Reusable Support Vector Regression model for predicting Big Five
    personality traits from free-text narrative responses.

    Supports both linear SVR and non-linear RBF SVR.
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
                SVR(
                    kernel=self.kernel,
                    C=self.C,
                    epsilon=self.epsilon,
                    gamma=self.gamma
                )
            ))
        )

        return Pipeline(steps)

    def fit(self, X_train, y_train):
        """Fit SVR model to all Big Five traits."""
        self.model = self._build_pipeline()
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X):
        """Predict all Big Five traits."""
        if self.model is None:
            raise ValueError("Model has not been fitted yet. Run .fit() first.")

        preds = self.model.predict(X)
        return pd.DataFrame(preds)

    def evaluate(self, X_test, y_test):
        """Evaluate model using R², MAE, and Pearson r."""
        y_pred = self.model.predict(X_test)

        rows = []
        for i, trait in enumerate(y_test.columns):
            r2 = r2_score(y_test.iloc[:, i], y_pred[:, i])
            mae = mean_absolute_error(y_test.iloc[:, i], y_pred[:, i])
            r = np.corrcoef(y_test.iloc[:, i], y_pred[:, i])[0, 1]

            rows.append({
                "Trait": trait,
                "R²": round(r2, 4),
                "MAE": round(mae, 4),
                "Pearson r": round(r, 4)
            })

            self.results[trait] = {
                "R²": r2,
                "MAE": mae,
                "Pearson r": r
            }

        return pd.DataFrame(rows).set_index("Trait")

    def cross_validate(self, X, y, cv=5):
        """Run k-fold cross-validation and return mean metrics by trait."""
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)

        kf = KFold(
            n_splits=cv,
            shuffle=True,
            random_state=self.random_state
        )

        rows = []

        for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            model = self._build_pipeline()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            for i, trait in enumerate(y.columns):
                rows.append({
                    "Fold": fold + 1,
                    "Trait": trait,
                    "MAE": mean_absolute_error(y_test.iloc[:, i], y_pred[:, i]),
                    "R²": r2_score(y_test.iloc[:, i], y_pred[:, i]),
                    "Pearson r": np.corrcoef(y_test.iloc[:, i], y_pred[:, i])[0, 1]
                })

        results = pd.DataFrame(rows)

        summary = results.groupby("Trait").agg({
            "MAE": "mean",
            "R²": ["mean", "std"],
            "Pearson r": "mean"
        })

        summary.columns = ["CV MAE", "CV Mean R²", "CV Std R²", "Pearson r"]
        return summary.round(4)