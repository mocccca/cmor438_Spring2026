import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score


class DecisionTreePersonality:
    """
    Reusable Decision Tree regression model for predicting Big Five
    personality traits from free-text narrative responses.
    """

    def __init__(
        self,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=1,
        max_features_tfidf=500,
        random_state=42
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features_tfidf = max_features_tfidf
        self.random_state = random_state
        self.pipelines = {}
        self.results = {}

    def _build_pipeline(self):
        """Build a TF-IDF + Decision Tree pipeline."""
        return Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=self.max_features_tfidf,
                stop_words="english",
                ngram_range=(1, 2)
            )),
            ("tree", DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state
            ))
        ])

    def fit(self, X_train, y_train):
        """Fit one Decision Tree model per Big Five trait."""
        for trait in y_train.columns:
            pipe = self._build_pipeline()
            pipe.fit(X_train, y_train[trait])
            self.pipelines[trait] = pipe

        return self

    def predict(self, X):
        """Predict all Big Five traits."""
        preds = {}

        for trait, pipe in self.pipelines.items():
            preds[trait] = pipe.predict(X)

        return pd.DataFrame(preds)

    def evaluate(self, X_test, y_test):
        """Evaluate model using R², MAE, and Pearson r."""
        y_pred = self.predict(X_test)

        rows = []
        for trait in y_test.columns:
            r2 = r2_score(y_test[trait], y_pred[trait])
            mae = mean_absolute_error(y_test[trait], y_pred[trait])
            r = np.corrcoef(y_test[trait], y_pred[trait])[0, 1]

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
        """Run k-fold cross-validation for each Big Five trait."""
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)

        kf = KFold(
            n_splits=cv,
            shuffle=True,
            random_state=self.random_state
        )

        rows = []

        for trait in y.columns:
            pipe = self._build_pipeline()

            fold_r2 = []
            fold_mae = []
            fold_r = []

            for train_idx, test_idx in kf.split(X):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y[trait].iloc[train_idx], y[trait].iloc[test_idx]

                pipe.fit(X_train, y_train)
                y_pred = pipe.predict(X_test)

                fold_r2.append(r2_score(y_test, y_pred))
                fold_mae.append(mean_absolute_error(y_test, y_pred))
                fold_r.append(np.corrcoef(y_test, y_pred)[0, 1])

            rows.append({
                "Trait": trait,
                "CV Mean R²": round(np.mean(fold_r2), 4),
                "CV Std R²": round(np.std(fold_r2), 4),
                "CV MAE": round(np.mean(fold_mae), 4),
                "Pearson r": round(np.mean(fold_r), 4)
            })

        return pd.DataFrame(rows).set_index("Trait")