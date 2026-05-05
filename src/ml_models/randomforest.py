import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score


class RandomForestPersonality:
    """
    Reusable Random Forest model for predicting Big Five personality traits
    from free-text narrative responses.
    """

    def __init__(
        self,
        n_estimators=200,
        max_depth=None,
        min_samples_split=5,
        max_features_tfidf=500,
        random_state=42
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features_tfidf = max_features_tfidf
        self.random_state = random_state
        self.pipelines = {}
        self.best_params = {}
        self.results = {}

    def _build_pipeline(self, params=None):
        """Build a TF-IDF + Random Forest pipeline."""
        if params is None:
            params = {
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "min_samples_split": self.min_samples_split
            }

        return Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=self.max_features_tfidf,
                stop_words="english",
                ngram_range=(1, 2)
            )),
            ("rf", RandomForestRegressor(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                min_samples_split=params["min_samples_split"],
                random_state=self.random_state,
                n_jobs=-1
            ))
        ])

    def tune(self, X_train, y_train, cv=5):
        """Tune Random Forest hyperparameters separately for each trait."""
        param_grid = {
            "rf__n_estimators": [100, 200],
            "rf__max_depth": [None, 10, 20],
            "rf__min_samples_split": [2, 5, 10]
        }

        base_pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=self.max_features_tfidf,
                stop_words="english",
                ngram_range=(1, 2)
            )),
            ("rf", RandomForestRegressor(
                random_state=self.random_state,
                n_jobs=-1
            ))
        ])

        for trait in y_train.columns:
            grid = GridSearchCV(
                base_pipeline,
                param_grid,
                cv=cv,
                scoring="r2",
                n_jobs=-1
            )
            grid.fit(X_train, y_train[trait])
            self.best_params[trait] = {
                "n_estimators": grid.best_params_["rf__n_estimators"],
                "max_depth": grid.best_params_["rf__max_depth"],
                "min_samples_split": grid.best_params_["rf__min_samples_split"]
            }

        return self.best_params

    def fit(self, X_train, y_train):
        """Fit one Random Forest model per Big Five trait."""
        for trait in y_train.columns:
            params = self.best_params.get(trait)
            pipe = self._build_pipeline(params=params)
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
        """Run k-fold cross-validation for each trait."""
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)

        kf = KFold(
            n_splits=cv,
            shuffle=True,
            random_state=self.random_state
        )

        rows = []

        for trait in y.columns:
            params = self.best_params.get(trait)
            pipe = self._build_pipeline(params=params)

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
    
    def plot_feature_importance(self, trait, top_n=15):
        """
        Plot the top words/features used by the Random Forest model
        for predicting one Big Five trait.
        """
        if trait not in self.pipelines:
            raise ValueError(
                f"{trait} has not been trained yet. "
                "Run .fit(X_train, y_train) before plotting feature importance."
            )

        pipe = self.pipelines[trait]
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
        ax.set_title(f"Top {top_n} Words for Predicting {trait}")
        plt.tight_layout()
        plt.show()