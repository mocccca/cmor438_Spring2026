import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


class KMeansAnalysis:
    """
    Reusable K-Means clustering analysis class.
    """

    def __init__(
        self,
        n_clusters=None,
        scale=True,
        random_state=42
    ):
        self.n_clusters = n_clusters
        self.scale = scale
        self.random_state = random_state

        self.scaler = StandardScaler() if scale else None

        self.model = None
        self.labels_ = None
        self.centroids_ = None
        self.feature_names = None
        self.X_processed = None

    def find_best_k(self, X, max_k=10):
        """
        Find best number of clusters using silhouette score.
        """
        X_processed = (
            self.scaler.fit_transform(X)
            if self.scale else X.values
        )

        best_score = -1
        best_k = None

        scores = []

        for k in range(2, max_k + 1):

            km = KMeans(
                n_clusters=k,
                random_state=self.random_state,
                n_init=10
            )

            labels = km.fit_predict(X_processed)

            score = silhouette_score(X_processed, labels)

            scores.append({
                "k": k,
                "silhouette_score": score
            })

            if score > best_score:
                best_score = score
                best_k = k

        self.n_clusters = best_k

        return pd.DataFrame(scores), best_k

    def fit(self, X):
        """Fit K-Means clustering model."""

        if self.n_clusters is None:
            raise ValueError(
                "n_clusters is not set. "
                "Run find_best_k() first or specify n_clusters manually."
            )

        self.feature_names = list(X.columns)

        self.X_processed = (
            self.scaler.fit_transform(X)
            if self.scale else X.values
        )

        self.model = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10
        )

        self.model.fit(self.X_processed)

        self.labels_ = self.model.labels_

        self.centroids_ = pd.DataFrame(
            self.model.cluster_centers_,
            columns=self.feature_names
        )

        return self

    def predict(self, X):
        """Predict cluster labels for new observations."""

        X_processed = (
            self.scaler.transform(X)
            if self.scale else X.values
        )

        return self.model.predict(X_processed)

    def fit_predict(self, X):
        """Fit model and return cluster labels."""

        self.fit(X)

        return self.labels_

    def inertia(self):
        """Return K-Means inertia."""

        return self.model.inertia_

    def silhouette(self):
        """Return silhouette score."""

        return silhouette_score(
            self.X_processed,
            self.labels_
        )

    def cluster_profiles(self, X):
        """Return mean feature values by cluster."""

        df = X.copy()

        df["Cluster"] = self.labels_

        return df.groupby("Cluster").mean().round(3)

    def plot_clusters(self, x, y):
        """
        Scatterplot of two variables colored by cluster.
        """

        plt.figure(figsize=(7, 5))

        scatter = plt.scatter(
            x,
            y,
            c=self.labels_,
            alpha=0.7
        )

        plt.xlabel("PC1")
        plt.ylabel("PC2")

        plt.title("K-Means Clusters")

        plt.legend(
            *scatter.legend_elements(),
            title="Cluster"
        )

        plt.tight_layout()
        plt.show()

    def elbow_plot(self, X, max_k=10):
        """Plot elbow curve for selecting number of clusters."""

        X_processed = (
            self.scaler.fit_transform(X)
            if self.scale else X.values
        )

        inertias = []

        ks = range(1, max_k + 1)

        for k in ks:

            km = KMeans(
                n_clusters=k,
                random_state=self.random_state,
                n_init=10
            )

            km.fit(X_processed)

            inertias.append(km.inertia_)

        plt.figure(figsize=(7, 5))

        plt.plot(
            ks,
            inertias,
            marker="o"
        )

        plt.xlabel("Number of Clusters (k)")
        plt.ylabel("Inertia")

        plt.title("K-Means Elbow Plot")

        plt.tight_layout()
        plt.show()