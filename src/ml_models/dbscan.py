import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors


class DBSCANAnalysis:
    """
    Reusable DBSCAN clustering analysis class.
    """

    def __init__(
        self,
        eps=0.5,
        min_samples=5,
        scale=True
    ):
        self.eps = eps
        self.min_samples = min_samples
        self.scale = scale

        self.scaler = StandardScaler() if scale else None
        self.model = None
        self.labels_ = None
        self.feature_names = None
        self.X_processed = None

    def fit(self, X):
        """Fit DBSCAN clustering model."""

        self.feature_names = list(X.columns)

        self.X_processed = (
            self.scaler.fit_transform(X)
            if self.scale else X.values
        )

        self.model = DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples
        )

        self.labels_ = self.model.fit_predict(self.X_processed)

        return self

    def fit_predict(self, X):
        """Fit DBSCAN and return cluster labels."""

        self.fit(X)

        return self.labels_

    def cluster_counts(self):
        """Return number of observations in each cluster, including noise."""

        return pd.Series(self.labels_).value_counts().sort_index()

    def n_clusters(self):
        """Return number of clusters, excluding noise cluster -1."""

        return len(set(self.labels_)) - (1 if -1 in self.labels_ else 0)

    def noise_count(self):
        """Return number of observations labeled as noise."""

        return int(np.sum(self.labels_ == -1))

    def noise_percentage(self):
        """Return percentage of observations labeled as noise."""

        return round(np.mean(self.labels_ == -1) * 100, 2)

    def silhouette(self):
        """
        Return silhouette score excluding noise points.
        Only works if at least 2 non-noise clusters exist.
        """

        mask = self.labels_ != -1

        labels_no_noise = self.labels_[mask]
        X_no_noise = self.X_processed[mask]

        if len(set(labels_no_noise)) < 2:
            return np.nan

        return silhouette_score(X_no_noise, labels_no_noise)

    def cluster_profiles(self, X):
        """Return mean feature values by DBSCAN cluster."""

        df = X.copy()
        df["Cluster"] = self.labels_

        return df.groupby("Cluster").mean().round(3)

    def parameter_search(
        self,
        X,
        eps_values,
        min_samples_values
    ):
        """
        Search DBSCAN parameter combinations using number of clusters,
        noise percentage, and silhouette score.
        """

        X_processed = (
            self.scaler.fit_transform(X)
            if self.scale else X.values
        )

        rows = []

        for eps in eps_values:
            for min_samples in min_samples_values:

                model = DBSCAN(
                    eps=eps,
                    min_samples=min_samples
                )

                labels = model.fit_predict(X_processed)

                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                noise_pct = np.mean(labels == -1) * 100

                if n_clusters >= 2:
                    mask = labels != -1

                    try:
                        sil = silhouette_score(
                            X_processed[mask],
                            labels[mask]
                        )
                    except ValueError:
                        sil = np.nan
                else:
                    sil = np.nan

                rows.append({
                    "eps": eps,
                    "min_samples": min_samples,
                    "n_clusters": n_clusters,
                    "noise_pct": round(noise_pct, 2),
                    "silhouette": sil
                })

        return pd.DataFrame(rows)

    def plot_k_distance(self, X, k=5):
        """
        Plot k-distance graph to help choose eps.
        """

        X_processed = (
            self.scaler.fit_transform(X)
            if self.scale else X.values
        )

        neighbors = NearestNeighbors(n_neighbors=k)
        neighbors_fit = neighbors.fit(X_processed)

        distances, indices = neighbors_fit.kneighbors(X_processed)

        k_distances = np.sort(distances[:, k - 1])

        plt.figure(figsize=(7, 5))
        plt.plot(k_distances)
        plt.xlabel("Points sorted by distance")
        plt.ylabel(f"{k}-Nearest Neighbor Distance")
        plt.title("DBSCAN k-Distance Plot")
        plt.tight_layout()
        plt.show()

    def plot_clusters(self, x, y):
        """
        Scatterplot of two variables colored by DBSCAN cluster labels.
        Noise points are labeled as -1.
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
        plt.title("DBSCAN Clusters")

        plt.legend(
            *scatter.legend_elements(),
            title="Cluster"
        )

        plt.tight_layout()
        plt.show()