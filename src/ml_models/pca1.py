import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class PCAAnalysis:
    """
    Reusable PCA analysis class for unsupervised dimensionality reduction.
    """

    def __init__(self, n_components=None, scale=True, random_state=42):
        self.n_components = n_components
        self.scale = scale
        self.random_state = random_state
        self.scaler = StandardScaler() if scale else None
        self.pca = PCA(n_components=n_components, random_state=random_state)
        self.feature_names = None
        self.scores = None

    def fit(self, X):
        """Fit PCA to the dataset."""
        self.feature_names = list(X.columns)

        X_processed = self.scaler.fit_transform(X) if self.scale else X.values
        self.pca.fit(X_processed)

        return self

    def transform(self, X):
        """Transform data into principal component scores."""
        X_processed = self.scaler.transform(X) if self.scale else X.values
        scores = self.pca.transform(X_processed)

        columns = [f"PC{i+1}" for i in range(scores.shape[1])]
        self.scores = pd.DataFrame(scores, columns=columns, index=X.index)

        return self.scores

    def fit_transform(self, X):
        """Fit PCA and return principal component scores."""
        self.fit(X)
        return self.transform(X)

    def explained_variance(self):
        """Return explained variance ratio for each principal component."""
        return pd.DataFrame({
            "Component": [f"PC{i+1}" for i in range(len(self.pca.explained_variance_ratio_))],
            "Explained Variance Ratio": self.pca.explained_variance_ratio_,
            "Cumulative Explained Variance": np.cumsum(self.pca.explained_variance_ratio_)
        })

    def loadings(self):
        """Return PCA component loadings."""
        loadings = pd.DataFrame(
            self.pca.components_.T,
            index=self.feature_names,
            columns=[f"PC{i+1}" for i in range(self.pca.components_.shape[0])]
        )

        return loadings

    def top_loadings(self, component="PC1", top_n=10):
        """Return variables with the strongest loadings for one component."""
        loadings = self.loadings()[component]

        return (
            loadings
            .reindex(loadings.abs().sort_values(ascending=False).index)
            .head(top_n)
            .to_frame(name="Loading")
        )

    def plot_explained_variance(self):
        """Plot cumulative explained variance."""
        variance = np.cumsum(self.pca.explained_variance_ratio_)

        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(variance) + 1), variance, marker="o")
        plt.xlabel("Number of Principal Components")
        plt.ylabel("Cumulative Explained Variance")
        plt.title("PCA Cumulative Explained Variance")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_scores(self, pc_x="PC1", pc_y="PC2"):
        """Scatterplot of observations using two principal components."""
        if self.scores is None:
            raise ValueError("Run .fit_transform(X) before plotting scores.")

        plt.figure(figsize=(7, 5))
        plt.scatter(self.scores[pc_x], self.scores[pc_y], alpha=0.7)
        plt.xlabel(pc_x)
        plt.ylabel(pc_y)
        plt.title(f"PCA Score Plot: {pc_x} vs {pc_y}")
        plt.tight_layout()
        plt.show()

    def plot_biplot(self, features=None, scale=3.0):
        """
        Create PCA biplot using PC1 and PC2 scores plus feature loading arrows.
        """
        if self.scores is None:
            raise ValueError("Run .fit_transform(X) before creating a biplot.")

        if self.pca.components_.shape[0] < 2:
            raise ValueError("Biplot requires at least 2 principal components.")

        if features is None:
            features = self.feature_names

        loadings = self.pca.components_.T

        fig, ax = plt.subplots(figsize=(8, 7))

        ax.scatter(
            self.scores["PC1"],
            self.scores["PC2"],
            alpha=0.3,
            s=20,
            label="Participants"
        )

        for i, feature in enumerate(features):
            ax.annotate(
                "",
                xy=(loadings[i, 0] * scale, loadings[i, 1] * scale),
                xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", lw=2)
            )

            ax.text(
                loadings[i, 0] * scale * 1.12,
                loadings[i, 1] * scale * 1.12,
                feature,
                fontsize=10,
                fontweight="bold"
            )

        ax.axhline(0, linewidth=0.5, linestyle="--")
        ax.axvline(0, linewidth=0.5, linestyle="--")

        ax.set_xlabel(
            f"PC1 ({self.pca.explained_variance_ratio_[0] * 100:.1f}%)"
        )
        ax.set_ylabel(
            f"PC2 ({self.pca.explained_variance_ratio_[1] * 100:.1f}%)"
        )

        ax.set_title("PCA Biplot — Big Five + OCB + CWB")
        plt.tight_layout()
        plt.show()