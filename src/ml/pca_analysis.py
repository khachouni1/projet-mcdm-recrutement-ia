"""
Analyse en Composantes Principales (ACP / PCA).
Directement alignée sur le Cours 2 de l'ECC (slides 17-37) :
Formulation mathématique, recherche des facteurs principaux, variance expliquée, projection 2D.
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class CandidatePCA:
    def __init__(self, n_components: int = 2):
        self.n_components = n_components
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components)
        self.feature_names = None

    def fit_transform(self, df_features: pd.DataFrame) -> pd.DataFrame:
        self.feature_names = df_features.columns.tolist()
        scaled_data = self.scaler.fit_transform(df_features)
        coords = self.pca.fit_transform(scaled_data)

        col_names = [f"PC{i+1}" for i in range(self.n_components)]
        res_df = pd.DataFrame(coords, columns=col_names, index=df_features.index)
        return res_df

    def get_explained_variance(self) -> dict:
        return {
            "individual_ratio": self.pca.explained_variance_ratio_.tolist(),
            "cumulative_ratio": float(np.sum(self.pca.explained_variance_ratio_))
        }

    def get_loadings(self) -> pd.DataFrame:
        """Contribution des variables d'origine aux axes principaux."""
        col_names = [f"PC{i+1}" for i in range(self.n_components)]
        return pd.DataFrame(self.pca.components_.T, columns=col_names, index=self.feature_names)
