"""
Techniques de Clustering (K-Means & CAH).
Directement alignées sur le Cours 2 de l'ECC (slides 38-77) :
Segmentation des candidats, calcul de la fonction de coût (inertie intra-classe),
choix optimal du nombre de clusters k via le score de Silhouette et la méthode du coude.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


class CandidateClustering:
    def __init__(self, n_clusters: int = 3, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = None

    def evaluate_optimal_k(self, df_features: pd.DataFrame, k_range: range = range(2, 8)) -> dict:
        """Calcule l'inertie et le score de silhouette pour différents k."""
        scaled_data = self.scaler.fit_transform(df_features)
        inertias = []
        silhouettes = []

        for k in k_range:
            km = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = km.fit_predict(scaled_data)
            inertias.append(km.inertia_)
            silhouettes.append(silhouette_score(scaled_data, labels))

        return {
            "k_values": list(k_range),
            "inertias": inertias,
            "silhouette_scores": silhouettes,
            "best_k_silhouette": int(list(k_range)[np.argmax(silhouettes)])
        }

    def fit_predict(self, df_features: pd.DataFrame, method: str = "kmeans") -> pd.Series:
        scaled_data = self.scaler.fit_transform(df_features)
        if method == "kmeans":
            self.model = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)
        elif method == "cah":
            self.model = AgglomerativeClustering(n_clusters=self.n_clusters)
        else:
            raise ValueError("Méthode inconnue : utiliser 'kmeans' ou 'cah'")

        cluster_labels = self.model.fit_predict(scaled_data)
        return pd.Series(cluster_labels, index=df_features.index, name="Cluster")
