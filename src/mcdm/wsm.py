"""
Weighted Sum Model (WSM) - Modèle de référence / Baseline simple.
La consigne impose la comparaison avec un modèle simple de référence.
"""
import numpy as np
import pandas as pd


class WeightedSumModel:
    """
    Modèle de somme pondérée simple : S_i = sum(w_j * x_ij)
    """
    def __init__(self, weights=None):
        self.weights = weights

    def fit_predict(self, df_criteria: pd.DataFrame, weights: dict = None) -> pd.DataFrame:
        """
        Calcule le score de chaque candidat et renvoie le classement.
        """
        weights = weights or self.weights
        if weights is None:
            # Poids égaux par défaut
            n_cols = df_criteria.shape[1]
            weights = {col: 1.0 / n_cols for col in df_criteria.columns}

        # Vérification et alignement des poids
        w_vector = np.array([weights[col] for col in df_criteria.columns])
        
        # Normalisation simple Min-Max pour ramener entre 0 et 1 si nécessaire
        min_vals = df_criteria.min()
        max_vals = df_criteria.max()
        denom = (max_vals - min_vals).replace(0, 1)
        normalized = (df_criteria - min_vals) / denom

        scores = normalized.values @ w_vector
        result = df_criteria.copy()
        result["WSM_Score"] = scores
        result["WSM_Rank"] = result["WSM_Score"].rank(ascending=False, method="min").astype(int)
        return result.sort_values(by="WSM_Rank")
