"""
Analyse de Sensibilité pour les méthodes MCDM.
Évalue la robustesse des classements face aux variations des poids des critères.
"""
import numpy as np
import pandas as pd
from .topsis import TOPSIS


class SensitivityAnalysis:
    def __init__(self, df_criteria: pd.DataFrame, base_weights: dict):
        self.df_criteria = df_criteria
        self.base_weights = base_weights

    def evaluate_criterion_variation(self, target_criterion: str, variations: list = [-0.3, -0.15, 0.0, 0.15, 0.3]) -> pd.DataFrame:
        """
        Fait varier le poids d'un critère cible et renormalise les autres poids proportionnellement.
        Renvoie l'évolution des rangs des meilleurs candidats.
        """
        base_w = self.base_weights.copy()
        target_orig = base_w[target_criterion]
        results_records = []

        for delta in variations:
            new_target_w = max(0.01, min(0.90, target_orig * (1 + delta)))
            remaining_mass = 1.0 - new_target_w
            
            # Redistribution proportionnelle aux autres critères
            other_sum = sum(w for k, w in base_w.items() if k != target_criterion)
            adjusted_weights = {}
            for k, w in base_w.items():
                if k == target_criterion:
                    adjusted_weights[k] = new_target_w
                else:
                    adjusted_weights[k] = (w / other_sum) * remaining_mass

            # Exécution de TOPSIS avec les nouveaux poids
            model = TOPSIS(weights=adjusted_weights)
            ranked = model.fit_predict(self.df_criteria)
            
            for idx, row in ranked.iterrows():
                results_records.append({
                    "candidat_id": idx,
                    "target_criterion": target_criterion,
                    "variation_pct": delta * 100,
                    "adjusted_weight": new_target_w,
                    "rank": row["TOPSIS_Rank"],
                    "score": row["TOPSIS_Score"]
                })

        return pd.DataFrame(results_records)
