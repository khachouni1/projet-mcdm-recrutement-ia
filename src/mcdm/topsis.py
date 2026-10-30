"""
TOPSIS (Technique for Order Preference by Similarity to Ideal Solution)
Classe les candidats selon leur distance relative à la solution idéale positive et négative.
"""
import numpy as np
import pandas as pd


class TOPSIS:
    def __init__(self, weights: dict, benefit_criteria: list = None):
        """
        weights: dictionnaire {nom_critere: poids}
        benefit_criteria: liste des critères à maximiser (bénéfice). Les autres sont considérés à minimiser (coût).
        """
        self.weights = weights
        self.benefit_criteria = benefit_criteria

    def fit_predict(self, df_criteria: pd.DataFrame) -> pd.DataFrame:
        criteria = list(self.weights.keys())
        matrix = df_criteria[criteria].values.astype(float)
        weights_arr = np.array([self.weights[c] for c in criteria])

        # 1. Normalisation vectorielle
        norm_factor = np.sqrt(np.sum(matrix ** 2, axis=0))
        norm_factor[norm_factor == 0] = 1.0
        normalized_matrix = matrix / norm_factor

        # 2. Matrice normalisée pondérée
        weighted_matrix = normalized_matrix * weights_arr

        # 3. Détermination des solutions idéales A+ et A-
        benefit_mask = np.array([c in (self.benefit_criteria or criteria) for c in criteria])
        ideal_best = np.where(benefit_mask, np.max(weighted_matrix, axis=0), np.min(weighted_matrix, axis=0))
        ideal_worst = np.where(benefit_mask, np.min(weighted_matrix, axis=0), np.max(weighted_matrix, axis=0))

        # 4. Calcul des distances euclidiennes
        dist_best = np.sqrt(np.sum((weighted_matrix - ideal_best) ** 2, axis=1))
        dist_worst = np.sqrt(np.sum((weighted_matrix - ideal_worst) ** 2, axis=1))

        # 5. Proximité relative (Score de performance)
        closeness = dist_worst / (dist_best + dist_worst + 1e-12)

        # 6. Tableau des résultats
        results = df_criteria.copy()
        results["Distance_Ideale_Pos"] = dist_best
        results["Distance_Ideale_Neg"] = dist_worst
        results["TOPSIS_Score"] = closeness
        results["TOPSIS_Rank"] = results["TOPSIS_Score"].rank(ascending=False, method="min").astype(int)

        return results.sort_values(by="TOPSIS_Rank")
