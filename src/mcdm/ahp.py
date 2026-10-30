"""
Analytic Hierarchy Process (AHP) - Méthode de Saaty.
Permet de dériver des poids mathématiquement rigoureux à partir de matrices de comparaisons par paires,
avec vérification du ratio de cohérence (CR < 0.10).
"""
import numpy as np
import pandas as pd


class AHP:
    # Random Index (RI) standard de Saaty selon le nombre de critères n
    RANDOM_INDEX = {
        1: 0.00,
        2: 0.00,
        3: 0.58,
        4: 0.90,
        5: 1.12,
        6: 1.24,
        7: 1.32,
        8: 1.41,
        9: 1.45,
        10: 1.49
    }

    def __init__(self, criteria_names: list):
        self.criteria_names = criteria_names
        self.n = len(criteria_names)
        self.comparison_matrix = np.eye(self.n)
        self.weights = None
        self.cr = None
        self.is_consistent = False

    def set_pairwise_matrix(self, matrix: np.ndarray):
        """Définit la matrice de comparaison par paires."""
        matrix = np.array(matrix, dtype=float)
        if matrix.shape != (self.n, self.n):
            raise ValueError(f"La matrice doit être de dimension ({self.n}, {self.n})")
        self.comparison_matrix = matrix
        return self

    def compute_weights(self) -> dict:
        """
        Calcule les poids selon la méthode du vecteur propre principal
        et évalue le ratio de cohérence (CR).
        """
        # Normalisation des colonnes
        col_sums = self.comparison_matrix.sum(axis=0)
        norm_matrix = self.comparison_matrix / col_sums
        
        # Poids = moyenne des lignes normalisées
        self.weights = norm_matrix.mean(axis=1)

        # Calcul de lambda_max pour la cohérence
        weighted_sum = self.comparison_matrix @ self.weights
        lambda_vec = weighted_sum / self.weights
        lambda_max = float(np.mean(lambda_vec))

        if self.n > 2:
            ci = (lambda_max - self.n) / (self.n - 1)
            ri = self.RANDOM_INDEX.get(self.n, 1.49)
            self.cr = ci / ri if ri > 0 else 0.0
            self.is_consistent = self.cr < 0.10
        else:
            self.cr = 0.0
            self.is_consistent = True

        return dict(zip(self.criteria_names, self.weights))

    def get_summary(self) -> dict:
        return {
            "criteria": self.criteria_names,
            "weights": dict(zip(self.criteria_names, self.weights)) if self.weights is not None else None,
            "consistency_ratio": self.cr,
            "is_consistent": self.is_consistent
        }
