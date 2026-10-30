"""
Classe de base pour les agents évaluateurs du Comité de Recrutement.
"""
from abc import ABC, abstractmethod
import pandas as pd


class BaseAgent(ABC):
    def __init__(self, name: str, role: str, weight: float = 1.0):
        self.name = name
        self.role = role
        self.weight = weight

    @abstractmethod
    def evaluate(self, candidate_profile: dict) -> dict:
        """
        Évalue un candidat et retourne un dictionnaire de notes et justifications.
        """
        pass

    def evaluate_batch(self, df_candidates: pd.DataFrame) -> pd.DataFrame:
        """Évalue une série de candidats."""
        evaluations = []
        for _, row in df_candidates.iterrows():
            evaluations.append(self.evaluate(row.to_dict()))
        return pd.DataFrame(evaluations)
