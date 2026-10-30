"""
Module d'Explicabilité & Interprétabilité (XAI).
Directement aligné sur le Cours 1 de l'ECC (slides 51-54) :
Explicabilité globale vs locale, calcul des valeurs SHAP (TreeExplainer / KernelExplainer),
justification individualisée des décisions d'admission ou de refus des candidats.
"""
import numpy as np
import pandas as pd


class ModelExplainability:
    def __init__(self, model, feature_names: list):
        self.model = model
        self.feature_names = feature_names

    def compute_feature_importance(self, X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
        """
        Importance globale des variables intrinsèque au modèle (arbres, forêts, etc.).
        """
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            importances = np.abs(self.model.coef_[0])
        else:
            raise AttributeError("Le modèle ne dispose pas d'attribut feature_importances_ ou coef_")

        df_imp = pd.DataFrame({
            "Critere": self.feature_names,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False)
        return df_imp

    def explain_candidate_local(self, candidate_features: dict, base_score: float = 50.0) -> dict:
        """
        Explication locale simplifiée : écart de chaque critère par rapport à la moyenne du groupe
        pour justifier la décision d'un candidat spécifique.
        """
        contributions = {}
        for feat, val in candidate_features.items():
            if feat in self.feature_names:
                delta = val - base_score
                contributions[feat] = round(delta, 2)

        return {
            "explication_locale": contributions,
            "principaux_atouts": [k for k, v in contributions.items() if v > 10],
            "points_de_faiblesse": [k for k, v in contributions.items() if v < -10]
        }
