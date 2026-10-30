"""
Agent Coordinateur & Consensus MCDM.
Orchestre les délibérations des agents experts, compile la matrice multicritère,
applique AHP pour la pondération de consensus et TOPSIS pour le classement final.
"""
import pandas as pd
from .hr_agent import HRAgent
from .tech_agent import TechAgent
from .experience_agent import ExperienceAgent
from .ethics_agent import EthicsAgent
from ..mcdm.ahp import AHP
from ..mcdm.topsis import TOPSIS
from ..mcdm.wsm import WeightedSumModel


class CoordinatorAgent:
    def __init__(self):
        self.hr_agent = HRAgent()
        self.tech_agent = TechAgent()
        self.exp_agent = ExperienceAgent()
        self.ethics_agent = EthicsAgent()

    def run_committee(self, candidates_list: list, pairwise_matrix: list = None) -> dict:
        """
        Exécute la délibération du comité sur l'ensemble des candidats.
        """
        records = []
        for cand in candidates_list:
            hr_eval = self.hr_agent.evaluate(cand)
            tech_eval = self.tech_agent.evaluate(cand)
            exp_eval = self.exp_agent.evaluate(cand)
            eth_eval = self.ethics_agent.evaluate(cand)

            records.append({
                "candidat_id": cand.get("id", "N/A"),
                "nom": cand.get("nom", "Inconnu"),
                "score_rh": hr_eval["score_rh"],
                "score_technique": tech_eval["score_technique"],
                "score_experience": exp_eval["score_experience"],
                "score_ethique": eth_eval["score_ethique"],
                "avis_rh": hr_eval["avis_rh"],
                "avis_technique": tech_eval["avis_technique"],
                "avis_experience": exp_eval["avis_experience"],
                "avis_ethique": eth_eval["avis_ethique"]
            })

        df_committee = pd.DataFrame(records).set_index("candidat_id")

        # 1. Calcul des poids AHP
        criteria = ["score_rh", "score_technique", "score_experience", "score_ethique"]
        ahp = AHP(criteria_names=criteria)
        if pairwise_matrix is not None:
            ahp.set_pairwise_matrix(pairwise_matrix)
        else:
            # Matrice standard de consensus (Tech > Exp > RH > Éthique)
            default_matrix = [
                [1.0,  1/3,  1/2,  2.0],
                [3.0,  1.0,  2.0,  4.0],
                [2.0,  1/2,  1.0,  3.0],
                [1/2,  1/4,  1/3,  1.0]
            ]
            ahp.set_pairwise_matrix(default_matrix)

        ahp_weights = ahp.compute_weights()

        # 2. Modèle Baseline : WSM
        wsm = WeightedSumModel()
        wsm_results = wsm.fit_predict(df_committee[criteria], weights=ahp_weights)

        # 3. Modèle Avancé : TOPSIS
        topsis = TOPSIS(weights=ahp_weights, benefit_criteria=criteria)
        topsis_results = topsis.fit_predict(df_committee[criteria])

        # Synthèse globale
        summary_df = df_committee.copy()
        summary_df["WSM_Rank"] = wsm_results["WSM_Rank"]
        summary_df["WSM_Score"] = wsm_results["WSM_Score"].round(4)
        summary_df["TOPSIS_Rank"] = topsis_results["TOPSIS_Rank"]
        summary_df["TOPSIS_Score"] = topsis_results["TOPSIS_Score"].round(4)

        return {
            "summary_df": summary_df.sort_values(by="TOPSIS_Rank"),
            "ahp_summary": ahp.get_summary(),
            "criteria_weights": ahp_weights
        }
