"""
Agent RH & Culture Fit.
Évalue les compétences relationnelles (soft skills), l'adéquation culturelle et la stabilité.
"""
from .base_agent import BaseAgent


class HRAgent(BaseAgent):
    def __init__(self, name: str = "Agent_RH"):
        super().__init__(name=name, role="Évaluation RH, Soft Skills et Stabilité", weight=1.0)

    def evaluate(self, candidate_profile: dict) -> dict:
        interview = float(candidate_profile.get("InterviewScore", 50.0))
        personality = float(candidate_profile.get("PersonalityScore", 50.0))
        prev_companies = float(candidate_profile.get("PreviousCompanies", 2.0))

        # Stabilité : 1 à 3 entreprises = optimal (score max), 5 = risque de volatilité
        stability = max(50.0, 100.0 - (prev_companies - 1) * 10.0)

        final_hr_score = float(0.50 * interview + 0.35 * personality + 0.15 * stability)

        if final_hr_score >= 75:
            feedback = f"Excellente aisance relationnelle et adéquation culturelle ({final_hr_score:.1f}/100)."
        elif final_hr_score >= 60:
            feedback = f"Profil relationnel équilibré ({final_hr_score:.1f}/100)."
        else:
            feedback = f"Points d'attention sur la communication ou l'intégration ({final_hr_score:.1f}/100)."

        return {
            "candidat_id": candidate_profile.get("id"),
            "agent": self.name,
            "score_rh": round(final_hr_score, 2),
            "avis_rh": feedback
        }
