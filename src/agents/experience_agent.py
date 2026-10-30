"""
Agent Expérience & Séniorité.
Évalue le niveau de maturité professionnelle, le recul métier et le niveau d'études.
"""
from .base_agent import BaseAgent


class ExperienceAgent(BaseAgent):
    def __init__(self, name: str = "Agent_Experience", target_experience: float = 5.0):
        super().__init__(name=name, role="Évaluation Séniorité & Parcours", weight=1.0)
        self.target_experience = target_experience

    def evaluate(self, candidate_profile: dict) -> dict:
        years_exp = float(candidate_profile.get("ExperienceYears", 0.0))
        edu_level = float(candidate_profile.get("EducationLevel", 1.0)) # 1: Licence, 2: Master, 3: PhD, 4: Spécialisé

        # Score d'expérience : valorise l'expérience (max atteint à 10-12 ans)
        exp_component = min(100.0, (years_exp / 10.0) * 100.0)
        # Score de diplôme : échelonné de 60 à 100
        edu_component = 50.0 + (edu_level / 4.0) * 50.0

        final_exp_score = float(0.65 * exp_component + 0.35 * edu_component)

        feedback = f"{int(years_exp)} ans d'expérience, niveau d'études {int(edu_level)}/4 (Score: {final_exp_score:.1f}/100)."

        return {
            "candidat_id": candidate_profile.get("id"),
            "agent": self.name,
            "score_experience": round(final_exp_score, 2),
            "avis_experience": feedback
        }
