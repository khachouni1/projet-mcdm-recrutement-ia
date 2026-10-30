"""
Agent Lead Technique.
Évalue l'adéquation technique exacte avec le poste sur la base de SkillScore.
"""
from .base_agent import BaseAgent


class TechAgent(BaseAgent):
    def __init__(self, name: str = "Agent_Lead_Tech"):
        super().__init__(name=name, role="Évaluation Technique & Hard Skills", weight=1.0)

    def evaluate(self, candidate_profile: dict) -> dict:
        skill_score = float(candidate_profile.get("SkillScore", 50.0))

        if skill_score >= 80:
            feedback = f"Excellence technique remarquable ({skill_score:.1f}/100)."
        elif skill_score >= 60:
            feedback = f"Compétences techniques solides et opérationnelles ({skill_score:.1f}/100)."
        else:
            feedback = f"Niveau technique intermédiaire ({skill_score:.1f}/100), montée en compétences requise."

        return {
            "candidat_id": candidate_profile.get("id"),
            "agent": self.name,
            "score_technique": round(skill_score, 2),
            "avis_technique": feedback
        }
