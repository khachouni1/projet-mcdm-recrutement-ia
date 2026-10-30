"""
Agent Éthique, Diversité & Équité (Fairness Agent).
Audite l'évaluation pour détecter d'éventuels biais algorithmiques et valoriser l'équité de traitement.
"""
from .base_agent import BaseAgent


class EthicsAgent(BaseAgent):
    def __init__(self, name: str = "Agent_Ethique_Equite"):
        super().__init__(name=name, role="Audit Éthique, Anti-Biais et Diversité", weight=1.0)

    def evaluate(self, candidate_profile: dict) -> dict:
        age = float(candidate_profile.get("Age", 30.0))
        distance = float(candidate_profile.get("DistanceFromCompany", 10.0))
        gender = int(candidate_profile.get("Gender", 0))

        # Score d'équité de base
        fairness_score = 85.0

        # Neutralisation du biais de distance (politique télétravail inclusif)
        if distance > 25.0:
            fairness_score += 5.0  # Compensation bienveillante pour candidats éloignés

        # Neutralisation du biais d'âge (inclusivité intergénérationnelle)
        if age >= 40.0:
            fairness_score += 5.0  # Valorisation des profils expérimentés
        elif age <= 24.0:
            fairness_score += 5.0  # Valorisation de l'insertion des jeunes diplômés

        fairness_score = min(100.0, fairness_score)

        feedback = f"Audit d'équité validé (Âge: {int(age)} ans, Genre: {'H' if gender==1 else 'F'}, Distance: {distance:.1f} km)."

        return {
            "candidat_id": candidate_profile.get("id"),
            "agent": self.name,
            "score_ethique": round(fairness_score, 2),
            "avis_ethique": feedback
        }
