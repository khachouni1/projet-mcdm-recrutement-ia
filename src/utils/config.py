from pathlib import Path

# Chemins racines du projet
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SYNTHETIC_DATA_DIR = DATA_DIR / "synthetic"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Paramètres de reproductibilité
RANDOM_STATE = 42

# Critères d'évaluation par défaut
DEFAULT_CRITERIA = [
    "score_technique",
    "score_experience",
    "score_formation",
    "score_soft_skills",
    "score_ethique_diversite"
]
