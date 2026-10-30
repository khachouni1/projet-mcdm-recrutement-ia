"""
Script d'exécution complet de la Phase 4 :
1. Système Multi-Agents (Lead Tech, RH & Fit, Expérience, Éthique)
2. Dérivation des poids par AHP de Saaty avec contrôle de cohérence (CR < 0.10)
3. Modèle de Référence (Baseline WSM) vs Modèle Avancé (TOPSIS)
4. Analyse de Sensibilité & Confrontation des Résultats
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

import sys
PROJECT_DIR = r"C:\Users\HP\.gemini\antigravity\scratch\PROJET MACHINE LEARNING"
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

# Import des modules du projet
from src.agents.coordinator_agent import CoordinatorAgent
from src.mcdm.ahp import AHP
from src.mcdm.topsis import TOPSIS
from src.mcdm.wsm import WeightedSumModel
from src.mcdm.sensitivity import SensitivityAnalysis
PROCESSED_DIR = os.path.join(PROJECT_DIR, "data", "processed")
FIGURES_DIR = os.path.join(PROJECT_DIR, "reports", "figures")
TEST_PATH = os.path.join(PROCESSED_DIR, "test.csv")
os.makedirs(FIGURES_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"font.size": 11, "figure.autolayout": True})

print("1. Chargement du Test Set pour l'évaluation décisionnelle (225 candidats)...")
test_df = pd.read_csv(TEST_PATH)
test_df["id"] = [f"CAND_{i+1:03d}" for i in range(len(test_df))]
test_df["nom"] = [f"Candidat #{i+1:03d}" for i in range(len(test_df))]

candidates_list = test_df.to_dict(orient="records")

# 2. Définition de la Matrice AHP du Comité (Saaty)
print("\n2. Délibération AHP : Matrice de comparaison par paires du jury...")
criteria = ["score_technique", "score_rh", "score_experience", "score_ethique"]
criteria_labels = ["Technique (Lead Tech)", "RH & Fit (Agent RH)", "Expérience (Agent Exp)", "Éthique (Fairness)"]

# Ordre d'importance défini par le comité : Technique > Expérience > RH > Éthique
pairwise_matrix = [
    [1.0,  3.0,  2.0,  4.0],  # Technique vs (Tech, RH, Exp, Eth)
    [1/3,  1.0,  1/2,  2.0],  # RH
    [1/2,  2.0,  1.0,  3.0],  # Expérience
    [1/4,  1/2,  1/3,  1.0]   # Éthique
]

ahp = AHP(criteria_names=criteria)
ahp.set_pairwise_matrix(pairwise_matrix)
ahp_weights = ahp.compute_weights()
ahp_summary = ahp.get_summary()

print("Poids dérivés par AHP (Somme = 1.0) :")
for c, w in ahp_weights.items():
    print(f"  • {c:20s} : {w*100:.2f}%")
print(f"Ratio de Cohérence AHP (CR) : {ahp_summary['consistency_ratio']:.4f}")
print(f"Cohérence mathématique validée (CR < 0.10) : {ahp_summary['is_consistent']}")

# 3. Évaluation par le Comité Multi-Agents
print("\n3. Exécution de la délibération du Comité d'Agents sur les 225 candidats...")
coordinator = CoordinatorAgent()
committee_result = coordinator.run_committee(candidates_list, pairwise_matrix=pairwise_matrix)
ranking_df = committee_result["summary_df"]

# Joindre la vérité terrain (HiringDecision réelle) et les attributs clés
ranking_df = ranking_df.merge(
    test_df[["id", "Age", "Gender", "EducationLevel", "ExperienceYears", "SkillScore", "InterviewScore", "HiringDecision"]].set_index("id"),
    left_index=True,
    right_index=True
)

# 4. Comparaison Baseline WSM vs TOPSIS
rho, p_val = spearmanr(ranking_df["WSM_Rank"], ranking_df["TOPSIS_Rank"])
rank_diff = (ranking_df["WSM_Rank"] - ranking_df["TOPSIS_Rank"]).abs()
inversions_count = (ranking_df["WSM_Rank"] != ranking_df["TOPSIS_Rank"]).sum()

print(f"\n4. Comparaison Baseline WSM vs Modèle Avancé TOPSIS :")
print(f"  • Corrélation de rang de Spearman (rho) : {rho:.4f} (p-value: {p_val:.2e})")
print(f"  • Nombre de candidats ayant changé de rang : {inversions_count} / {len(ranking_df)} ({inversions_count/len(ranking_df)*100:.1f}%)")
print(f"  • Écart moyen de classement               : {rank_diff.mean():.2f} places (Écart max: {rank_diff.max()} places)")

# Top 5 candidats retenus par TOPSIS
print("\nTop 5 Candidats Recommandés par le Système Multi-Agents (TOPSIS) :")
cols_top = ["nom", "TOPSIS_Rank", "TOPSIS_Score", "WSM_Rank", "score_technique", "score_rh", "score_experience", "score_ethique", "HiringDecision"]
print(ranking_df[cols_top].head(5).to_string(index=True))

# 5. Analyse de Sensibilité
print("\n5. Analyse de Sensibilité (Impact d'une variation du poids Technique de -30% à +30%)...")
sensitivity = SensitivityAnalysis(ranking_df[criteria], base_weights=ahp_weights)
df_sensitivity = sensitivity.evaluate_criterion_variation(
    target_criterion="score_technique",
    variations=[-0.30, -0.15, 0.0, 0.15, 0.30]
)

# Suivi du Top 5 initial
top_5_ids = ranking_df.index[:5].tolist()
df_sens_top5 = df_sensitivity[df_sensitivity["candidat_id"].isin(top_5_ids)]

# --- GÉNÉRATION DES FIGURES ---
print("\n6. Génération des graphiques (reports/figures/)...")

# Figure 14 : Poids AHP et validation de cohérence
fig, ax = plt.subplots(figsize=(8, 4.5))
weights_list = [ahp_weights[c] * 100 for c in criteria]
colors_ahp = ["#2980b9", "#27ae60", "#e67e22", "#8e44ad"]
bars = ax.bar(criteria_labels, weights_list, color=colors_ahp, edgecolor="black", width=0.5)
ax.set_title(f"Pondération des Critères par AHP de Saaty\n[Ratio de Cohérence CR = {ahp_summary['consistency_ratio']:.3f} < 0.10 (Validé ✅)]", fontsize=12, fontweight="bold")
ax.set_ylabel("Poids du Critère (%)")
ax.set_ylim(0, 55)
for bar in bars:
    y = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, y + 1.2, f"{y:.1f}%", ha="center", fontweight="bold")
plt.xticks(rotation=12)
plt.savefig(os.path.join(FIGURES_DIR, "fig14_ahp_criteria_weights.png"), dpi=200)
plt.close()

# Figure 15 : Comparaison du Classement WSM vs TOPSIS
fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(
    data=ranking_df,
    x="WSM_Rank", y="TOPSIS_Rank",
    hue="TOPSIS_Score",
    palette="viridis",
    s=50,
    ax=ax
)
ax.plot([1, len(ranking_df)], [1, len(ranking_df)], "r--", linewidth=1.5, label="Concordance parfaite (WSM = TOPSIS)")
ax.set_title(f"Comparaison des Classements : Baseline WSM vs TOPSIS\n[Corrélation de Spearman : ρ = {rho:.3f} | {inversions_count} réajustements de rang]", fontsize=12, fontweight="bold")
ax.set_xlabel("Rang selon la Baseline Simple (WSM)")
ax.set_ylabel("Rang selon le Modèle Avancé (TOPSIS)")
ax.legend(loc="upper left")
plt.savefig(os.path.join(FIGURES_DIR, "fig15_wsm_vs_topsis_ranking.png"), dpi=200)
plt.close()

# Figure 16 : Profil Multicritère du Top 10 Candidats (Heatmap)
fig, ax = plt.subplots(figsize=(10, 5))
top10_df = ranking_df.head(10)[["score_technique", "score_rh", "score_experience", "score_ethique"]]
top10_df.index = [f"#{r} {ranking_df.loc[idx, 'nom']}" for r, idx in zip(ranking_df.head(10)["TOPSIS_Rank"], top10_df.index)]
sns.heatmap(top10_df, annot=True, fmt=".1f", cmap="YlGnBu", cbar_kws={'label': 'Score / 100'}, ax=ax)
ax.set_title("Évaluation des 4 Agents pour le Top 10 des Candidats Recommandés", fontsize=12, fontweight="bold")
ax.set_xticklabels(criteria_labels, rotation=10)
plt.savefig(os.path.join(FIGURES_DIR, "fig16_agents_scores_heatmap.png"), dpi=200)
plt.close()

# Figure 17 : Analyse de Sensibilité du Top 5
fig, ax = plt.subplots(figsize=(9, 5))
for cand_id in top_5_ids:
    sub = df_sens_top5[df_sens_top5["candidat_id"] == cand_id]
    ax.plot(sub["variation_pct"], sub["rank"], "o-", linewidth=2, label=f"Candidat #{cand_id}")

ax.set_title("Analyse de Sensibilité : Stabilité du Rang du Top 5 lors de la Variation du Poids Technique", fontsize=12, fontweight="bold")
ax.set_xlabel("Variation du Poids Technique par rapport à la base (%)")
ax.set_ylabel("Rang TOPSIS (1 = Premier)")
ax.invert_yaxis() # Rang 1 en haut
ax.legend(title="Candidats Top 5")
plt.savefig(os.path.join(FIGURES_DIR, "fig17_mcdm_sensitivity_analysis.png"), dpi=200)
plt.close()

# 7. Sauvegarde du tableau décisionnel final
output_csv = os.path.join(PROCESSED_DIR, "candidates_mcdm_final_ranking.csv")
ranking_df.to_csv(output_csv, index=True)
print(f"\nClassement décisionnel complet sauvegardé dans : {output_csv}")
print("Phase 4 (Système Multi-Agents & MCDM) terminée avec succès !")
