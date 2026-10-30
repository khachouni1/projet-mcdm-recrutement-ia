"""
Script complet d'analyse exploratoire des données (EDA)
et de découpage Three-Way Holdout (Train / Val / Test)
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# Définition des chemins
PROJECT_DIR = r"C:\Users\HP\.gemini\antigravity\scratch\PROJET MACHINE LEARNING"
RAW_DATA_PATH = os.path.join(PROJECT_DIR, "data", "raw", "recruitment_data.csv")
PROCESSED_DIR = os.path.join(PROJECT_DIR, "data", "processed")
FIGURES_DIR = os.path.join(PROJECT_DIR, "reports", "figures")
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# Configuration graphique
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"font.size": 11, "figure.autolayout": True})

print("1. Chargement du dataset...")
df = pd.read_csv(RAW_DATA_PATH)
print(f"Dimensions initiales : {df.shape[0]} lignes, {df.shape[1]} colonnes")

# --- EDA & FIGURES ---
print("\n2. Génération des visualisations graphiques (reports/figures/)...")

# Figure 1 : Distribution de la variable cible (Déséquilibre de classe)
fig, ax = plt.subplots(figsize=(7, 4.5))
counts = df["HiringDecision"].value_counts()
labels = ["Rejeté (0)", "Embauché (1)"]
colors = ["#e74c3c", "#2ecc71"]
bars = ax.bar(labels, counts, color=colors, width=0.5, edgecolor="black", linewidth=1.2)
ax.set_title("Distribution de la Variable Cible (HiringDecision)\n[Déséquilibre : 69% vs 31%]", fontsize=13, fontweight="bold")
ax.set_ylabel("Nombre de Candidats", fontsize=11)
for bar in bars:
    yval = bar.get_height()
    pct = (yval / len(df)) * 100
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 20, f"{yval} ({pct:.1f}%)", ha="center", va="bottom", fontweight="bold")
plt.savefig(os.path.join(FIGURES_DIR, "fig1_target_distribution.png"), dpi=200)
plt.close()

# Figure 2 : Matrice de corrélation
fig, ax = plt.subplots(figsize=(10, 8))
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", vmin=-1, vmax=1, mask=mask, cbar_kws={'shrink': .8}, ax=ax)
ax.set_title("Matrice de Corrélation Linéaire des Variables", fontsize=14, fontweight="bold")
plt.savefig(os.path.join(FIGURES_DIR, "fig2_correlation_matrix.png"), dpi=200)
plt.close()

# Figure 3 : Comparaison des scores selon la décision d'embauche
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
score_cols = [("SkillScore", "Score Technique"), ("InterviewScore", "Score Entretien"), ("PersonalityScore", "Score Personnalité")]
for i, (col, title) in enumerate(score_cols):
    sns.boxplot(data=df, x="HiringDecision", y=col, ax=axes[i], palette=["#e74c3c", "#2ecc71"])
    axes[i].set_title(f"{title} vs Décision", fontweight="bold")
    axes[i].set_xticklabels(["Rejeté (0)", "Embauché (1)"])
    axes[i].set_xlabel("")
plt.savefig(os.path.join(FIGURES_DIR, "fig3_scores_by_decision.png"), dpi=200)
plt.close()

# Figure 4 : Audit Équité (Genre & Âge)
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
# Taux par genre
gender_df = df.groupby("Gender")["HiringDecision"].mean().reset_index()
gender_df["GenderLabel"] = gender_df["Gender"].map({0: "Femme (0)", 1: "Homme (1)"})
axes[0].bar(gender_df["GenderLabel"], gender_df["HiringDecision"] * 100, color=["#9b59b6", "#3498db"], width=0.4, edgecolor="black")
axes[0].set_title("Taux d'Embauche par Genre (Parité)", fontweight="bold")
axes[0].set_ylabel("Taux d'Embauche (%)")
axes[0].set_ylim(0, 50)
for idx, row in gender_df.iterrows():
    axes[0].text(idx, row["HiringDecision"] * 100 + 1.5, f"{row['HiringDecision']*100:.1f}%", ha="center", fontweight="bold")

# Âge vs Expérience
sns.scatterplot(data=df, x="Age", y="ExperienceYears", hue="HiringDecision", palette=["#e74c3c", "#2ecc71"], alpha=0.7, ax=axes[1])
axes[1].set_title("Âge vs Années d'Expérience selon l'Embauche", fontweight="bold")
plt.savefig(os.path.join(FIGURES_DIR, "fig4_demographics_fairness.png"), dpi=200)
plt.close()

print("Figures sauvegardées avec succès !")

# --- THREE-WAY HOLDOUT SPLIT ---
print("\n3. Découpage Three-Way Holdout (Train 70% / Val 15% / Test 15%)...")
# Séparation Test (15%) et Train+Val (85%)
train_val_df, test_df = train_test_split(
    df,
    test_size=0.15,
    random_state=42,
    stratify=df["HiringDecision"]
)

# Séparation Train (70/85 = 82.35%) et Val (15/85 = 17.65%)
val_ratio_in_train_val = 0.15 / 0.85
train_df, val_df = train_test_split(
    train_val_df,
    test_size=val_ratio_in_train_val,
    random_state=42,
    stratify=train_val_df["HiringDecision"]
)

print(f"Jeu d'entraînement (Train) : {train_df.shape[0]} candidats ({train_df.shape[0]/len(df)*100:.1f}%)")
print(f"Jeu de validation (Val)    : {val_df.shape[0]} candidats ({val_df.shape[0]/len(df)*100:.1f}%)")
print(f"Jeu de test (Test)         : {test_df.shape[0]} candidats ({test_df.shape[0]/len(df)*100:.1f}%)")

# Sauvegarde des fichiers CSV
train_df.to_csv(os.path.join(PROCESSED_DIR, "train.csv"), index=False)
val_df.to_csv(os.path.join(PROCESSED_DIR, "val.csv"), index=False)
test_df.to_csv(os.path.join(PROCESSED_DIR, "test.csv"), index=False)
print("Fichiers train.csv, val.csv et test.csv enregistrés dans data/processed/ !")
