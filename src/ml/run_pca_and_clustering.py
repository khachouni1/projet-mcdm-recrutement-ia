"""
Script d'exécution complet de la Phase 2 :
Analyse en Composantes Principales (ACP) & Clustering Non Supervisé (K-Means & CAH)
Directement aligné sur le Cours 2 de l'ECC (slides 17-77)
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score

# Chemins
PROJECT_DIR = r"C:\Users\HP\.gemini\antigravity\scratch\PROJET MACHINE LEARNING"
PROCESSED_DIR = os.path.join(PROJECT_DIR, "data", "processed")
FIGURES_DIR = os.path.join(PROJECT_DIR, "reports", "figures")
TRAIN_PATH = os.path.join(PROCESSED_DIR, "train.csv")

os.makedirs(FIGURES_DIR, exist_ok=True)
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"font.size": 11, "figure.autolayout": True})

print("1. Chargement du jeu d'entraînement (Train Set)...")
train_df = pd.read_csv(TRAIN_PATH)
print(f"Nombre de candidats dans Train : {train_df.shape[0]}")

# Variables numériques et ordinales retenues pour le profilage
feature_cols = [
    "SkillScore", "InterviewScore", "PersonalityScore",
    "ExperienceYears", "EducationLevel", "PreviousCompanies",
    "Age", "DistanceFromCompany"
]
X = train_df[feature_cols]

# 2. Standardisation des variables (Cours 2, slide 20)
print("\n2. Standardisation des données (StandardScaler)...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Analyse en Composantes Principales (ACP) (Cours 2, slides 17-37)
print("3. Calcul de l'ACP (Recherche des axes principaux)...")
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
var_pc1 = pca.explained_variance_ratio_[0] * 100
var_pc2 = pca.explained_variance_ratio_[1] * 100
total_var = var_pc1 + var_pc2

print(f"Variance expliquée Axe 1 (PC1) : {var_pc1:.2f}%")
print(f"Variance expliquée Axe 2 (PC2) : {var_pc2:.2f}%")
print(f"Inertie cumulée sur 2 axes     : {total_var:.2f}%")

# Matrice des contributions / Loadings
loadings = pd.DataFrame(
    pca.components_.T,
    columns=["PC1", "PC2"],
    index=feature_cols
)
print("\nContributions des variables aux axes principaux (Loadings) :")
print(loadings.round(3))

# 4. Clustering Non Supervisé & Choix optimal de k (Cours 2, slides 46 & 68-74)
print("\n4. Évaluation du nombre optimal de clusters k (Silhouette & Coude)...")
k_values = list(range(2, 7))
inertias = []
silhouettes = []

for k in k_values:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    sil = silhouette_score(X_scaled, labels)
    silhouettes.append(sil)
    print(f"k = {k} -> Silhouette = {sil:.4f}, Inertie intra-classe = {km.inertia_:.1f}")

best_k = k_values[np.argmax(silhouettes)]
print(f"\n=> Nombre optimal de clusters selon le Score de Silhouette : k = {best_k}")

# 5. Application de K-Means et CAH avec k optimal
print(f"\n5. Application de K-Means (k={best_k}) et CAH (k={best_k})...")
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
train_df["Cluster_KMeans"] = kmeans.fit_predict(X_scaled)

cah = AgglomerativeClustering(n_clusters=best_k)
train_df["Cluster_CAH"] = cah.fit_predict(X_scaled)

train_df["PC1"] = X_pca[:, 0]
train_df["PC2"] = X_pca[:, 1]

# 6. Profilage métier des clusters obtenus
print("\n6. Profilage moyen des clusters K-Means :")
cluster_summary = train_df.groupby("Cluster_KMeans")[feature_cols + ["HiringDecision"]].mean()
print(cluster_summary.round(2))

# --- GÉNÉRATION DES FIGURES ---
print("\n7. Génération des graphiques (reports/figures/)...")

# Figure 5 : Analyse de la Silhouette et du Coude
fig, ax1 = plt.subplots(figsize=(8, 4.5))
color_inert = "#2980b9"
color_sil = "#e67e22"

ax1.plot(k_values, inertias, "o-", color=color_inert, linewidth=2, label="Inertie intra-classe (Coude)")
ax1.set_xlabel("Nombre de clusters (k)", fontsize=11, fontweight="bold")
ax1.set_ylabel("Inertie (Inertia)", color=color_inert, fontsize=11, fontweight="bold")
ax1.tick_params(axis="y", labelcolor=color_inert)

ax2 = ax1.twinx()
ax2.plot(k_values, silhouettes, "s--", color=color_sil, linewidth=2, label="Score de Silhouette")
ax2.set_ylabel("Score de Silhouette moyen", color=color_sil, fontsize=11, fontweight="bold")
ax2.tick_params(axis="y", labelcolor=color_sil)
ax2.axvline(x=best_k, color="red", linestyle=":", alpha=0.7, label=f"Optimal (k={best_k})")

plt.title("Détermination du Nombre Optimal de Clusters (Coude & Silhouette)\n[Cours 2 ECC - slides 46 & 68]", fontsize=12, fontweight="bold")
fig.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "fig5_clustering_evaluation.png"), dpi=200)
plt.close()

# Figure 6 : Projection ACP 2D avec les Clusters
fig, ax = plt.subplots(figsize=(10, 6.5))
cluster_palette = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12", "#9b59b6"][:best_k]
scatter = sns.scatterplot(
    data=train_df,
    x="PC1", y="PC2",
    hue="Cluster_KMeans",
    palette=cluster_palette,
    alpha=0.75,
    s=45,
    ax=ax
)

# Ajout des centroïdes en 2D
centers_scaled = kmeans.cluster_centers_
centers_pca = pca.transform(centers_scaled)
ax.scatter(
    centers_pca[:, 0], centers_pca[:, 1],
    c="black", marker="X", s=180, edgecolors="white", linewidth=2,
    label="Centroïdes des Clusters"
)

# Cercle des corrélations simplifié (flèches des variables)
scale_arrow = 2.8
for i, feature in enumerate(feature_cols):
    ax.arrow(
        0, 0,
        loadings.loc[feature, "PC1"] * scale_arrow,
        loadings.loc[feature, "PC2"] * scale_arrow,
        color="darkred", alpha=0.6, head_width=0.08, linewidth=1.2
    )
    ax.text(
        loadings.loc[feature, "PC1"] * scale_arrow * 1.15,
        loadings.loc[feature, "PC2"] * scale_arrow * 1.15,
        feature, color="darkred", fontsize=9, fontweight="bold", ha="center"
    )

ax.set_title(f"Plan Factoriel ACP (PC1={var_pc1:.1f}% / PC2={var_pc2:.1f}%)\nProjection des Candidats par Cluster [Cours 2 ECC - slides 28-34]", fontsize=12, fontweight="bold")
ax.set_xlabel(f"Composante Principale 1 ({var_pc1:.1f}% variance)")
ax.set_ylabel(f"Composante Principale 2 ({var_pc2:.1f}% variance)")
ax.legend(title="Archétype / Cluster", loc="upper right")
plt.savefig(os.path.join(FIGURES_DIR, "fig6_pca_clusters_projection.png"), dpi=200)
plt.close()

# Figure 7 : Profilage comparatif des Archétypes
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Taux d'embauche par cluster
hiring_rate = train_df.groupby("Cluster_KMeans")["HiringDecision"].mean() * 100
bars = axes[0].bar([f"Cluster {i}" for i in range(best_k)], hiring_rate, color=cluster_palette, edgecolor="black")
axes[0].set_title("Taux d'Embauche Réel par Cluster (%)", fontweight="bold")
axes[0].set_ylabel("Taux d'Embauche (%)")
axes[0].set_ylim(0, 100)
for bar in bars:
    y = bar.get_height()
    axes[0].text(bar.get_x() + bar.get_width()/2.0, y + 2, f"{y:.1f}%", ha="center", fontweight="bold")

# Radar/bar des scores clés par cluster
scores_df = train_df.groupby("Cluster_KMeans")[["SkillScore", "InterviewScore", "PersonalityScore"]].mean()
scores_df.plot(kind="bar", ax=axes[1], colormap="Set2", edgecolor="black")
axes[1].set_title("Notes Moyennes (Skill, Interview, Soft Skills) par Cluster", fontweight="bold")
axes[1].set_ylabel("Score Moyen (0-100)")
axes[1].set_ylim(0, 100)
axes[1].set_xticklabels([f"Cluster {i}" for i in range(best_k)], rotation=0)

plt.savefig(os.path.join(FIGURES_DIR, "fig7_clusters_archetypes_profile.png"), dpi=200)
plt.close()

# 8. Sauvegarde du dataset enrichi avec clusters et ACP
output_path = os.path.join(PROCESSED_DIR, "train_with_clusters.csv")
train_df.to_csv(output_path, index=False)
print(f"Jeu d'entraînement enrichi sauvegardé dans : {output_path}")
print("Phase 2 (ACP & Clustering) terminée avec succès !")
