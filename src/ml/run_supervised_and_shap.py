"""
Script d'exécution complet de la Phase 3 :
1. Modèles de Classification Supervisée (Régression Logistique, Random Forest, Gradient Boosting)
2. Traitement du déséquilibre par SMOTE (Cours 1 ECC, slides 57-59)
3. Évaluation rigoureuse sur Validation et Test (Three-Way Holdout, slides 84-86)
4. Interprétabilité et Explicabilité globale et locale avec SHAP (Cours 1 ECC, slides 51-54)
"""
import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve
)
from imblearn.over_sampling import SMOTE
import shap

# Chemins
PROJECT_DIR = r"C:\Users\HP\.gemini\antigravity\scratch\PROJET MACHINE LEARNING"
PROCESSED_DIR = os.path.join(PROJECT_DIR, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
FIGURES_DIR = os.path.join(PROJECT_DIR, "reports", "figures")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"font.size": 11, "figure.autolayout": True})

print("1. Chargement des sous-ensembles Train, Val, Test (Three-Way Holdout)...")
train_df = pd.read_csv(os.path.join(PROCESSED_DIR, "train.csv"))
val_df = pd.read_csv(os.path.join(PROCESSED_DIR, "val.csv"))
test_df = pd.read_csv(os.path.join(PROCESSED_DIR, "test.csv"))

features = [
    "Age", "Gender", "EducationLevel", "ExperienceYears",
    "PreviousCompanies", "DistanceFromCompany", "InterviewScore",
    "SkillScore", "PersonalityScore", "RecruitmentStrategy"
]
target = "HiringDecision"

X_train, y_train = train_df[features], train_df[target]
X_val, y_val = val_df[features], val_df[target]
X_test, y_test = test_df[features], test_df[target]

print(f"Train : {X_train.shape[0]} candidats | Val : {X_val.shape[0]} | Test : {X_test.shape[0]}")

# 2. Standardisation des variables (Fit uniquement sur Train pour éviter les fuites)
print("\n2. Standardisation (Fit sur Train uniquement)...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.joblib"))

# 3. Traitement du déséquilibre avec SMOTE (Cours 1, slide 58-59)
print("3. Application de SMOTE sur le Train set uniquement (Évite le Data Leakage)...")
print(f"Distribution initiale Train : 0={sum(y_train==0)}, 1={sum(y_train==1)}")
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
print(f"Distribution après SMOTE : 0={sum(y_train_smote==0)}, 1={sum(y_train_smote==1)}")

# 4. Définition et entraînement des modèles comparés
print("\n4. Entraînement et comparaison des modèles...")
models = {
    "Régression Logistique (Baseline)": LogisticRegression(random_state=42, max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, max_depth=4, learning_rate=0.08, random_state=42)
}

results = []
trained_models = {}

for name, model in models.items():
    # Entraînement sur Train équilibré par SMOTE
    model.fit(X_train_smote, y_train_smote)
    trained_models[name] = model

    # Évaluation sur Validation set
    val_pred = model.predict(X_val_scaled)
    val_proba = model.predict_proba(X_val_scaled)[:, 1]

    # Évaluation sur Test set
    test_pred = model.predict(X_test_scaled)
    test_proba = model.predict_proba(X_test_scaled)[:, 1]

    results.append({
        "Modèle": name,
        "Val Accuracy": accuracy_score(y_val, val_pred),
        "Val F1-Score": f1_score(y_val, val_pred),
        "Val ROC-AUC": roc_auc_score(y_val, val_proba),
        "Test Accuracy": accuracy_score(y_test, test_pred),
        "Test Precision": precision_score(y_test, test_pred),
        "Test Recall": recall_score(y_test, test_pred),
        "Test F1-Score": f1_score(y_test, test_pred),
        "Test ROC-AUC": roc_auc_score(y_test, test_proba)
    })

df_metrics = pd.DataFrame(results)
print("\nTableau Comparatif des Performances :")
print(df_metrics.to_string(index=False))

# Sauvegarde du meilleur modèle (Random Forest)
best_model_name = "Random Forest"
best_model = trained_models[best_model_name]
joblib.dump(best_model, os.path.join(MODELS_DIR, "best_random_forest.joblib"))
print(f"\nMeilleur modèle sauvegardé : {best_model_name} -> models/best_random_forest.joblib")

# --- VISUALISATIONS DES PERFORMANCES ---
print("\n5. Génération des graphiques de performance...")

# Figure 8 : Courbes ROC comparatives
fig, ax = plt.subplots(figsize=(8, 6))
for name, model in trained_models.items():
    y_test_proba = model.predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_test_proba)
    auc_val = roc_auc_score(y_test, y_test_proba)
    ax.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC = {auc_val:.3f})")

ax.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Aléatoire (AUC = 0.500)")
ax.set_title("Comparaison des Courbes ROC sur le Jeu de Test\n[Baseline vs Modèles Avancés]", fontsize=12, fontweight="bold")
ax.set_xlabel("Taux de Faux Positifs (1 - Spécificité)")
ax.set_ylabel("Taux de Vrais Positifs (Sensibilité / Rappel)")
ax.legend(loc="lower right")
plt.savefig(os.path.join(FIGURES_DIR, "fig8_models_roc_comparison.png"), dpi=200)
plt.close()

# Figure 9 : Matrices de Confusion sur le Test Set
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for idx, (name, model) in enumerate(trained_models.items()):
    pred = model.predict(X_test_scaled)
    cm = confusion_matrix(y_test, pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[idx],
                xticklabels=["Rejeté (0)", "Embauché (1)"],
                yticklabels=["Rejeté (0)", "Embauché (1)"])
    axes[idx].set_title(f"Matrice de Confusion\n{name}", fontweight="bold", fontsize=11)
    axes[idx].set_ylabel("Vérité Réelle")
    axes[idx].set_xlabel("Prédiction du Modèle")
plt.savefig(os.path.join(FIGURES_DIR, "fig9_confusion_matrices.png"), dpi=200)
plt.close()

# --- 6. EXPLICABILITÉ AVEC SHAP (Cours 1, slides 51-54) ---
print("\n6. Calcul de l'explicabilité avec SHAP (TreeExplainer)...")
# Conversion en DataFrame avec noms de colonnes pour les graphiques SHAP
X_test_df = pd.DataFrame(X_test_scaled, columns=features)
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test_df)

# Pour les problèmes binaires, shap_values peut être une liste de 2 classes ou un array 3D
if isinstance(shap_values, list):
    shap_vals_class1 = shap_values[1]
elif len(shap_values.shape) == 3:
    shap_vals_class1 = shap_values[:, :, 1]
else:
    shap_vals_class1 = shap_values

# Figure 10 : Explicabilité Globale (SHAP Beeswarm Summary Plot)
plt.figure(figsize=(10, 6.5))
shap.summary_plot(shap_vals_class1, X_test_df, show=False)
plt.title("Explicabilité Globale : Impact de chaque Critère sur la Décision d'Embauche\n[SHAP Summary Plot - Cours 1 ECC slides 51-54]", fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "fig10_shap_global_summary.png"), dpi=200, bbox_inches="tight")
plt.close()

# Figure 11 : Importance Globale Moyenne des Variables (|SHAP|)
plt.figure(figsize=(9, 5))
shap.summary_plot(shap_vals_class1, X_test_df, plot_type="bar", show=False)
plt.title("Classement Global de l'Importance des Critères (|Valeur SHAP Moyenne|)\n[Cours 1 ECC slide 54]", fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "fig11_shap_feature_importance.png"), dpi=200, bbox_inches="tight")
plt.close()

# Figure 12 : Explicabilité Locale pour un Candidat Embauché (Exemple individuel)
# Trouver un candidat du test set bien classé comme embauché (y_test == 1)
idx_accepted = np.where((y_test.values == 1) & (best_model.predict(X_test_scaled) == 1))[0][0]
plt.figure(figsize=(10, 5))
shap.plots.waterfall(
    shap.Explanation(
        values=shap_vals_class1[idx_accepted],
        base_values=explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value,
        data=X_test.iloc[idx_accepted].values,
        feature_names=features
    ),
    show=False
)
plt.title(f"Explicabilité Locale (Candidat Retenu #{test_df.index[idx_accepted]})\n[Justification individuelle - Cours 1 ECC slide 53]", fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "fig12_shap_local_accepted.png"), dpi=200, bbox_inches="tight")
plt.close()

# Figure 13 : Explicabilité Locale pour un Candidat Rejeté
idx_rejected = np.where((y_test.values == 0) & (best_model.predict(X_test_scaled) == 0))[0][0]
plt.figure(figsize=(10, 5))
shap.plots.waterfall(
    shap.Explanation(
        values=shap_vals_class1[idx_rejected],
        base_values=explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value,
        data=X_test.iloc[idx_rejected].values,
        feature_names=features
    ),
    show=False
)
plt.title(f"Explicabilité Locale (Candidat Écarté #{test_df.index[idx_rejected]})\n[Justification du refus - Cours 1 ECC slide 53]", fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "fig13_shap_local_rejected.png"), dpi=200, bbox_inches="tight")
plt.close()

print("Figures SHAP générées avec succès !")
print("\nPhase 3 (Modèles Supervisés, SMOTE & SHAP) terminée avec un succès complet !")
