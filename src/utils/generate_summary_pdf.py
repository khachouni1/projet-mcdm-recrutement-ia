"""
Mise à jour finale du PDF synthétique récapitulatif des étapes du projet
Toutes les phases sont complétées (EDA, ACP, ML/SHAP, Multi-Agents/MCDM, Déploiement Streamlit)
"""
import os
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm

PDF_PATH = r"C:\Users\HP\.gemini\antigravity\scratch\PROJET MACHINE LEARNING\Etapes_du_Projet.pdf"

doc = SimpleDocTemplate(
    PDF_PATH,
    pagesize=A4,
    rightMargin=1.3*cm,
    leftMargin=1.3*cm,
    topMargin=1.1*cm,
    bottomMargin=1.1*cm
)

styles = getSampleStyleSheet()

title_style = ParagraphStyle(
    "DocTitle",
    parent=styles["Normal"],
    fontName="Helvetica-Bold",
    fontSize=14.5,
    leading=18,
    textColor=colors.HexColor("#1a365d"),
    alignment=1,
    spaceAfter=2
)

subtitle_style = ParagraphStyle(
    "DocSubtitle",
    parent=styles["Normal"],
    fontName="Helvetica",
    fontSize=8.5,
    leading=11,
    textColor=colors.HexColor("#4a5568"),
    alignment=1,
    spaceAfter=6
)

section_title_style = ParagraphStyle(
    "SectionTitle",
    parent=styles["Normal"],
    fontName="Helvetica-Bold",
    fontSize=9.5,
    leading=12.5,
    textColor=colors.HexColor("#2b6cb0"),
    spaceBefore=3.5,
    spaceAfter=1.5
)

bullet_style = ParagraphStyle(
    "BulletText",
    parent=styles["Normal"],
    fontName="Helvetica",
    fontSize=7.8,
    leading=10.2,
    textColor=colors.HexColor("#2d3748"),
    leftIndent=8,
    firstLineIndent=-6,
    spaceAfter=1
)

story = []

# En-tête
story.append(Paragraph("PROJET MACHINE LEARNING & STATISTIQUE (ECC)", title_style))
story.append(Paragraph("<b>Sujet 11 :</b> Système Multi-Agents & Méthodes MCDM pour la Sélection de Candidats<br/><i>Feuille de Route & Synthèse Finale des Travaux Réalisés</i>", subtitle_style))
story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2b6cb0"), spaceAfter=4))

# 1. Source
story.append(Paragraph("1. Source & Données Brutes (Kaggle / UC Berkeley Haas)", section_title_style))
story.append(Paragraph("• <b>Jeu de données :</b> <i>Predicting Hiring Decisions in Recruitment Data</i> (Auteur : Rabie El Kharoua).", bullet_style))
story.append(Paragraph("• <b>Lien public :</b> https://www.kaggle.com/datasets/rabieelkharoua/predicting-hiring-decisions-in-recruitment-data", bullet_style))
story.append(Paragraph("• <b>Volumétrie :</b> 1 500 candidats réels, 11 variables, 0 valeur manquante (enregistré dans <code>data/raw/</code>).", bullet_style))

# 2. EDA & Holdout
story.append(Paragraph("2. EDA & Découpage Three-Way Holdout (Anti-Fuite de Données) [FAIT ✅]", section_title_style))
story.append(Paragraph("• <b>Déséquilibre de classe :</b> 31% embauchés (465) vs 69% rejetés (1 035) $\\rightarrow$ justifie l'emploi de SMOTE.", bullet_style))
story.append(Paragraph("• <b>Three-Way Holdout (Cours 1, slides 84-86) :</b> Découpage strict AVANT pré-traitement : Train (70% - 1 049), Val (15% - 226), Test (15% - 225).", bullet_style))
story.append(Paragraph("• <b>Figures produites (4) :</b> Enregistrées dans <code>reports/figures/fig1_ à fig4_</code>.", bullet_style))

# 3. Phase 2 : ACP & Clustering
story.append(Paragraph("3. Analyse Factorielle (ACP) & Clustering Non Supervisé (Cours 2) [FAIT ✅]", section_title_style))
story.append(Paragraph("• <b>ACP 2D (Cours 2, slides 17-37) :</b> PC1 capture l'excellence technique (+0.55 <i>SkillScore</i>) ; PC2 capture la maturité relationnelle (+0.54 <i>PersonalityScore</i>, +0.46 <i>Age</i>).", bullet_style))
story.append(Paragraph("• <b>Validation du nombre de clusters :</b> Test de $k=2$ à $k=6$, optimal à $k=2$ validé par le <b>Score de Silhouette</b> (slide 68).", bullet_style))
story.append(Paragraph("• <b>Figures produites (3) :</b> <code>fig5_clustering_evaluation.png</code>, <code>fig6_pca_clusters_projection.png</code>, <code>fig7_clusters_archetypes_profile.png</code>.", bullet_style))

# 4. Phase 3 : ML Supervisé & SHAP
story.append(Paragraph("4. Machine Learning Supervisé, SMOTE & Explicabilité SHAP (Cours 1) [FAIT ✅]", section_title_style))
story.append(Paragraph("• <b>Traitement du déséquilibre (Cours 1, slides 57-59) :</b> SMOTE appliqué <b>uniquement</b> sur Train (équilibrage 724/724).", bullet_style))
story.append(Paragraph("• <b>Performances sur Test :</b> Random Forest atteint <b>91.1% d'Accuracy</b>, <b>85.7% de Précision/Rappel</b> et <b>ROC-AUC = 0.921</b> (surpassant la Baseline logistique à 82.7%).", bullet_style))
story.append(Paragraph("• <b>Explicabilité SHAP (slides 51-54) :</b> Importance globale dominée par <i>InterviewScore</i> et <i>SkillScore</i> ; cascades locales individuelles (waterfall).", bullet_style))
story.append(Paragraph("• <b>Figures produites (6) :</b> <code>fig8_ à fig13_</code>.", bullet_style))

# 5. Phase 4 : Système Multi-Agents & MCDM
story.append(Paragraph("5. Système Multi-Agents & Moteur Décisionnel MCDM (Sujet 11) [FAIT ✅]", section_title_style))
story.append(Paragraph("• <b>Comité d'Évaluation :</b> Délibération simultanée de 4 agents (Lead Tech, RH & Fit, Expérience & Séniorité, Éthique & Équité).", bullet_style))
story.append(Paragraph("• <b>Pondération AHP (Saaty) :</b> Dérivation des poids : Technique (46.6%), Expérience (27.7%), RH (16.1%), Éthique (9.6%).", bullet_style))
story.append(Paragraph("• <b>Validation de cohérence :</b> Ratio de cohérence <b>$CR = 0.0115 < 0.10$</b> (Jugement mathématiquement rigoureux et cohérent).", bullet_style))
story.append(Paragraph("• <b>Confrontation Baseline WSM vs TOPSIS :</b> $\\rho = 0.895$. TOPSIS réajuste 94.7% des rangs en pénalisant les profils déséquilibrés et en récompensant la proximité géométrique avec le profil idéal.", bullet_style))
story.append(Paragraph("• <b>Analyse de Sensibilité :</b> Stabilité prouvée du Top 5 lors de variations de pondération technique de -30% à +30%.", bullet_style))
story.append(Paragraph("• <b>Figures produites (4) :</b> <code>fig14_ à fig17_</code>.", bullet_style))

# 6. Phase 5 : Déploiement Interactif
story.append(Paragraph("6. Déploiement Interactif & Démonstration Web (Streamlit) [FAIT ✅]", section_title_style))
story.append(Paragraph("• <b>Application opérationnelle (<code>app/app_streamlit.py</code>) :</b> Hébergée en local à <code>http://localhost:8501</code>.", bullet_style))
story.append(Paragraph("• <b>Onglet 1 (Classement MCDM) :</b> Tableau interactif filtrable par genre/scores avec téléchargement CSV.", bullet_style))
story.append(Paragraph("• <b>Onglet 2 (Fiche Candidat) :</b> Délibérations textuelles complètes des 4 agents + radar de compétences.", bullet_style))
story.append(Paragraph("• <b>Onglet 3 (Simulateur Direct) :</b> Formulaire interactif pour évaluer instantanément un nouveau profil avec calcul TOPSIS et probabilité ML.", bullet_style))
story.append(Paragraph("• <b>Onglets 4 & 5 (XAI & ACP) :</b> Galerie interactive des graphiques factoriels ACP et d'explicabilité SHAP.", bullet_style))

doc.build(story)
print("PDF mis à jour avec le déploiement Streamlit !")
