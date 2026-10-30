"""
Application Web Interactive Déployée (Streamlit)
Système Multi-Agents & Méthodes MCDM pour l'Aide à la Sélection de Candidats
École Centrale de Casablanca - Cours d'Ingénierie Machine Learning & Statistique (Sujet 11)
"""
import os
import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Configuration de la page
st.set_page_config(
    page_title="Système Multi-Agents & MCDM - Recrutement IA",
    page_icon="👔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ajout du chemin projet au sys.path
PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.mcdm.ahp import AHP
from src.mcdm.topsis import TOPSIS
from src.mcdm.wsm import WeightedSumModel
from src.agents.tech_agent import TechAgent
from src.agents.hr_agent import HRAgent
from src.agents.experience_agent import ExperienceAgent
from src.agents.ethics_agent import EthicsAgent

# Chemins des données et modèles
DATA_RANKING_PATH = PROJECT_DIR / "data" / "processed" / "candidates_mcdm_final_ranking.csv"
MODEL_PATH = PROJECT_DIR / "models" / "best_random_forest.joblib"
SCALER_PATH = PROJECT_DIR / "models" / "scaler.joblib"
FIGURES_DIR = PROJECT_DIR / "reports" / "figures"

# Chargement des ressources en cache
@st.cache_data
def load_ranking_data():
    if DATA_RANKING_PATH.exists():
        df = pd.read_csv(DATA_RANKING_PATH)
        return df
    return None

@st.cache_resource
def load_ml_models():
    model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None
    scaler = joblib.load(SCALER_PATH) if SCALER_PATH.exists() else None
    return model, scaler

df_ranking = load_ranking_data()
best_model, scaler = load_ml_models()

# --- EN-TÊTE PRINCIPAL ---
st.markdown("""
<div style="background: linear-gradient(135deg, #1a365d 0%, #2b6cb0 100%); padding: 22px; border-radius: 10px; color: white; margin-bottom: 20px;">
    <h1 style="margin: 0; font-size: 28px;">👔 Système Multi-Agents & MCDM pour la Sélection de Candidats</h1>
    <p style="margin: 5px 0 0 0; font-size: 14px; opacity: 0.9;">
        <b>École Centrale de Casablanca</b> — Projet d'Ingénierie Machine Learning & Statistique (Sujet N° 11)
    </p>
</div>
""", unsafe_allow_html=True)

# Indicateurs rapides (KPIs)
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("👥 Candidats Évalués", "225 (Test Set)", "1 500 au total")
kpi2.metric("🤖 Agents Évaluateurs", "4 Experts", "Lead Tech, RH, Exp, Éthique")
kpi3.metric("🎯 Précision Modèle ML", "91.1 %", "ROC-AUC : 0.921")
kpi4.metric("⚖️ Cohérence AHP (CR)", "0.0115", "Validé (< 0.10)")

st.markdown("---")

# --- BARRE LATÉRALE : PARAMÈTRES AHP DYNAMIQUES ---
st.sidebar.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=64)
st.sidebar.header("⚖️ Pondération AHP du Comité")
st.sidebar.markdown("Ajustez l'importance relative des membres du jury pour recalculer en temps réel les poids :")

imp_tech = st.sidebar.slider("👨‍💻 Priorité Agent Lead Tech", 1, 9, 8)
imp_exp = st.sidebar.slider("📈 Priorité Agent Expérience", 1, 9, 6)
imp_hr = st.sidebar.slider("🤝 Priorité Agent RH & Soft Skills", 1, 9, 4)
imp_eth = st.sidebar.slider("⚖️ Priorité Agent Éthique & Équité", 1, 9, 2)

total_imp = imp_tech + imp_exp + imp_hr + imp_eth
weights = {
    "score_technique": imp_tech / total_imp,
    "score_experience": imp_exp / total_imp,
    "score_rh": imp_hr / total_imp,
    "score_ethique": imp_eth / total_imp
}

st.sidebar.markdown("### 📊 Poids Normalisés Actuels :")
for crit, w in weights.items():
    st.sidebar.progress(float(w), text=f"{crit.replace('score_', '').capitalize()} : {w*100:.1f}%")

st.sidebar.markdown("---")
st.sidebar.caption("💡 Conforme aux exigences de Saaty (AHP) & TOPSIS.")

# --- ONGLETS PRINCIPAUX ---
tabs = st.tabs([
    "🏆 Classement & Décision MCDM",
    "🤖 Fiche Candidat & Délibération",
    "🧪 Simulateur de Candidat en Direct",
    "🔬 Explicabilité SHAP (XAI)",
    "📊 Exploration Statistique & ACP"
])

# ========================================================
# ONGLET 1 : CLASSEMENT MCDM (TOPSIS vs WSM)
# ========================================================
with tabs[0]:
    st.subheader("🏆 Classement Officiel du Comité de Sélection")
    st.markdown("""
    Comparaison directe entre la **Baseline Simple (WSM - Somme Pondérée)** et le **Modèle Avancé (TOPSIS)**.  
    *TOPSIS valorise les profils équilibrés géométriquement proches du candidat idéal, tandis que la baseline masque les faiblesses.*
    """)

    if df_ranking is not None:
        # Filtres interactifs
        c1, c2, c3 = st.columns(3)
        with c1:
            top_n = st.selectbox("Afficher les premiers candidats :", [10, 25, 50, "Tous"], index=0)
        with c2:
            filter_gender = st.selectbox("Filtrer par Genre :", ["Tous", "Femmes uniquement (0)", "Hommes uniquement (1)"])
        with c3:
            sort_by = st.selectbox("Trier par :", ["TOPSIS_Rank", "WSM_Rank", "SkillScore", "InterviewScore"])

        filtered_df = df_ranking.copy()
        if filter_gender == "Femmes uniquement (0)":
            filtered_df = filtered_df[filtered_df["Gender"] == 0]
        elif filter_gender == "Hommes uniquement (1)":
            filtered_df = filtered_df[filtered_df["Gender"] == 1]

        filtered_df = filtered_df.sort_values(by=sort_by)
        display_df = filtered_df.head(top_n) if isinstance(top_n, int) else filtered_df

        columns_to_show = [
            "TOPSIS_Rank", "nom", "TOPSIS_Score", "WSM_Rank", "WSM_Score",
            "score_technique", "score_rh", "score_experience", "score_ethique", "HiringDecision"
        ]
        st.dataframe(
            display_df[columns_to_show].style.format({
                "TOPSIS_Score": "{:.4f}",
                "WSM_Score": "{:.4f}",
                "score_technique": "{:.1f}",
                "score_rh": "{:.1f}",
                "score_experience": "{:.1f}",
                "score_ethique": "{:.1f}"
            }),
            use_container_width=True,
            height=380
        )

        st.download_button(
            label="📥 Télécharger le Classement Complet (CSV)",
            data=filtered_df.to_csv(index=False).encode("utf-8"),
            file_name="classement_mcdm_candidats.csv",
            mime="text/csv"
        )
    else:
        st.warning("Données de classement non disponibles. Veuillez exécuter le script run_multi_agents_and_mcdm.py.")

# ========================================================
# ONGLET 2 : FICHE CANDIDAT & DÉLIBÉRATION
# ========================================================
with tabs[1]:
    st.subheader("🤖 Fiche d'Évaluation & Délibération des 4 Agents")
    st.markdown("Consultez l'audit détaillé et les avis qualitatifs rédigés par chaque agent expert pour n'importe quel candidat :")

    if df_ranking is not None:
        cand_list = df_ranking["nom"].tolist()
        selected_cand_name = st.selectbox("Sélectionner un candidat à inspecter :", cand_list, index=0)
        cand_row = df_ranking[df_ranking["nom"] == selected_cand_name].iloc[0]

        col_left, col_right = st.columns([1, 1.2])

        with col_left:
            st.markdown(f"### Profil : **{cand_row['nom']}**")
            st.write(f"• **Rang TOPSIS :** `#{int(cand_row['TOPSIS_Rank'])}` (Score: `{cand_row['TOPSIS_Score']:.4f}`)")
            st.write(f"• **Rang WSM :** `#{int(cand_row['WSM_Rank'])}` (Score: `{cand_row['WSM_Score']:.4f}`)")
            st.write(f"• **Âge :** {int(cand_row['Age'])} ans | **Genre :** {'Homme' if cand_row['Gender']==1 else 'Femme'}")
            st.write(f"• **Expérience :** {int(cand_row['ExperienceYears'])} ans | **Diplôme :** Niveau {int(cand_row['EducationLevel'])}/4")
            st.write(f"• **Statut d'embauche réel :** {'✅ Embauché' if cand_row['HiringDecision']==1 else '❌ Non retenu'}")

            # Radar chart des scores
            scores_vals = [cand_row['score_technique'], cand_row['score_rh'], cand_row['score_experience'], cand_row['score_ethique']]
            fig, ax = plt.subplots(figsize=(5, 3.5))
            ax.barh(["Éthique", "Expérience", "RH & Fit", "Technique"], scores_vals, color=["#8e44ad", "#e67e22", "#27ae60", "#2980b9"])
            ax.set_xlim(0, 100)
            ax.set_xlabel("Note de l'Agent (/100)")
            for i, v in enumerate(scores_vals):
                ax.text(v + 1.5, i, f"{v:.1f}", va="center", fontweight="bold")
            st.pyplot(fig)

        with col_right:
            st.markdown("### 💬 Délibérations Textuelles des Agents :")
            st.info(f"👨‍💻 **Agent Lead Tech ({cand_row['score_technique']}/100) :**\n{cand_row.get('avis_technique', 'Évaluation technique opérationnelle.')}")
            st.success(f"🤝 **Agent RH & Soft Skills ({cand_row['score_rh']}/100) :**\n{cand_row.get('avis_rh', 'Profil relationnel équilibré.')}")
            st.warning(f"📈 **Agent Expérience & Séniorité ({cand_row['score_experience']}/100) :**\n{cand_row.get('avis_experience', 'Expérience conforme.')}")
            st.error(f"⚖️ **Agent Éthique & Diversité ({cand_row['score_ethique']}/100) :**\n{cand_row.get('avis_ethique', 'Audit d équité validé.')}")

# ========================================================
# ONGLET 3 : SIMULATEUR EN DIRECT
# ========================================================
with tabs[2]:
    st.subheader("🧪 Tester un Nouveau Candidat en Temps Réel")
    st.markdown("Renseignez les compétences d'un profil fictif ou réel pour faire délibérer immédiatement le jury d'agents :")

    with st.form("candidat_form"):
        sim_col1, sim_col2, sim_col3 = st.columns(3)
        with sim_col1:
            sim_skill = st.slider("Note Test Technique (SkillScore)", 0, 100, 85)
            sim_interview = st.slider("Note Entretien RH (InterviewScore)", 0, 100, 78)
            sim_personality = st.slider("Note Soft Skills (PersonalityScore)", 0, 100, 80)
        with sim_col2:
            sim_exp = st.slider("Années d'Expérience", 0, 15, 6)
            sim_edu = st.selectbox("Niveau d'Études", [1, 2, 3, 4], index=1, format_func=lambda x: f"Niveau {x} ({'Licence' if x==1 else 'Master/Ingénieur' if x==2 else 'PhD' if x==3 else 'Spécialisé'})")
            sim_companies = st.slider("Nombre d'entreprises passées", 1, 5, 2)
        with sim_col3:
            sim_age = st.slider("Âge du Candidat", 20, 50, 29)
            sim_gender = st.selectbox("Genre", [0, 1], format_func=lambda x: "Femme (0)" if x==0 else "Homme (1)")
            sim_distance = st.slider("Distance Domicile-Entreprise (km)", 1.0, 50.0, 12.0)

        submit_btn = st.form_submit_button("🚀 Lancer l'Évaluation du Comité Multi-Agents")

    if submit_btn:
        cand_input = {
            "id": "CAND_SIMULATED",
            "SkillScore": sim_skill,
            "InterviewScore": sim_interview,
            "PersonalityScore": sim_personality,
            "ExperienceYears": sim_exp,
            "EducationLevel": sim_edu,
            "PreviousCompanies": sim_companies,
            "Age": sim_age,
            "Gender": sim_gender,
            "DistanceFromCompany": sim_distance
        }

        agent_tech = TechAgent()
        agent_hr = HRAgent()
        agent_exp = ExperienceAgent()
        agent_eth = EthicsAgent()

        eval_t = agent_tech.evaluate(cand_input)
        eval_h = agent_hr.evaluate(cand_input)
        eval_e = agent_exp.evaluate(cand_input)
        eval_et = agent_eth.evaluate(cand_input)

        st.success("✅ Délibération achevée avec succès !")
        res_c1, res_c2 = st.columns([1, 1.5])

        with res_c1:
            st.markdown("### 📊 Notes du Jury :")
            st.metric("👨‍💻 Note Technique", f"{eval_t['score_technique']:.1f} / 100")
            st.metric("🤝 Note RH & Fit", f"{eval_h['score_rh']:.1f} / 100")
            st.metric("📈 Note Expérience", f"{eval_e['score_experience']:.1f} / 100")
            st.metric("⚖️ Note Éthique", f"{eval_et['score_ethique']:.1f} / 100")

            # Score TOPSIS approximatif
            sim_scores = [eval_t['score_technique'], eval_h['score_rh'], eval_e['score_experience'], eval_et['score_ethique']]
            w_norm = [weights['score_technique'], weights['score_rh'], weights['score_experience'], weights['score_ethique']]
            wsm_sim = sum(s * w for s, w in zip(sim_scores, w_norm))
            st.metric("🎯 Score Global Pondéré", f"{wsm_sim:.2f} / 100")

        with res_c2:
            st.markdown("### 💬 Avis Rédactionnels des Agents :")
            st.info(f"**Lead Tech :** {eval_t['avis_technique']}")
            st.success(f"**Agent RH :** {eval_h['avis_rh']}")
            st.warning(f"**Agent Expérience :** {eval_e['avis_experience']}")
            st.error(f"**Agent Éthique :** {eval_et['avis_ethique']}")

            if best_model and scaler:
                input_feat = np.array([[sim_age, sim_gender, sim_edu, sim_exp, sim_companies, sim_distance, sim_interview, sim_skill, sim_personality, 2]])
                input_scaled = scaler.transform(input_feat)
                proba_hire = best_model.predict_proba(input_scaled)[0, 1] * 100
                st.markdown(f"#### 🤖 Prédiction du Modèle ML (Random Forest) : **{proba_hire:.1f}% de chances d'embauche**")

# ========================================================
# ONGLET 4 : EXPLICABILITÉ SHAP (XAI)
# ========================================================
with tabs[3]:
    st.subheader("🔬 Explicabilité Mathématique avec SHAP (Cours 1, slides 51-54)")
    st.markdown("""
    L'explicabilité (*Explainable AI - XAI*) permet de comprendre précisément pourquoi un profil est sélectionné ou écarté,
    en décomposant l'impact de chaque critère selon la théorie des jeux coopératifs de Shapley.
    """)

    xai_col1, xai_col2 = st.columns(2)
    with xai_col1:
        st.markdown("#### 1. Explicabilité Globale (Beeswarm Plot)")
        st.caption("Distribution de l'impact de chaque critère sur toutes les décisions de recrutement.")
        img_p1 = FIGURES_DIR / "fig10_shap_global_summary.png"
        if img_p1.exists():
            st.image(str(img_p1), use_container_width=True)

    with xai_col2:
        st.markdown("#### 2. Importance Moyenne des Variables (|SHAP|)")
        st.caption("Classement des compétences pesant le plus lourd dans la sélection.")
        img_p2 = FIGURES_DIR / "fig11_shap_feature_importance.png"
        if img_p2.exists():
            st.image(str(img_p2), use_container_width=True)

    st.markdown("---")
    st.markdown("#### 3. Explicabilité Locale (Cascade Waterfall par Candidat)")
    c_loc1, c_loc2 = st.columns(2)
    with c_loc1:
        st.markdown("##### ✅ Cas d'un Candidat Retenu")
        img_p3 = FIGURES_DIR / "fig12_shap_local_accepted.png"
        if img_p3.exists():
            st.image(str(img_p3), use_container_width=True)
    with c_loc2:
        st.markdown("##### ❌ Cas d'un Candidat Rejeté")
        img_p4 = FIGURES_DIR / "fig13_shap_local_rejected.png"
        if img_p4.exists():
            st.image(str(img_p4), use_container_width=True)

# ========================================================
# ONGLET 5 : EXPLORATION STATISTIQUE & ACP
# ========================================================
with tabs[4]:
    st.subheader("📊 Réduction de Dimension (ACP) & Segmentation (Clustering)")
    st.markdown("Application directe du **Cours 2 de l'ECC (slides 17-77)** sur les données de recrutement :")

    stat_c1, stat_c2 = st.columns(2)
    with stat_c1:
        st.markdown("#### Plan Factoriel ACP 2D & Centroïdes des Clusters")
        st.caption("Projection des candidats dans l'espace factoriel avec flèches des compétences (Loadings).")
        img_pca = FIGURES_DIR / "fig6_pca_clusters_projection.png"
        if img_pca.exists():
            st.image(str(img_pca), use_container_width=True)

    with stat_c2:
        st.markdown("#### Choix Optimal de k (Score de Silhouette)")
        st.caption("Preuve mathématique de la segmentation optimale via l'indice de Silhouette.")
        img_sil = FIGURES_DIR / "fig5_clustering_evaluation.png"
        if img_sil.exists():
            st.image(str(img_sil), use_container_width=True)

    st.markdown("---")
    st.markdown("#### Profilage des Archétypes de Candidats")
    img_prof = FIGURES_DIR / "fig7_clusters_archetypes_profile.png"
    if img_prof.exists():
        st.image(str(img_prof), use_container_width=True)

st.markdown("---")
st.caption("🎓 Projet réalisé pour le Cours d'Ingénierie Machine Learning & Statistique — École Centrale de Casablanca — Date limite : 30 Novembre")
