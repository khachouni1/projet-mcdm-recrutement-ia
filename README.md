# Système Multi-Agents & Méthodes MCDM pour l'Aide à la Sélection de Candidats

**Cours :** Ingénierie Machine Learning et Statistique  
**Établissement :** École Centrale de Casablanca (ECC)  
**Sujet N° 11 :** *Système multi-agents fondé sur les méthodes MCDM pour l'aide à la sélection de candidats en entreprise.*  
**Date limite de remise des livrables :** 30 novembre  

---

## 📌 Présentation du Projet

Ce projet propose une solution innovante et rigoureuse pour l'aide au recrutement en entreprise :
1. **Comité d'Évaluation Multi-Agents :** Chaque candidat est analysé sous plusieurs prismes complémentaires par des agents spécialisés (Agent Lead Tech, Agent RH & Culture Fit, Agent Séniorité & Expérience, Agent Éthique & Diversité).
2. **Moteur Décisionnel MCDM :** 
   - Dérivation mathématique des poids via **AHP (Analytic Hierarchy Process)** avec contrôle de cohérence ($CR < 0.10$).
   - Classement multicritère robuste via **TOPSIS (Technique for Order Preference by Similarity to Ideal Solution)**.
   - Comparaison rigoureuse avec une **Baseline naïve (WSM - Weighted Sum Model)**.
3. **Ancrage Machine Learning & Statistique (Cours ECC) :**
   - Prévention des fuites de données (*Data Leakage*) via **Three-Way Holdout** (Train / Val / Test).
   - Réduction de dimension par **ACP (Analyse en Composantes Principales)** pour visualiser les profils.
   - Segmentation non supervisée des profils par **Clustering K-Means / CAH** avec validation par coefficient de Silhouette.
   - Interprétabilité et explicabilité des décisions (**SHAP & Feature Importance**, explicabilité globale et locale).
4. **Déploiement Interactif :** Application de démonstration en direct sous **Streamlit**.

---

## 📂 Structure du Répertoire

```text
PROJET MACHINE LEARNING/
├── app/
│   └── app_streamlit.py                  # Interface web de démonstration (Streamlit)
├── data/
│   ├── raw/                              # Données brutes (CVs, descriptions de postes)
│   ├── processed/                        # Données préparées et découpées (Train/Val/Test)
│   └── synthetic/                        # Données synthétiques haute fidélité
├── models/                               # Modèles sauvegardés, scalers et matrices AHP
├── notebooks/                            # Expérimentations pas à pas
│   ├── 01_eda_and_preprocessing.ipynb
│   ├── 02_clustering_and_pca.ipynb
│   ├── 03_multi_agents_evaluation.ipynb
│   ├── 04_mcdm_ranking_and_sensitivity.ipynb
│   └── 05_interpretability_shap.ipynb
├── reports/
│   └── figures/                          # Graphiques, visualisations ACP, clusters, SHAP
├── src/
│   ├── agents/                           # Comité d'agents évaluateurs
│   │   ├── base_agent.py
│   │   ├── hr_agent.py
│   │   ├── tech_agent.py
│   │   ├── experience_agent.py
│   │   ├── ethics_agent.py
│   │   └── coordinator_agent.py
│   ├── mcdm/                             # Méthodes multicritères de décision
│   │   ├── ahp.py                        # Méthode AHP (Saaty, CR < 0.10)
│   │   ├── topsis.py                     # Algorithme TOPSIS (idéal pos/nég)
│   │   ├── wsm.py                        # Baseline simple (Somme pondérée)
│   │   └── sensitivity.py                # Analyse de sensibilité des poids
│   ├── ml/                               # Composantes ML & Statistiques (Cours ECC)
│   │   ├── pca_analysis.py               # ACP (Cours 2, slides 17-37)
│   │   ├── clustering.py                 # K-Means & CAH (Cours 2, slides 38-77)
│   │   └── explainability.py             # SHAP & XAI (Cours 1, slides 51-54)
│   ├── preprocessing/
│   │   └── data_pipeline.py              # Three-Way Holdout (Cours 1, slides 84-86)
│   └── utils/
│       └── config.py                     # Chemins et constantes globales
├── Cours_ML_ECC_1.pdf                    # Support de cours ML partie 1
├── Cours_ML_ECC_2.pdf                    # Support de cours ML partie 2
├── Projets ML_ECC.xlsx                   # Liste officielle des projets et consignes
├── requirements.txt                      # Dépendances Python du projet
└── README.md                             # Documentation principale du projet
```

---

## 🚀 Installation & Démarrage

```bash
# 1. Cloner ou ouvrir le dossier
cd "PROJET MACHINE LEARNING"

# 2. Installer les dépendances
pip install -r requirements.txt

# 3. Lancer l'application de démonstration
streamlit run app/app_streamlit.py
```
