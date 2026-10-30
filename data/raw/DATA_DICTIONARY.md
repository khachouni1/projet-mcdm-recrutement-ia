# Dictionnaire des Données (Data Dictionary)
## Jeu de Données : *Predicting Hiring Decisions in Recruitment Data*

- **Fichier associé :** [`recruitment_data.csv`](file:///C:/Users/HP/.gemini/antigravity/scratch/PROJET%20MACHINE%20LEARNING/data/raw/recruitment_data.csv)
- **Source :** Kaggle / UC Berkeley Haas
- **Auteur :** Rabie El Kharoua
- **Lien web :** [Kaggle Dataset](https://www.kaggle.com/datasets/rabieelkharoua/predicting-hiring-decisions-in-recruitment-data)
- **Dimensions :** 1 500 observations × 11 variables

---

## 📋 Description Détaillée des Variables

| # | Nom de la colonne | Type | Plage / Valeurs | Signification métier | Utilisation dans notre Système Multi-Agents & MCDM |
|---|---|---|---|---|---|
| **1** | **`Age`** | Numérique (Entier) | 20 à 50 ans | Âge chronologique du candidat à la date de candidature. | **Agent Éthique & Équité :** Utilisé pour auditer l'absence de discrimination liée à l'âge (*ageism*) et vérifier l'équité des décisions. |
| **2** | **`Gender`** | Catégoriel binaire | `0` = Femme<br>`1` = Homme | Genre déclaré du candidat. | **Agent Éthique & Équité :** Audit de parité femme/homme pour vérifier la parité des taux de sélection (*Fairness Metrics / Disparate Impact*). |
| **3** | **`EducationLevel`** | Ordinal (Entier) | `1` : Licence / Bachelor (Bac+3)<br>`2` : Master / Ingénieur (Bac+5)<br>`3` : Doctorat / PhD (Bac+8)<br>`4` : Autre / Certifications | Niveau de diplôme le plus élevé obtenu par le candidat. | **Agent RH & Agent Expérience :** Critère de qualification académique de base dans la matrice de décision. |
| **4** | **`ExperienceYears`** | Numérique (Entier) | 0 à 15 ans | Nombre total d'années d'expérience professionnelle cumulée. | **Agent Séniorité & Expérience :** Critère central pour évaluer l'autonomie et le niveau de séniorité requis pour le poste. |
| **5** | **`PreviousCompanies`** | Numérique (Entier) | 1 à 5 entreprises | Nombre d'entreprises différentes dans lesquelles le candidat a travaillé. | **Agent RH :** Analyse de la fidélité et de la stabilité en poste (détection du *job hopping* vs diversité des expériences). |
| **6** | **`DistanceFromCompany`** | Numérique (Continu) | 1.0 à 50.0 km | Distance géographique estimée entre le domicile du candidat et les locaux de l'entreprise. | **Critère logistique / Éthique :** Évaluation de la contrainte de trajet. L'Agent Éthique veille à ce qu'une distance élevée ne pénalise pas un candidat (politique de télétravail). |
| **7** | **`InterviewScore`** | Numérique (Continu) | 0.0 à 100.0 | Note globale obtenue lors de l'entretien avec le recruteur RH (aisance relationnelle, motivation, clarté). | **Agent RH & Culture Fit :** Critère majeur dans l'évaluation de la compatibilité humaine et culturelle. |
| **8** | **`SkillScore`** | Numérique (Continu) | 0.0 à 100.0 | Note obtenue lors du test technique d'évaluation des compétences (*coding test*, étude de cas, QCM technique). | **Agent Lead Technique :** Critère prépondérant dans l'AHP pour mesurer l'excellence opérationnelle. |
| **9** | **`PersonalityScore`** | Numérique (Continu) | 0.0 à 100.0 | Score issu d'un test psychométrique / comportemental (*Soft skills*, travail d'équipe, gestion du stress). | **Agent RH :** Mesure des compétences interpersonnelles et du travail collaboratif. |
| **10** | **`RecruitmentStrategy`** | Catégoriel (Entier) | `1` : Candidature directe (Site carrières)<br>`2` : Cabinet / Chasseur de têtes<br>`3` : Cooptation / Réseau professionnel | Canal d'acquisition de la candidature. | **Analyse contextuelle :** Utilisé en EDA pour analyser si certains canaux amènent des profils plus qualifiés. |
| **11** | **`HiringDecision`** | Binaire (*Target*) | `0` = Candidat rejeté (69%)<br>`1` = Candidat embauché (31%) | Décision finale d'embauche prise par l'entreprise (*Ground Truth*). | **Vérité terrain pour l'évaluation :** Utilisé pour entraîner nos modèles ML supervisés, évaluer la précision de la sélection et calculer les explications SHAP. |
