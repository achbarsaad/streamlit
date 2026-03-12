# 🏠 TP3 — Déploiement Application ML : House Prices

Application Streamlit de Machine Learning pour la prédiction de prix immobiliers.  
Développée dans le cadre du TP3 — Déploiement & Sécurisation.

---

## 🚀 Démo en ligne

> **URL** : `https://votre-app.streamlit.app` *(à remplir après déploiement)*

**Identifiants de démo :**
| Champ | Valeur |
|-------|--------|
| Utilisateur | `admin` |
| Mot de passe | `tp3_house2024` |

---

## 📁 Structure du projet

```
app/
├── app.py                   ← Accueil + authentification
├── pages/
│   ├── 1_Data.py            ← Upload CSV + exploration + visualisations
│   ├── 2_Training.py        ← Entraînement modèle + performances
│   └── 3_Prediction.py      ← Interface de prédiction interactive
├── utils/
│   ├── data_loader.py       ← Chargement des données
│   ├── preprocessing.py     ← Encodage + préparation features
│   └── visualization.py     ← Graphiques réutilisables
├── assets/
│   └── style.css            ← Thème personnalisé
├── data/
│   └── house_prices_clean.csv  ← Dataset par défaut
├── .streamlit/
│   ├── config.toml          ← Thème et configuration serveur
│   └── secrets.toml         ← Secrets locaux (NON commité)
├── requirements.txt         ← Dépendances Python
└── .gitignore               ← Fichiers exclus de Git
```

---

## ⚙️ Installation locale

### Prérequis
- Python 3.10+
- Git

### Étapes

```bash
# 1. Cloner le dépôt
git clone https://github.com/VOTRE_USERNAME/tp3-house-prices.git
cd tp3-house-prices

# 2. Créer un environnement virtuel
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Créer le fichier secrets local
mkdir -p .streamlit
# Copier le contenu de secrets.toml (voir section Secrets)

# 5. Lancer l'application
streamlit run app.py
```

L'application sera accessible sur : **http://localhost:8501**

---

## ☁️ Déploiement sur Streamlit Cloud

### Étape 1 — Préparer GitHub

```bash
git init
git add .
git commit -m "Initial commit — TP3 House Prices ML"
git branch -M main
git remote add origin https://github.com/VOTRE_USERNAME/tp3-house-prices.git
git push -u origin main
```

> ⚠️ Vérifiez que `.streamlit/secrets.toml` est bien dans `.gitignore` avant de pusher !

### Étape 2 — Créer l'app sur Streamlit Cloud

1. Aller sur [https://share.streamlit.io](https://share.streamlit.io)
2. Cliquer **"New app"**
3. Sélectionner votre dépôt GitHub
4. **Main file path** : `app.py`
5. Cliquer **"Deploy"**

### Étape 3 — Configurer les Secrets

Dans Streamlit Cloud → votre app → **Settings > Secrets**, coller :

```toml
[auth]
admin_user     = "admin"
admin_password = "tp3_house2024"

[app]
secret_key = "votre_cle_secrete_unique"
```

---

## 🔒 Sécurité

| Mesure | Implémentation |
|--------|---------------|
| Authentification | Formulaire login avec vérification via `st.secrets` |
| HTTPS | ✅ Automatique sur Streamlit Cloud (cadenas 🔒) |
| Secrets | Variables dans `st.secrets`, jamais en dur dans le code |
| Validation des entrées | Vérification taille fichier (max 10 MB), format CSV, longueur champs |
| Protection XSRF | Activée dans `config.toml` |
| Logs | `logging` Python standard, visibles dans Streamlit Cloud Logs |

---

## 📦 Dépendances

| Package | Version | Rôle |
|---------|---------|------|
| streamlit | ≥1.32.0 | Framework web |
| scikit-learn | ≥1.4.0 | Modèles ML |
| plotly | ≥5.20.0 | Graphiques interactifs |
| pandas | ≥2.0.0 | Manipulation données |
| numpy | ≥1.26.0 | Calculs numériques |
| python-dotenv | ≥1.0.0 | Variables d'environnement locales |

---

## 🧪 Fonctionnalités

### 📁 Page 1 — Data
- Upload d'un fichier CSV personnalisé (max 10 MB)
- Exploration : aperçu, types, statistiques, valeurs manquantes
- Visualisations interactives : histogramme, boxplot, scatter, violin, barplot

### 🤖 Page 2 — Training
- Choix de la **colonne cible** et des **features** depuis l'interface
- 4 algorithmes : Random Forest, Gradient Boosting, Régression Linéaire, Ridge
- Détection automatique régression / classification
- Graphiques : scatter prédit/réel, résidus, matrice de confusion, importance features

### 🔮 Page 3 — Prediction
- Sliders dynamiques sur le top 10 features importantes
- Affichage des probabilités par classe (classification) avec vrais noms
- Comparaison de la valeur prédite vs la distribution réelle (régression)

---

## 📝 Auteur

**ACHBARS** — TP3 Streamlit — 2024
