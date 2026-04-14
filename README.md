# 🤖 Chatbot RAG : Analyse de CV

## 📚 Informations du projet

- **Filière** : 4IASDR (Intelligence Artificielle et Science des Données Avancée)
- **Établissement** : EMSI (École Marocaine des Sciences de l'Ingénieur)
- **Encadrant** : Pr. Idrissi Zouggari Nadia
- **Étudiant** : Othmane Boushabi
- **Année académique** : 2024-2025

## 🎯 Objectif du projet

Ce projet implémente un **chatbot intelligent** basé sur l'architecture **RAG (Retrieval-Augmented Generation)** pour analyser automatiquement le contenu d'un CV et répondre aux questions des utilisateurs de manière précise et contextuelle.

## 🏗️ Architecture du système

```
CV (PDF/DOCX)
    ↓
[1] Extraction du texte
    ↓
[2] Nettoyage et prétraitement
    ↓
[3] Chunking (découpage en segments)
    ↓
[4] Génération d'embeddings (all-MiniLM-L6-v2)
    ↓
[5] Stockage dans FAISS (Vector Store)
    ↓
[6] Recherche de similarité lors d'une question
    ↓
[7] Génération de réponse avec Gemini (LLM)
```

## 🛠️ Technologies utilisées

| Technologie | Usage | Version |
|------------|-------|---------|
| **Streamlit** | Interface utilisateur web | 1.31.0 |
| **LangChain** | Framework RAG | 0.1.6 |
| **FAISS** | Vector database | 1.7.4 |
| **Sentence Transformers** | Génération d'embeddings | 2.3.1 |
| **Google Gemini** | Modèle de langage (LLM) | API v1 |
| **PyPDF** | Extraction de texte PDF | 4.0.1 |
| **python-docx** | Extraction de texte DOCX | 1.1.0 |

## 📁 Structure du projet

```
rag_cv_project/
│
├── app.py                  # Interface Streamlit principale
├── requirements.txt        # Dépendances Python
├── README.md              # Documentation
│
├── src/                   # Package principal
│   ├── __init__.py        # Initialisation du package
│   └── pipeline.py        # Pipeline RAG (chargement, chunking, RAG)
│
├── data/                  # Dossier pour les CV uploadés (créé auto)
│
└── .gitignore            # Fichiers à ignorer dans Git
```

## 🚀 Installation et lancement

### 1️⃣ Prérequis

- Python 3.8 ou supérieur
- pip (gestionnaire de packages Python)
- Connexion Internet (pour les API)

### 2️⃣ Installation des dépendances

```bash
# Cloner le projet (ou télécharger les fichiers)
cd rag_cv_project

# Créer un environnement virtuel (recommandé)
python -m venv venv

# Activer l'environnement virtuel
# Sur Windows :
venv\Scripts\activate
# Sur Linux/Mac :
source venv/bin/activate

# Installer les dépendances
pip install -r requirements.txt
```

### 3️⃣ Configuration de l'API Gemini

1. Obtenir une clé API Google Gemini :
   - Aller sur https://makersuite.google.com/app/apikey
   - Créer ou récupérer votre clé API

2. Modifier la clé dans `src/pipeline.py` :
   ```python
   GOOGLE_API_KEY = "VOTRE_CLE_API_ICI"
   ```

### 4️⃣ Lancement de l'application

```bash
streamlit run app.py
```

L'application s'ouvrira automatiquement dans votre navigateur à l'adresse : `http://localhost:8501`

## 💡 Utilisation

### Étape 1 : Charger un CV
- Cliquez sur "📤 Charger un CV" dans la barre latérale
- Sélectionnez un fichier PDF ou DOCX
- Attendez le traitement automatique

### Étape 2 : Poser des questions
Exemples de questions :
- "Quelle est la formation mentionnée dans ce CV ?"
- "Quelles sont les compétences techniques ?"
- "Décris l'expérience professionnelle"
- "Quels projets sont mentionnés ?"
- "Quelles langues cette personne parle-t-elle ?"

### Étape 3 : Analyser les réponses
Le chatbot répond en se basant **uniquement** sur le contenu du CV chargé.

## 🔧 Fonctionnalités principales

✅ **Chargement multi-format** : PDF et DOCX  
✅ **Chunking intelligent** : Découpage optimisé avec overlap  
✅ **Recherche sémantique** : Utilisation de FAISS pour la similarité  
✅ **Génération contextuelle** : Réponses basées uniquement sur le CV  
✅ **Interface intuitive** : Chat conversationnel avec Streamlit  
✅ **Historique de conversation** : Sauvegarde des échanges  
✅ **Gestion d'erreurs** : Fallback automatique entre modèles Gemini  

## ⚡ Corrections apportées

### Problème initial
```
Erreur : 404 models/gemini-1.5-flash is not found for API version v1beta
```

### Solution implémentée
Le fichier `src/pipeline.py` utilise maintenant :
1. **Premier choix** : `gemini-2.0-flash-exp` (modèle gratuit le plus récent)
2. **Fallback** : `gemini-pro` (version stable)

```python
def generate_answer(prompt):
    try:
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        # ...
    except:
        model = genai.GenerativeModel('gemini-pro')
        # ...
```

## 🎓 Aspects pédagogiques

Ce projet illustre les concepts suivants :

- **RAG (Retrieval-Augmented Generation)** : Combiner recherche documentaire et génération de texte
- **Vector Embeddings** : Représentation sémantique du texte
- **Similarity Search** : Recherche par similarité cosinus
- **Prompt Engineering** : Optimisation des prompts pour le LLM
- **State Management** : Gestion de l'état avec Streamlit

## 📊 Métriques de performance

- **Chunk size** : 800 caractères
- **Overlap** : 100 caractères
- **Nombre de chunks récupérés** : 4 (k=4)
- **Modèle d'embeddings** : all-MiniLM-L6-v2 (384 dimensions)
- **Temperature LLM** : 0.3 (génération déterministe)

## 🐛 Résolution de problèmes

### Erreur : "Module not found"
```bash
pip install -r requirements.txt
```

### Erreur : "API Key invalid"
Vérifier la clé dans `src/pipeline.py` et régénérer si nécessaire sur Google AI Studio.

### L'application ne se lance pas
```bash
# Vérifier l'installation de Streamlit
streamlit --version

# Réinstaller si nécessaire
pip install --upgrade streamlit
```

## 📧 Contact

**Othmane Boushabi**  
- Email : othmane.boushabi@gmail.com  
- LinkedIn : [À compléter]  
- GitHub : [À compléter]

## 📝 Licence

Projet académique - 4IASDR - EMSI - 2024/2025

---

**Made with ❤️ for EMSI - 4IASDR**
