Réalisé par : Yassine El khantour - Zakani Bakr - Youness Aatoure - Khalil Lekhdim - Mohammed Er-rahmouiy 


# Machine-Learning-Project

# Projet SVM – Classification sur Iris (scikit-learn)

Ce dépôt contient un notebook Jupyter qui met en place un pipeline complet de **classification** (baseline + **SVM**) sur le dataset **Iris** fourni par `scikit-learn`, puis ajoute :

- **Recherche d’hyperparamètres** (GridSearchCV – grille courte)
- **Évaluation par validation croisée** (k=10)
- **Interprétation** (importance des variables, erreurs typiques, limites)
- **Packaging “Colab-ready”** (reproductibilité, versions, cellules exécutables)

> Notebook principal : **`ML_Projet_etapes_image.ipynb`**

---

## 1) Objectifs pédagogiques

- Comprendre la préparation d’un dataset tabulaire (features/target)
- Mettre en place un **Pipeline** propre : `StandardScaler` → `SVC`
- Comparer un modèle **baseline** (DummyClassifier) à un modèle SVM
- Ajuster les hyperparamètres via GridSearchCV
- Évaluer correctement via **validation croisée**
- Interpréter les résultats (rapport de classification, matrice de confusion, importances, erreurs)

---

## 2) Prérequis

- Python **3.9+** (recommandé)
- Jupyter Notebook / JupyterLab **ou** Google Colab

### Dépendances principales
- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`

---

## 3) Installation

### Option A — Environnement local (pip)
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

pip install -U pip
pip install numpy pandas matplotlib scikit-learn jupyter

