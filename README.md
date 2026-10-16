# Traffic Congestion Propagation

Prédiction de la propagation de la congestion entre intersections sur le réseau autoroutier de Los Angeles (dataset METR-LA), avec LightGBM et Graph WaveNet, et un agent autonome qui prévient les intersections concernées via une API FastAPI.

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-Graph%20WaveNet-ee4c2c) ![LightGBM](https://img.shields.io/badge/LightGBM-SHAP-green) ![FastAPI](https://img.shields.io/badge/FastAPI-agent-009688) ![Streamlit](https://img.shields.io/badge/Streamlit-dashboard-ff4b4b)

## Le problème

Un bouchon ne reste pas sur place : il se propage d'un capteur à ses voisins sur la route. L'objectif est de **prédire la vitesse de chaque capteur à 15, 30 et 60 minutes** en utilisant le graphe routier, puis de **détecter à l'avance les capteurs qui vont être touchés** par un bouchon qui arrive d'un voisin.

## Contexte : un DSO dans un projet multi-agents

Ce dépôt est ma partie individuelle (DSO 4) d'un projet de groupe sur la mobilité urbaine à ESPRIT (filière Data Science). Le pipeline du groupe :

```
Données trafic -> Prédiction ML (ce dépôt) -> Simulation SUMO + météo -> Dataset commun -> Agents coopératifs
```

Mon livrable est un **agent de prédiction et de coordination** : il publie des prédictions et des alertes de congestion que les agents des autres membres du groupe consomment (par exemple pour le calcul d'itinéraires). L'agent ne fait que des **recommandations** ; les actions sont exécutées par les autres DSO.

## Ce que fait le projet

- **Modèles de prévision** : baselines (LastValue, Historical Average), LightGBM avec features des voisins amont/aval, Graph WaveNet (PyTorch, GPU).
- **Ablation du graphe** : vrai graphe routier vs graphe aléatoire de même densité vs aucun voisin, pour vérifier que l'information de propagation aide vraiment.
- **Explicabilité** : SHAP (TreeSHAP de LightGBM) par capteur, avec une explication en langage simple générée par l'API Claude (optionnelle).
- **Suivi des expériences** : MLflow.
- **Agent autonome** (FastAPI + boucle en arrière-plan) :
  1. *Perceive* : lit les vitesses actuelles des 207 capteurs et les prédictions à 15/30/60 min ;
  2. *Self-evaluate* : calcule la MAE glissante sur la dernière heure de chaque modèle (LightGBM, Graph WaveNet, Historical Average), uniquement sur des prédictions dont la vraie valeur est déjà connue, et choisit le meilleur par horizon ;
  3. *Decide* : repère les capteurs encore fluides qui vont passer en congestion, et ceux dont un voisin est déjà bloqué (propagation) ;
  4. *Act* : envoie des alertes avec une action recommandée, et les garde dans un journal de décisions.
- **Dashboard Streamlit** : carte animée des capteurs, alertes de propagation, suivi d'un capteur (réel vs prédit), SHAP et résultats.

## Données

- **METR-LA** : 207 capteurs de boucles magnétiques sur les autoroutes de Los Angeles, une mesure de vitesse (mph) toutes les 5 minutes, mars à juin 2012.
- **Graphe routier** : matrice d'adjacence dirigée `adj_mx.pkl` et coordonnées des capteurs, issues du dépôt [DCRNN](https://github.com/liyaguang/DCRNN).
- Les données **ne sont pas versionnées**. Pour les télécharger :

```bash
python -m src.download_data      # -> data/metr-la.h5 + data/adj_mx.pkl
```

Ajouter aussi `graph_sensor_locations.csv` (dépôt DCRNN) dans `data/` pour la carte du dashboard.

## Protocole d'évaluation (anti-fuite)

- Split **chronologique** 70 % train / 10 % validation / 20 % test ; aucune fenêtre (12 pas en entrée, 12 pas en sortie) ne traverse deux blocs.
- Normalisation et moyenne historique calculées **sur le train uniquement**.
- Vitesse 0 = valeur manquante : exclue des statistiques et des métriques (**MAE / RMSE / MAPE masquées**).
- Horizons : 15, 30 et 60 minutes (3, 6 et 12 pas).
- Le **test est verrouillé** (`src/leakage.py`) et ne sera ouvert qu'une seule fois, à la fin.

## Installation

```bash
git clone https://github.com/<votre-compte>/traffic-congestion-propagation.git
cd traffic-congestion-propagation
python -m venv .venv
# Windows : .venv\Scripts\activate    |    Linux/macOS : source .venv/bin/activate
pip install -r requirements.txt
pip install torch                 # pour Graph WaveNet (version CUDA conseillée si GPU)
python -m src.download_data
```

## Utilisation

Toutes les commandes se lancent depuis la racine du dépôt.

```bash
# 1. Baselines et LightGBM (validation)
python -m src.run_baselines
python -m src.run_lgbm                     # --graph real | random | none pour l'ablation

# 2. Graph WaveNet (GPU conseillé)
python -m src.run_gwn --graph real         # ablation : --graph random

# 3. Prédictions de validation utilisées par l'agent et le dashboard
python -m src.precompute_val               # -> results/dashboard/val_preds.npz

# 4. Dashboard
streamlit run dashboard/app.py             # ou double-clic sur run_dashboard.bat (Windows)
                                           # -> http://localhost:8501

# 5. API de l'agent
uvicorn agent.api:app --port 8004          # -> http://localhost:8004/docs
```

Endpoints principaux de l'agent :

| Méthode | Route | Rôle |
|---|---|---|
| GET | `/agent/status` | modèle choisi par horizon, MAE glissante, nombre d'alertes |
| POST | `/agent/step?n=12` | fait avancer l'agent de n pas de 5 min |
| GET | `/agent/alerts` | dernières alertes (congestion à venir, propagation) |
| GET | `/agent/predictions?horizon=30min` | vitesse prédite et niveau pour les 207 capteurs |
| POST | `/agent/predict` | prédiction sur une entrée fournie (exemple : `examples/exemple_entree_predict.json`) |

L'agent rejoue la période de **validation** ; il refuse de rejouer le test.

## Structure

```
src/         chargement, split, métriques masquées, baselines, contrôles anti-fuite,
             features LightGBM, graphes (réel / aléatoire), Graph WaveNet (gwn.py, run_gwn.py)
agent/       agent autonome (core.py), modèles (predictor.py), analyse de propagation, API FastAPI
dashboard/   application Streamlit (carte, démo, propagation, SHAP, explication LLM)
examples/    exemple d'entrée pour POST /agent/predict
notebooks/   exploration des données et baselines
tests/       tests unitaires
data/        données METR-LA (non versionnées)
results/     métriques, modèles et prédictions générés (non versionnés)
```

## Résultats (validation)

MAE masquée en mph, une seule seed. **Ce sont des résultats de validation** (early stopping sur la validation, donc légèrement optimistes) ; le test n'a pas encore été ouvert.

| Modèle | 15 min | 30 min | 60 min |
|---|---|---|---|
| LastValue | 3.22 | 3.82 | 4.78 |
| Historical Average | 4.32 | 4.32 | 4.31 |
| LightGBM, sans voisins | 2.79 | 3.24 | 3.69 |
| LightGBM, graphe aléatoire | 2.77 | 3.20 | 3.63 |
| LightGBM, vrai graphe | 2.69 | 3.12 | 3.59 |
| Graph WaveNet, graphe aléatoire | 2.55 | 2.87 | 3.23 |
| **Graph WaveNet, vrai graphe** | **2.48** | **2.82** | **3.19** |

**Ce qu'on retient** : pour les deux modèles, le vrai graphe routier bat un graphe aléatoire de même densité à tous les horizons. L'information des voisins (la propagation) aide donc vraiment, mais le gain est petit et diminue à 60 minutes.

### Propagation observée

Un capteur est considéré en congestion quand sa vitesse passe sous **50 % de sa vitesse fluide** (85e percentile des vitesses du train), et en ralentissement sous 75 %.

Sur 6 jours ouvrés de la validation (pics du matin et du soir) :
- **78 %** des 310 transitions de congestion vont vers le capteur **en amont** (10 % vers l'aval, 12 % sur des routes à double sens) : le bouchon remonte la route, à contresens de la circulation (onde de choc) ;
- vitesse médiane de l'onde : **environ 8 km/h** (distance à vol d'oiseau, donc sous-estimée).

### Agent

- Sur toute la validation : **1343 alertes** ; dans **83 %** des cas la vitesse réelle passe ensuite sous 75 % de la vitesse fluide, et dans **65 %** des cas sous 50 %.
- Auto-évaluation : l'agent choisit Graph WaveNet environ 85 % du temps à 15 min et 75 % du temps à 60 min.
- Anticipation : alerte en médiane **35 minutes avant** le début du bouchon ; elle couvre **50 %** des débuts de congestion (423 sur 854), car l'agent n'alerte que les capteurs encore fluides.

## Limites

- Résultats sur la validation uniquement, avec une seule seed : pas encore d'intervalle de confiance.
- L'agent rejoue des données historiques, il n'est pas branché sur un flux temps réel.
- La vitesse de l'onde est calculée à vol d'oiseau, pas sur la distance routière.
- Les seuils de congestion (50 % / 75 % de la vitesse fluide) sont un choix de modélisation, pas une norme.
- METR-LA date de 2012 et ne couvre que des autoroutes.

## Prochaines étapes

- Ouvrir le **test** une seule fois pour les chiffres finaux.
- Ajouter **Chronos-Bolt** (foundation model de séries temporelles, zero-shot) comme baseline.
- **Docker Compose** pour lancer l'agent avec les agents du groupe.
- Tester plusieurs seeds pour mesurer la variance.

## Références

- Li et al., *Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting*, ICLR 2018 (dataset METR-LA et graphe).
- Wu et al., *Graph WaveNet for Deep Spatial-Temporal Graph Modeling*, IJCAI 2019.

## Auteur

Rayan Zikoukout, ESPRIT, Data Science (dernière année).
