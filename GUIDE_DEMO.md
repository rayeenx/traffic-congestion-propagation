# DSO 4 · Guide de démo, bilan et intégration

## 1. Démarrer le projet

**Dashboard (le plus important pour la prof)**
1. Ouvrir le dossier `C:\Users\GIGABYTE\Desktop\dso4`
2. Double-cliquer sur `run_dashboard.bat`
3. Le navigateur s'ouvre sur `http://localhost:8501` (sinon, ouvrir cette adresse dans Chrome)

Ou bien depuis un terminal VS Code, à la racine du projet :
```bash
.venv\Scripts\activate
streamlit run dashboard/app.py
```

**API de l'agent (pour montrer que c'est un vrai agent / pour l'intégration)**
```bash
.venv\Scripts\activate
uvicorn agent.api:app --port 8004
```
Puis ouvrir `http://localhost:8004/docs` : on peut cliquer sur chaque endpoint et l'exécuter.
- `POST /agent/step?n=12` : l'agent avance d'1 h (12 pas de 5 min)
- `GET /agent/status` : modèle choisi par horizon, MAE glissante, nombre d'alertes
- `GET /agent/alerts` : dernières alertes
- `GET /agent/predictions?horizon=30min` : vitesse prédite + niveau pour les 207 capteurs

**Conseils pour demain**
- PC **branché** et internet actif (le fond de carte vient d'OpenStreetMap).
- Si la carte reste blanche, choisir « Sans fond » dans la barre latérale.
- Le dashboard démarre sur jeudi 24/05 17:30 (pic de congestion). Éviter lundi 28/05, c'est Memorial Day.
- Démo conseillée (3 minutes) :
  1. Onglet **🎬 Démo** : lire la phrase du haut, choisir jeudi 24/05, cliquer **▶ Play**. On voit la journée défiler :
     les points violets (alertes) apparaissent **avant** que les points deviennent rouges (bouchons).
  2. Sous la carte, la courbe : rouge = bouchons réels, violet pointillé = ce que l'agent avait prévu 30 min avant.
     Elles se suivent, donc l'agent anticipe. Les barres violettes = alertes, elles montent avant les pics.
  3. Onglet **🤖 Agent** : le modèle choisi tout seul (souvent Graph WaveNet) et le flux d'alertes.
  4. Onglet **📊 Résultats** : le vrai graphe bat un graphe aléatoire, donc la propagation entre voisins existe.

## 2. Ce qu'on a fait

| Étape | État | Fichiers |
|---|---|---|
| Données METR-LA (207 capteurs, 5 min) + graphe DCRNN | fait | `data/`, `src/data.py` |
| Protocole anti-fuite : split chrono 70/10/20, scaler train-only, 0 = manquant, test verrouillé | fait | `src/leakage.py` |
| Baselines LastValue, Historical Average | fait | `src/baselines.py` |
| LightGBM avec voisins amont/aval + ablation (vrai graphe / aléatoire / sans voisins) + SHAP | fait | `src/run_lgbm.py` |
| Graph WaveNet (GPU) + ablation graphe aléatoire | fait | `src/gwn.py`, `src/run_gwn.py` |
| Agent autonome sur données réelles + API FastAPI | fait | `agent/core.py`, `agent/api.py` |
| Dashboard Streamlit (carte, agent, SHAP, résultats) | fait | `dashboard/app.py` |

**Résultats (validation, MAE masquée en mph)**

| Modèle | 15 min | 30 min | 60 min |
|---|---|---|---|
| LastValue | 3.22 | 3.81 | 4.78 |
| Historical Average | 4.32 | 4.32 | 4.31 |
| LightGBM sans voisins | 2.79 | 3.24 | 3.69 |
| LightGBM graphe aléatoire | 2.77 | 3.20 | 3.63 |
| LightGBM vrai graphe | 2.69 | 3.12 | 3.59 |
| Graph WaveNet graphe aléatoire | 2.55 | 2.87 | 3.23 |
| **Graph WaveNet vrai graphe** | **2.48** | **2.82** | **3.19** |

Message clé : le vrai graphe routier bat un graphe aléatoire de même densité pour les deux modèles, ce qui veut dire
que l'information des voisins (la propagation) aide vraiment. Le gain est petit et diminue à 60 min.

Agent sur toute la validation : 1343 alertes ; 65 % voient la vitesse réelle passer sous 50 % de la vitesse fluide,
83 % sous 75 %.

## 3. Ce qui reste

- **Test final** : ouvrir le test une seule fois (`TestGuard`) pour les chiffres définitifs du rapport. À faire à la
  fin, quand plus rien ne change. Les chiffres actuels sont de la validation (early stopping sur val, donc un peu optimistes).
- **Chronos-Bolt** (foundation model zero-shot) : prévu, pas fait.
- **Docker Compose** : pas fait (utile pour l'intégration avec le groupe, voir section 5).
- Optionnel : ablation Graph WaveNet « graphe appris seul » (`python -m src.run_gwn --graph adaptive`, ~1 h, PC branché).
- Confirmer avec la prof le rôle exact de l'agent (Agent 4 Cooperative Traffic Coordination ?).

## 4. Comment l'agent marche

Il tourne en boucle, un pas toutes les 5 minutes (sur la validation, il « rejoue » le temps) :

1. **Perceive** : il lit la vitesse actuelle des 207 capteurs et les prédictions à 15/30/60 min de trois
   modèles (LightGBM, Graph WaveNet, Historical Average).
2. **Self-evaluate** : pour chaque horizon, il calcule l'erreur (MAE) de chaque modèle sur la dernière heure,
   mais seulement avec les prédictions dont la vraie valeur est déjà connue (pas de triche). Il choisit le
   modèle le plus fiable en ce moment (Graph WaveNet ~80 % du temps, LightGBM ou HA sinon).
3. **Decide** : pour chaque capteur, ratio = vitesse prédite / vitesse fluide du capteur
   (vitesse fluide = 85e percentile de la moyenne historique du train).
   Congestion si ratio < 50 %, ralenti si < 75 %.
4. **Act** : si un capteur est fluide maintenant mais prévu en congestion, il lance une alerte :
   - **propagation** si un voisin routier (amont ou aval) est déjà congestionné,
   - **congestion à venir** sinon,
   avec une action recommandée (adapter les feux, prévenir l'intersection en amont, itinéraire bis) et un
   cooldown de 30 min par capteur pour ne pas spammer.

Ce n'est pas du RL : l'agent ne contrôle pas les feux, il prédit, se surveille lui-même et recommande.

## 5. Intégrer ton DSO avec ceux du groupe

Je ne connais pas les DSO de tes collègues, donc voici une proposition simple à leur montrer :

**Principe : chaque DSO = un service avec une API, et on se met d'accord sur un format de message.**

1. **Ton rôle dans le système** : tu fournis la *prévision de congestion*. Les autres agents
   (ex. gestion des feux, transport en commun, parking, itinéraires) consomment ta sortie.
2. **Ton contrat d'API** (déjà prêt, port 8004) :
   - `GET /agent/predictions?horizon=30min` → liste `{sensor_id, pred_speed_mph, free_flow_mph, level}`
   - `GET /agent/alerts` → liste `{time, kind, sensor, horizon, pred_speed, source, model, action}`
3. **À fixer ensemble** :
   - l'identifiant commun des lieux (sensor_id METR-LA, ou lat/lon de `data/graph_sensor_locations.csv`
     si leurs données sont ailleurs),
   - l'horloge commune (pas de 5 min, même instant de simulation pour tous),
   - le format JSON des messages entre agents.
4. **Mise en commun technique** : un `docker-compose.yml` avec un service par DSO (chacun son port) et un
   dashboard commun qui appelle toutes les API. Version plus simple si le temps manque : chacun garde son
   FastAPI et un script « orchestrateur » appelle les API dans l'ordre à chaque pas.
5. **Exemple de coopération** : ton agent publie « capteur 717497 en congestion dans 30 min, propagation
   depuis 717495 » → l'agent feux allonge le vert en amont → l'agent itinéraires propose un détour.
