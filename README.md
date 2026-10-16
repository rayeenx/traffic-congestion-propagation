# DSO4 : prédiction de la propagation de la congestion (METR-LA)

Partie individuelle d'un projet de groupe (mobilité urbaine). Prédiction uniquement.

## Installation (VS Code, local)
```bash
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install -r requirements.txt
python -m src.download_data      # data/metr-la.h5 + data/adj_mx.pkl
python -m src.run_baselines      # baselines sur la validation
```
Dans VS Code : `Python: Select Interpreter` -> `.venv`, et le même kernel pour les notebooks. Toutes les commandes se lancent depuis la racine du dépôt.

## Structure
- `data/` : `metr-la.h5`, `adj_mx.pkl` (non versionnés, `python -m src.download_data`)
- `src/` : `data.py` (chargement, split, fenêtres, scaler), `metrics.py` (métriques masquées), `baselines.py`, `leakage.py` (contrôles anti-fuite + verrou du test), `run_baselines.py`
- `notebooks/01_data_baselines.ipynb` : exploration + baselines
- `results/` : CSV des métriques

## Protocole
- Split chronologique 70/10/20 sur l'axe temps ; fenêtres 12 pas -> 12 pas entièrement contenues dans un bloc.
- Scaler et moyenne historique ajustés sur le train uniquement. Vitesse 0 = manquant, exclue des métriques et des statistiques.
- Horizons 15/30/60 min. Test ouvert une seule fois à la fin (verrou `results/TEST_OPENED.lock`).

## Résultats
À remplir uniquement avec les sorties du code.

## Démo : agent + dashboard (validation uniquement)
```bash
pip install streamlit plotly fastapi uvicorn
python -m src.precompute_val          # prédictions LightGBM/HA/LastValue sur la val -> results/dashboard/val_preds.npz
streamlit run dashboard/app.py        # ou double-clic sur run_dashboard.bat (Windows)
uvicorn agent.api:app --port 8004     # API de l'agent : /agent/status, POST /agent/step?n=12, /agent/alerts
```
- `agent/core.py` : boucle perceive -> self-evaluate (MAE glissante LightGBM vs HA, repli HA) -> decide -> act (alertes).
- `data/graph_sensor_locations.csv` : coordonnées des capteurs (dépôt DCRNN).
- Si la carte reste blanche (navigateur sans WebGL2), choisir « Sans fond » dans la barre latérale.
- Graph WaveNet : `python -m src.run_gwn --graph real` (GPU, ~40 min sur RTX 4060 branchée), puis relancer
  `python -m src.precompute_val` pour l'ajouter à l'agent et au dashboard. Ablation : `--graph random` / `--graph adaptive`.
