"""Agent de coordination du trafic (prédiction seulement, pas de RL), rejoué sur la VALIDATION (METR-LA ou PEMS08).
Boucle à chaque pas de 5 min :
  1. perceive      : vitesses observées à t0 + prédictions LightGBM 15/30/60 min
  2. self_evaluate : MAE glissante de chaque modèle (LightGBM, Graph WaveNet si disponible, HA) sur les
                     prédictions déjà vérifiables (cible passée) ; l'agent utilise le meilleur par horizon
  3. decide        : niveau de congestion = vitesse prédite / vitesse fluide du capteur (p85 HA train)
  4. act           : alertes « congestion à venir » et « propagation depuis un voisin », avec cooldown
PEMS08 + GO-FLOW (DSO de la collègue) : sa prévision d'occupation à +15 min sert (a) d'entrée au modèle
« LightGBM+GO-FLOW » mis en concurrence avec les autres, et (b) de confirmation : une alerte n'est émise que si
GO-FLOW prévoit aussi une occupation élevée sur ce capteur (ses seuils train : occupancy_high ou hausse forte).
Toutes les entrées viennent de <résultats du dataset>/dashboard/val_preds.npz (python -m src.precompute_val).
Dataset choisi par la variable d'environnement DSO4_DATASET (metr-la par défaut, ou pems08)."""
from dataclasses import dataclass, field

import numpy as np

HEAVY, MODERATE = 0.5, 0.75          # ratio vitesse / vitesse fluide
STEPS = {"15min": 3, "30min": 6, "60min": 12}
MODEL_KEYS = {"LightGBM": "lgbm", "GraphWaveNet": "gwn", "LightGBM+GO-FLOW": "lgbm_gf", "HA": "ha"}  # nom -> clé .npz


@dataclass
class Alert:
    time: np.datetime64
    kind: str            # "congestion_a_venir" | "propagation"
    sensor: int
    horizon: str
    pred_speed: float
    free_flow: float
    source: int = -1     # voisin déjà congestionné (propagation)
    model: str = "LightGBM"
    action: str = ""
    goflow_occ15: float = float("nan")   # occupation prévue par GO-FLOW à +15 min (confirmation)
    # remplis a posteriori (vérité val) pour mesurer la qualité des alertes
    true_speed: float = float("nan")


@dataclass
class TrafficAgent:
    data: dict
    adj: np.ndarray
    window: int = 12             # 1 h de prédictions vérifiables pour l'auto-évaluation
    cooldown: int = 6            # 30 min sans nouvelle alerte sur le même capteur
    use_goflow: bool = True      # PEMS08 : utiliser la sortie GO-FLOW si elle est dans le .npz
    alerts: list = field(default_factory=list)
    model_used: list = field(default_factory=list)   # par pas : {horizon: nom du modèle choisi}
    rolling: list = field(default_factory=list)      # par pas : {horizon: {modèle: MAE glissante}}

    def __post_init__(self):
        A = (self.adj > 0) & ~np.eye(self.adj.shape[0], dtype=bool)
        self.upstream = [np.flatnonzero(A[:, k]) for k in range(A.shape[0])]  # i -> k
        self.ff = self.data["ha_free_flow"]
        self._last_alert = np.full(A.shape[0], -10**9)
        self.gf = None
        if self.use_goflow and "gf_pred_occ15" in self.data:
            import json
            th = json.loads(str(self.data["gf_thresholds"]))
            P, O = self.data["gf_pred_occ15"], self.data["gf_occupancy"]
            self.gf = (P >= th["occupancy_high"]) | (P - O >= th["strong_change"])   # [S, N] GO-FLOW « congested »
        keys = MODEL_KEYS if self.gf is not None else {m: k for m, k in MODEL_KEYS.items() if k != "lgbm_gf"}
        self.models = [m for m, key in keys.items() if key in self.data]

    # 1. perception
    def perceive(self, i):
        return self.data["current"][i], {m: self.data[MODEL_KEYS[m]][i] for m in self.models}

    # 2. auto-évaluation : seules les prédictions dont la cible est déjà observée (j + h <= i)
    def self_evaluate(self, i):
        out, used = {}, {}
        for k, (hname, h) in enumerate(STEPS.items()):
            lo, hi = max(0, i - h - self.window + 1), i - h + 1
            out[hname] = {m: np.nan for m in self.models}
            if hi > lo:
                y = self.data["true"][lo:hi, k]; msk = y > 0
                if msk.any():
                    for m in self.models:
                        out[hname][m] = float(np.abs(self.data[MODEL_KEYS[m]][lo:hi, k] - y)[msk].mean())
            fin = {m: e for m, e in out[hname].items() if np.isfinite(e)}
            used[hname] = min(fin, key=fin.get) if fin else "LightGBM"
        return out, used

    # 3. décision
    def decide(self, i, current, preds, used):
        pred = np.stack([preds[used[h]][k] for k, h in enumerate(STEPS)])
        ratio_now = np.where(current > 0, current / self.ff, np.nan)
        ratio_pred = pred / self.ff
        return pred, ratio_now, ratio_pred

    # 4. action
    def act(self, i, pred, ratio_now, ratio_pred, used):
        t = self.data["times"][i]
        new = []
        congested_now = ratio_now < HEAVY
        for k, hname in enumerate(STEPS):
            cand = ratio_pred[k] < HEAVY
            if self.gf is not None:
                cand &= self.gf[i]       # confirmation par GO-FLOW
            for s in np.flatnonzero(cand):
                if not (ratio_now[s] >= MODERATE) or i - self._last_alert[s] < self.cooldown:
                    continue  # déjà congestionné (ou inconnu) maintenant, ou alerte récente
                src = [u for u in self.upstream[s] if congested_now[u]] + \
                      [d for d in np.flatnonzero((self.adj[s] > 0)) if d != s and congested_now[d]]
                kind = "propagation" if src else "congestion_a_venir"
                a = Alert(t, kind, int(s), hname, float(pred[k, s]), float(self.ff[s]),
                          source=int(src[0]) if src else -1, model=used[hname],
                          action=_action(kind, hname),
                          goflow_occ15=float(self.data["gf_pred_occ15"][i, s]) if self.gf is not None else float("nan"),
                          true_speed=float(self.data["true"][i, k, s]))
                new.append(a); self._last_alert[s] = i
        self.alerts.extend(new)
        return new

    def step(self, i):
        current, preds = self.perceive(i)
        roll, used = self.self_evaluate(i)
        pred, ratio_now, ratio_pred = self.decide(i, current, preds, used)
        new = self.act(i, pred, ratio_now, ratio_pred, used)
        self.rolling.append(roll); self.model_used.append(used)
        return dict(index=i, time=self.data["times"][i], used=used, rolling=roll, alerts=new,
                    pred=pred, ratio_now=ratio_now, ratio_pred=ratio_pred)

    def run(self, n=None):
        for i in range(n or len(self.data["times"])):
            self.step(i)
        return self


def _action(kind, horizon):
    if kind == "propagation":
        return f"Prévenir l'intersection en amont, adapter les feux et proposer un itinéraire bis avant {horizon}"
    return f"Allonger la phase verte sur l'axe et afficher un message de ralentissement ({horizon})"


def alert_quality(alerts):
    """Sur la validation : une alerte est juste si la vitesse réelle à l'horizon est bien < HEAVY x vitesse fluide."""
    ok = [a for a in alerts if a.true_speed > 0]
    if not ok:
        return dict(n=0, precision=np.nan)
    hit = [a.true_speed < HEAVY * a.free_flow for a in ok]
    near = [a.true_speed < MODERATE * a.free_flow for a in ok]
    return dict(n=len(ok), precision=float(np.mean(hit)), precision_moderate=float(np.mean(near)))


def load(npz=None, adj_path=None, dataset=None):
    import os
    import pickle

    from src.data import DATASETS, active_dataset, load_dataset, results_dir
    dataset = dataset or active_dataset()
    d = dict(np.load(npz or os.path.join(results_dir(dataset), "dashboard", "val_preds.npz"), allow_pickle=False))
    if dataset != "metr-la":
        _, adj, _ = load_dataset(dataset, adj_path=adj_path)
        return d, adj
    adj_path = adj_path or DATASETS[dataset]["adj"]
    with open(adj_path, "rb") as f:
        _, _, adj = pickle.load(f, encoding="latin1")
    return d, np.asarray(adj, dtype=np.float32)
