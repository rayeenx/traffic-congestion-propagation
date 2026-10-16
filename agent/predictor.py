"""Prédiction sur des données NOUVELLES (ex. sortie du DSO d'un collègue), avec les modèles déjà entraînés.
Entrée attendue : la dernière heure de vitesses (12 pas de 5 min, en mph) pour les capteurs METR-LA,
identifiés par leur sensor_id (ceux de data/adj_mx.pkl). Capteur absent ou valeur 0/None = manquant.
Les statistiques (scaler, moyenne historique) restent celles du train : rien n'est réappris ici."""
import os

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.baselines import HistoricalAverage
from src.data import MaskedScaler, chrono_split, load_metr_la, time_features
from src.features import build_features, neighbor_mats
from src.metrics import HORIZONS

N_IN = 12


class Predictor:
    def __init__(self, h5="data/metr-la.h5", adj_path="data/adj_mx.pkl", models_dir="results/models"):
        df, adj, sensor_ids = load_metr_la(h5, adj_path)
        values = df.values.astype(np.float64)
        tr = chrono_split(len(values))[0]
        slot, dow = time_features(df.index)
        self.sensor_ids = sensor_ids
        self.index = {s: j for j, s in enumerate(sensor_ids)}
        self.ha = HistoricalAverage().fit(values[tr.start:tr.end], slot[tr.start:tr.end], dow[tr.start:tr.end])
        self.scaler = MaskedScaler().fit(values[tr.start:tr.end])
        self.A_up, self.A_down = neighbor_mats(adj)
        self.free_flow = np.nanpercentile(np.where(self.ha.week > 0, self.ha.week, np.nan), 85, axis=0)
        self.lgbm = {h: lgb.Booster(model_file=os.path.join(models_dir, f"LightGBM_{h}.txt")) for h in HORIZONS}
        self.gwn = None
        ckpt = os.path.join(models_dir, "GraphWaveNet.pt")
        if os.path.exists(ckpt):
            try:
                import torch
                from src.gwn import GraphWaveNet, transition_matrices
                self.gwn = GraphWaveNet(len(sensor_ids), transition_matrices(adj))
                self.gwn.load_state_dict(torch.load(ckpt, map_location="cpu"))
                self.gwn.eval()
            except ImportError:
                self.gwn = None  # PyTorch absent : LightGBM + HA seulement

    def window(self, speeds):
        """speeds : {sensor_id: [12 vitesses, de la plus ancienne à t0]} -> X [12, N], rapport de couverture."""
        N = len(self.sensor_ids)
        X = np.zeros((N_IN, N))
        unknown = []
        for sid, seq in speeds.items():
            j = self.index.get(str(sid))
            if j is None:
                unknown.append(str(sid)); continue
            seq = [0.0 if v is None else float(v) for v in seq]
            if len(seq) != N_IN:
                raise ValueError(f"capteur {sid} : {len(seq)} valeurs, il en faut {N_IN} (1 h à 5 min)")
            X[:, j] = seq
        covered = int((X > 0).any(axis=0).sum())
        return X, dict(sensors_with_data=covered,
                       sensors_total=N, unknown_ids=unknown[:20], n_unknown=len(unknown))

    def predict(self, t0, speeds):
        t0 = pd.Timestamp(t0).floor("5min")
        X, report = self.window(speeds)
        if report["sensors_with_data"] < 0.5 * len(self.sensor_ids):
            report["warning"] = "moins de la moitié des capteurs ont des données : prédictions peu fiables"
        stamps = pd.date_range(t0 - pd.Timedelta(minutes=5 * (N_IN - 1)), periods=2 * N_IN, freq="5min")
        slot, dow = time_features(stamps)
        x_idx = np.arange(N_IN)[None, :]
        ha = self.ha

        class FullHA:  # même feature HA qu'en validation (train complet)
            def predict(self, y, s, d, w=None):
                return ha.predict_at(s[y], d[y])

        out = {}
        for hname, h in HORIZONS.items():
            F, _ = build_features(X[None], x_idx, h, slot, dow, FullHA(), self.A_up, self.A_down, True)
            res = {"LightGBM": self.lgbm[hname].predict(F), "HA": ha.predict_at(slot[[N_IN - 1 + h]], dow[[N_IN - 1 + h]])[0]}
            out[hname] = res
        if self.gwn is not None:
            import torch
            z = np.where(X > 0, self.scaler.transform(X), 0.0)
            tod = np.repeat((slot[:N_IN] / 288.0)[:, None], X.shape[1], 1)
            x = torch.tensor(np.stack([z, tod])[None].transpose(0, 1, 3, 2), dtype=torch.float32)  # [1, 2, N, 12]
            with torch.no_grad():
                y = self.scaler.inverse(self.gwn(x))[0].numpy()  # [12, N]
            for hname, h in HORIZONS.items():
                out[hname]["GraphWaveNet"] = y[h - 1]
        rows = []
        for j, sid in enumerate(self.sensor_ids):
            row = dict(sensor_id=sid, current_mph=round(float(X[-1, j]), 1) if X[-1, j] > 0 else None,
                       free_flow_mph=round(float(self.free_flow[j]), 1))
            for hname, res in out.items():
                best = res.get("GraphWaveNet", res["LightGBM"])  # meilleur modèle sur la validation
                ratio = best[j] / self.free_flow[j]
                row[f"pred_{hname}_mph"] = round(float(best[j]), 1)
                row[f"level_{hname}"] = "congestion" if ratio < 0.5 else ("ralenti" if ratio < 0.75 else "fluide")
                for m, v in res.items():
                    row[f"{m}_{hname}"] = round(float(v[j]), 1)
            rows.append(row)
        return dict(t0=str(t0), model_used="GraphWaveNet" if self.gwn is not None else "LightGBM",
                    input_report=report, sensors=rows)
