"""Prédiction sur des données NOUVELLES (ex. sortie du DSO d'un collègue), avec les modèles déjà entraînés.
Entrée attendue : la dernière heure de vitesses (12 pas de 5 min, en mph) pour les capteurs du dataset actif
(METR-LA : sensor_id de data/adj_mx.pkl ; PEMS08 : sensor_id 0..169 comme GO-FLOW). Capteur absent ou valeur 0/None = manquant.
Les statistiques (scaler, moyenne historique) restent celles du train : rien n'est réappris ici."""
import os

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.baselines import HistoricalAverage
from src.data import MaskedScaler, active_dataset, dataset_splits, load_dataset, results_dir, time_features
from src.features import build_features, neighbor_mats
from src.metrics import HORIZONS

N_IN = 12


class Predictor:
    def __init__(self, h5=None, adj_path=None, models_dir=None, dataset=None):
        self.dataset = dataset or active_dataset()
        models_dir = models_dir or os.path.join(results_dir(self.dataset), "models")
        df, adj, sensor_ids = load_dataset(self.dataset, h5, adj_path)
        values = df.values.astype(np.float64)
        tr = dataset_splits(self.dataset, len(values))[0]
        slot, dow = time_features(df.index)
        self.sensor_ids = sensor_ids
        self.index = {s: j for j, s in enumerate(sensor_ids)}
        self.ha = HistoricalAverage().fit(values[tr.start:tr.end], slot[tr.start:tr.end], dow[tr.start:tr.end])
        self.scaler = MaskedScaler().fit(values[tr.start:tr.end])
        self.A_up, self.A_down = neighbor_mats(adj)
        self.free_flow = np.nanpercentile(np.where(self.ha.week > 0, self.ha.week, np.nan), 85, axis=0)
        self.lgbm = {h: lgb.Booster(model_file=os.path.join(models_dir, f"LightGBM_{h}.txt")) for h in HORIZONS}
        gf_files = {h: os.path.join(models_dir, f"LightGBM_goflow_{h}.txt") for h in HORIZONS}
        self.lgbm_gf = ({h: lgb.Booster(model_file=f) for h, f in gf_files.items()}
                        if all(os.path.exists(f) for f in gf_files.values()) else None)
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

    def predict(self, t0, speeds, goflow=None):
        """goflow (PEMS08) : {"occupancy" | "flow" | "pred_occ15": {sensor_id: valeur à t0}}, sortie GO-FLOW.
        Si fourni et que les modèles LightGBM_goflow existent, c'est le modèle « LightGBM+GO-FLOW » qui sert."""
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

        gf = None
        if goflow and self.lgbm_gf is not None:
            gf = {}
            for name in ("occupancy", "flow", "pred_occ15"):
                arr = np.full((2 * N_IN, len(self.sensor_ids)), np.nan)
                for sid, v in (goflow.get(name) or {}).items():
                    j = self.index.get(str(sid))
                    if j is not None and v is not None:
                        arr[N_IN - 1, j] = float(v)
                gf[name] = arr
            report["goflow_sensors"] = int(np.isfinite(gf["pred_occ15"][N_IN - 1]).sum())
        out = {}
        for hname, h in HORIZONS.items():
            F, _ = build_features(X[None], x_idx, h, slot, dow, FullHA(), self.A_up, self.A_down, True)
            res = {"LightGBM": self.lgbm[hname].predict(F), "HA": ha.predict_at(slot[[N_IN - 1 + h]], dow[[N_IN - 1 + h]])[0]}
            if gf is not None:
                F, _ = build_features(X[None], x_idx, h, slot, dow, FullHA(), self.A_up, self.A_down, True, goflow=gf)
                res["LightGBM+GO-FLOW"] = self.lgbm_gf[hname].predict(F)
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
                # meilleur modèle sur la validation ; avec la sortie GO-FLOW, le modèle qui l'utilise
                best = res.get("LightGBM+GO-FLOW", res.get("GraphWaveNet", res["LightGBM"]))
                ratio = best[j] / self.free_flow[j]
                row[f"pred_{hname}_mph"] = round(float(best[j]), 1)
                row[f"level_{hname}"] = "congestion" if ratio < 0.5 else ("ralenti" if ratio < 0.75 else "fluide")
                for m, v in res.items():
                    row[f"{m}_{hname}"] = round(float(v[j]), 1)
            rows.append(row)
        used = "LightGBM+GO-FLOW" if gf is not None else ("GraphWaveNet" if self.gwn is not None else "LightGBM")
        return dict(t0=str(t0), model_used=used,
                    input_report=report, sensors=rows)
