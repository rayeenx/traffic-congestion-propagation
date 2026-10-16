"""Précalcule, sur la VALIDATION uniquement, les prédictions utilisées par le dashboard et l'agent :
LightGBM (vrai graphe), Historical Average, LastValue, vérité terrain, pour 15/30/60 min.
Le test n'est jamais lu. Usage : python -m src.precompute_val"""
import os
import warnings

import lightgbm as lgb
import numpy as np

from src.baselines import HistoricalAverage, last_value
from src.data import chrono_split, load_metr_la, make_windows, time_features, window_anchors
from src.features import OOFHistoricalAverage, build_features, neighbor_mats
from src.leakage import check_splits, check_windows
from src.metrics import HORIZONS, masked_metrics

N_IN, N_OUT = 12, 12


def main(h5="data/metr-la.h5", adj_path="data/adj_mx.pkl", models_dir="results/models",
         out="results/dashboard/val_preds.npz"):
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df, adj, sensor_ids = load_metr_la(h5, adj_path)
    values = df.values.astype(np.float64)
    T, N = values.shape
    splits = chrono_split(T); check_splits(splits, T)
    tr, va = splits[0], splits[1]
    slot, dow = time_features(df.index)
    week = np.arange(T) // (7 * 288)
    ha_feat = OOFHistoricalAverage().fit(values[tr.start:tr.end], slot[tr.start:tr.end], dow[tr.start:tr.end],
                                         week[tr.start:tr.end])
    ha = HistoricalAverage().fit(values[tr.start:tr.end], slot[tr.start:tr.end], dow[tr.start:tr.end])
    A_up, A_down = neighbor_mats(adj)

    a_va = window_anchors(va, N_IN, N_OUT)
    Xva, Yva, xi_va, yi_va = make_windows(values, a_va, N_IN, N_OUT); check_windows(va, xi_va, yi_va)
    S = len(a_va)
    hs = list(HORIZONS.values())
    lgbm = np.zeros((S, len(hs), N), np.float32)
    for k, (hname, h) in enumerate(HORIZONS.items()):
        model = lgb.Booster(model_file=os.path.join(models_dir, f"LightGBM_{hname}.txt"))
        F, _ = build_features(Xva, xi_va, h, slot, dow, ha_feat, A_up, A_down, True)
        lgbm[:, k] = model.predict(F).reshape(S, N)
        m = masked_metrics(lgbm[:, k].astype(np.float64), Yva[:, h - 1])
        print(f"LightGBM {hname} val MAE={m['MAE']:.3f} (doit correspondre à results/LightGBM_val.csv)")
    true = np.stack([Yva[:, h - 1] for h in hs], 1).astype(np.float32)
    ha_pred = np.stack([ha.predict(yi_va, slot, dow)[:, h - 1] for h in hs], 1).astype(np.float32)
    lv, _ = last_value(Xva, ha.sensor)
    lv = np.repeat(lv[:, None, :], len(hs), 1).astype(np.float32)
    extra = {}
    gwn_path = os.path.join(os.path.dirname(out), "GraphWaveNet_val_pred.npy")  # python -m src.run_gwn
    if os.path.exists(gwn_path):
        g = np.load(gwn_path)
        assert g.shape[0] == S, "prédictions Graph WaveNet pas alignées sur les fenêtres val"
        extra["gwn"] = np.stack([g[:, h - 1] for h in hs], 1).astype(np.float32)
        for k, hname in enumerate(HORIZONS):
            print(f"GraphWaveNet {hname} val MAE={masked_metrics(extra['gwn'][:, k].astype(np.float64), true[:, k])['MAE']:.3f}")
    current = Xva[:, -1, :].astype(np.float32)  # vitesse observée à t0 (0 = manquant)
    np.savez_compressed(out, anchors=a_va, times=df.index[a_va].values.astype("datetime64[ns]"),
                        sensor_ids=np.array(sensor_ids), horizons=np.array(list(HORIZONS)),
                        lgbm=lgbm, ha=ha_pred, **extra, last_value=lv, true=true, current=current,
                        ha_free_flow=np.nanpercentile(np.where(ha.week > 0, ha.week, np.nan), 85, axis=0).astype(np.float32))
    print(f"[ok] {out} : {S} instants val x {N} capteurs x {len(hs)} horizons")


if __name__ == "__main__":
    main()
