"""Étape 1 : baselines HA et last value sous le protocole DSO4.
Usage : python -m src.run_baselines --h5 data/metr-la.h5 --adj data/adj_mx.pkl [--open-test]"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from src.baselines import HistoricalAverage, last_value
from src.data import MaskedScaler, chrono_split, load_metr_la, make_windows, time_features, window_anchors
from src.leakage import TestGuard, check_index_time, check_scaler, check_splits, check_windows
from src.metrics import horizon_table

SEED = 42
N_IN, N_OUT = 12, 12


def main(h5, adj_path, out_dir, open_test):
    np.random.seed(SEED)
    os.makedirs(out_dir, exist_ok=True)
    df, adj, ids = load_metr_la(h5, adj_path)
    values = df.values.astype(np.float64)
    T, N = values.shape
    print(f"données : T={T} pas, N={N} capteurs, {df.index[0]} -> {df.index[-1]}")
    print(f"taux de 0 (manquants) : {(values == 0).mean():.2%}")
    print(f"graphe : {int((adj > 0).sum())} entrées non nulles (dont {int((np.diag(adj) > 0).sum())} sur la diagonale), "
          f"symétrique={np.allclose(adj, adj.T)}")

    splits = chrono_split(T)
    check_splits(splits, T)
    check_index_time(df.index, splits)
    train = splits[0]

    scaler = MaskedScaler().fit(values[train.start:train.end])
    check_scaler(scaler, values, train)  # non utilisé par HA / last value, prêt pour LightGBM et GNN

    slot, dow = time_features(df.index)
    ha = HistoricalAverage().fit(values[train.start:train.end], slot[train.start:train.end], dow[train.start:train.end])
    print(f"[ok] HA ajustée sur le train seul ; cases (capteur x créneau semaine) sans donnée -> repli : {ha.n_fallback}")

    eval_splits = ["val"] + (["test"] if open_test else [])
    if open_test:
        TestGuard(os.path.join(out_dir, "TEST_OPENED.lock")).open("baselines HA + last value")

    rows = []
    for sp in splits:
        if sp.name not in eval_splits:
            continue
        anchors = window_anchors(sp, N_IN, N_OUT)
        X, Y, x_idx, y_idx = make_windows(values, anchors, N_IN, N_OUT)
        check_windows(sp, x_idx, y_idx)

        p_ha = ha.predict(y_idx, slot, dow)
        rows += horizon_table(p_ha, Y, "HistoricalAverage", sp.name)

        last, frac_empty = last_value(X, fallback=ha.sensor)
        p_lv = np.repeat(last[:, None, :], N_OUT, axis=1)
        rows += horizon_table(p_lv, Y, "LastValue", sp.name)
        print(f"[info] {sp.name}: fenêtres d'entrée entièrement manquantes (repli moyenne train) : {frac_empty:.2%}")

        # canari : un oracle qui lit la cible doit donner 0, sinon le masque/les métriques sont faux
        assert horizon_table(Y.copy(), Y, "oracle", sp.name)[0]["MAE"] == 0.0

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(out_dir, f"baselines_{'_'.join(eval_splits)}.csv"), index=False)
    with open(os.path.join(out_dir, "baselines_config.json"), "w") as f:
        json.dump(dict(seed=SEED, n_in=N_IN, n_out=N_OUT, ratios=[0.7, 0.1, 0.2],
                       scaler=dict(mean=scaler.mean, std=scaler.std),
                       splits={s.name: [s.start, s.end] for s in splits}), f, indent=2)
    pd.set_option("display.precision", 3); pd.set_option("display.width", 250); pd.set_option("display.max_columns", None)
    print(res.pivot_table(index=["split", "model"], columns="horizon", values=["MAE", "RMSE", "MAPE"])
          .reindex(columns=["15min", "30min", "60min"], level=1))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", default="data/metr-la.h5")
    ap.add_argument("--adj", default="data/adj_mx.pkl")
    ap.add_argument("--out", default="results")
    ap.add_argument("--open-test", action="store_true")
    a = ap.parse_args()
    main(a.h5, a.adj, a.out, a.open_test)
