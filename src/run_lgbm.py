"""LightGBM sur lags du capteur + voisins amont/aval, un modèle par horizon (3, 6, 12 pas).
Validation seulement (early stopping sur val -> chiffres val légèrement optimistes ; le test tranchera).
Usage : python -m src.run_lgbm [--no-neighbors] [--train-stride 3]"""
import argparse
import json
import os
import warnings

import lightgbm as lgb
try:
    import mlflow
except ImportError:  # tracking optionnel
    mlflow = None
import numpy as np
import pandas as pd

from src.data import chrono_split, load_metr_la, make_windows, time_features, window_anchors
from src.graphs import get_graph
from src.features import OOFHistoricalAverage, build_features, neighbor_mats
from src.leakage import check_splits, check_windows
from src.metrics import HORIZONS, masked_metrics

SEED = 42
N_IN, N_OUT = 12, 12
PARAMS = dict(objective="regression", metric="l1", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              seed=SEED, deterministic=True, force_row_wise=True, verbose=-1)


def main(h5, adj_path, out_dir, use_neighbors, train_stride, graph="real"):
    warnings.filterwarnings("ignore", category=RuntimeWarning)  # nanmean sur fenêtres vides -> NaN voulu
    np.random.seed(SEED)
    tag = "LightGBM" + ("" if use_neighbors else "_noNeighbors") + ("" if graph == "real" or not use_neighbors else f"_{graph}Graph")
    os.makedirs(os.path.join(out_dir, "models"), exist_ok=True)
    df, adj, _ = load_metr_la(h5, adj_path)
    values = df.values.astype(np.float64)
    T, N = values.shape
    splits = chrono_split(T); check_splits(splits, T)
    tr, va = splits[0], splits[1]
    slot, dow = time_features(df.index)
    week = np.arange(T) // (7 * 288)
    ha_feat = OOFHistoricalAverage().fit(values[tr.start:tr.end], slot[tr.start:tr.end], dow[tr.start:tr.end],
                                         week[tr.start:tr.end])
    print(f"[ok] HA-feature : train out-of-fold par semaine ({len(ha_feat.by_week)} semaines), val = HA du train complet")
    adj = get_graph(adj, graph, SEED)
    print(f"graphe '{graph}' : {int(((adj > 0) & ~np.eye(N, dtype=bool)).sum())} arêtes orientées hors diagonale")
    A_up, A_down = neighbor_mats(adj)

    a_tr = window_anchors(tr, N_IN, N_OUT)[::train_stride]
    a_va = window_anchors(va, N_IN, N_OUT)
    Xtr, Ytr, xi_tr, yi_tr = make_windows(values, a_tr, N_IN, N_OUT); check_windows(tr, xi_tr, yi_tr)
    Xva, Yva, xi_va, yi_va = make_windows(values, a_va, N_IN, N_OUT); check_windows(va, xi_va, yi_va)

    if mlflow is not None:
        mlflow.set_tracking_uri("sqlite:///mlflow.db")  # mlflow ui --backend-store-uri sqlite:///mlflow.db
        mlflow.set_experiment("dso4-metr-la")
        mlflow.start_run(run_name=tag)
        mlflow.log_params({**PARAMS, "train_stride": train_stride, "use_neighbors": use_neighbors, "graph": graph, "split": "val"})
    rows, importances, shap_imp = [], {}, {}
    for hname, h in HORIZONS.items():
        Ftr, names = build_features(Xtr, xi_tr, h, slot, dow, ha_feat, A_up, A_down, use_neighbors, week_all=week)
        Fva, _ = build_features(Xva, xi_va, h, slot, dow, ha_feat, A_up, A_down, use_neighbors)
        ytr, yva = Ytr[:, h - 1, :].reshape(-1), Yva[:, h - 1, :].reshape(-1)
        mtr, mva = ytr > 0, yva > 0                        # cibles manquantes exclues de l'entraînement
        cat = [names.index("sensor"), names.index("dow_target")]
        dtr = lgb.Dataset(Ftr[mtr], ytr[mtr], feature_name=names, categorical_feature=cat, free_raw_data=True)
        dva = lgb.Dataset(Fva[mva], yva[mva], reference=dtr)
        model = lgb.train(PARAMS, dtr, num_boost_round=3000, valid_sets=[dva],
                          callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
        pred = model.predict(Fva, num_iteration=model.best_iteration)
        m = masked_metrics(pred, yva)
        rows.append(dict(model=tag, split="val", horizon=hname, best_iter=model.best_iteration,
                         n_train_rows=int(mtr.sum()), **m))
        importances[hname] = dict(zip(names, model.feature_importance("gain").round(1).tolist()))
        # TreeSHAP (intégré à LightGBM) sur 20k lignes val tirées avec graine fixe : mean |SHAP| par feature
        sub = np.random.default_rng(SEED).choice(np.flatnonzero(mva), size=min(20000, int(mva.sum())), replace=False)
        contrib = model.predict(Fva[sub], num_iteration=model.best_iteration, pred_contrib=True)[:, :-1]
        shap_imp[hname] = dict(zip(names, np.abs(contrib).mean(axis=0).round(4).tolist()))
        if mlflow is not None:
            mlflow.log_metrics({f"val_{k}_{hname}": m[k] for k in ("MAE", "RMSE", "MAPE")})
        model.save_model(os.path.join(out_dir, "models", f"{tag}_{hname}.txt"), num_iteration=model.best_iteration)
        print(f"{tag} {hname}: MAE={m['MAE']:.3f} RMSE={m['RMSE']:.3f} MAPE={m['MAPE']:.2f}% "
              f"(best_iter={model.best_iteration}, {int(mtr.sum())} lignes train)")

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(out_dir, f"{tag}_val.csv"), index=False)
    with open(os.path.join(out_dir, f"{tag}_config.json"), "w") as f:
        json.dump(dict(params=PARAMS, train_stride=train_stride, use_neighbors=use_neighbors,
                       importance_gain=importances), f, indent=2)
    imp = pd.DataFrame(importances); imp = (imp / imp.sum()).round(3).sort_values("60min", ascending=False)
    print("\nimportance (gain, normalisée), top 10 :\n", imp.head(10))
    shp = pd.DataFrame(shap_imp).sort_values("60min", ascending=False)
    shp.to_csv(os.path.join(out_dir, f"{tag}_shap_val.csv"))
    print("\nmean |SHAP| (mph), top 10 :\n", shp.head(10).round(3))
    if mlflow is not None:
        for p in (f"{tag}_val.csv", f"{tag}_shap_val.csv", f"{tag}_config.json"):
            mlflow.log_artifact(os.path.join(out_dir, p))
        mlflow.end_run()
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", default="data/metr-la.h5")
    ap.add_argument("--adj", default="data/adj_mx.pkl")
    ap.add_argument("--out", default="results")
    ap.add_argument("--no-neighbors", action="store_true")
    ap.add_argument("--graph", choices=["real", "random", "none"], default="real",
                    help="ablation : vrai graphe, aléatoire même densité, ou aucune arête")
    ap.add_argument("--train-stride", type=int, default=3, help="1 fenêtre train sur k (mémoire)")
    a = ap.parse_args()
    main(a.h5, a.adj, a.out, not a.no_neighbors, a.train_stride, a.graph)
