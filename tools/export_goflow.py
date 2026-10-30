"""Sortie de GO-FLOW (DSO de la collègue) -> entrée de DSO4.
Fait tourner SON modèle XGBoost (son code, chargé tel quel depuis son dossier, rien n'est modifié ni réentraîné)
sur tout PEMS08 et sauvegarde sa prévision d'occupation à +15 min pour chaque capteur et chaque pas de 5 min.

Lancement depuis la racine dso4, dans un Python qui a xgboost==3.2.0 :
    python tools/export_goflow.py --goflow ..\\yassminet\\GO-FLOW
Sortie : results/pems08/goflow/goflow_outputs.npz
    pred_occ15 [T, N] : forecast.predicted_occupancy de GO-FLOW faite à t (NaN pour t < 13 : il lui faut 1 h)
    occupancy, flow, speed [T, N] : traffic_data courant (son CSV)
    thresholds : ses seuils train-only (occupancy_high, occupancy_severe, ...)
Ce script n'importe PAS le paquet src de DSO4 : les deux projets s'appellent « src »."""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

LAGS, WINDOWS = (1, 2, 3, 6, 12), (3, 6, 12)


def features_all(values, t_idx, s):
    """Features GO-FLOW (ordre de son export) pour le capteur s aux indices t_idx (0-based, t_idx >= 12),
    calculées en vectoriel exactement comme son build_features (lags, shift(1).rolling, float64 -> float32)."""
    ts = t_idx + 1                                    # son timestep : origine 1
    rel = ts - 1
    slot = rel % 288
    cols = {"location": np.full(len(t_idx), s), "day_index": rel // 288, "slot_in_day": slot,
            "hour": slot * 5 // 60, "minute": slot * 5 % 60}
    for name in ("flow", "speed", "occupy"):
        cols[name] = values[name][t_idx, s]
    for name in ("flow", "speed", "occupy"):
        for k in LAGS:
            cols[f"{name}_lag_{k}"] = values[name][t_idx - k, s]
        for w in WINDOWS:
            cols[f"{name}_rolling_mean_{w}"] = np.mean(
                np.stack([values[name][t_idx - j, s] for j in range(1, w + 1)]), axis=0)
    return cols


def main(goflow, out):
    goflow = os.path.abspath(goflow)
    sys.path.insert(0, goflow)
    from src.prediction import XGBoostTrafficPredictor              # code GO-FLOW
    from src.prediction.xgboost_predictor import FEATURE_COLUMNS    # code GO-FLOW
    pred = XGBoostTrafficPredictor(goflow)                          # charge et vérifie son modèle sauvegardé
    model, it_range = pred._model, pred._iteration_range

    df = pd.read_csv(os.path.join(goflow, "data", "raw", "traffic_dataset.csv"))
    T, N = int(df.timestep.max()), int(df.location.max()) + 1
    assert len(df) == T * N, "CSV GO-FLOW incomplet"
    df = df.sort_values(["timestep", "location"])
    values = {c: df[c].to_numpy(np.float64).reshape(T, N) for c in ("flow", "speed", "occupy")}

    t_idx = np.arange(12, T)                                        # il faut t-12..t
    out_pred = np.full((T, N), np.nan, np.float32)
    t0 = time.time()
    for s in range(N):
        F = pd.DataFrame(features_all(values, t_idx, s))[list(FEATURE_COLUMNS)]
        F["location"] = F["location"].astype(pd.CategoricalDtype(categories=list(range(170))))
        num = list(FEATURE_COLUMNS[1:])
        F[num] = F[num].astype("float32")
        out_pred[t_idx, s] = model.predict(F, iteration_range=it_range)
        if s % 34 == 0:
            print(f"capteur {s}/{N} ({time.time() - t0:.0f} s)", flush=True)

    # contrôle : sur 300 couples (t, capteur) tirés au hasard, notre calcul groupé == son predict() officiel
    rng = np.random.default_rng(0)
    errs = []
    for t, s in zip(rng.integers(12, T, 300), rng.integers(0, N, 300)):
        hist = pd.DataFrame(dict(location=s, timestep=np.arange(t - 11, t + 2),
                                 flow=values["flow"][t - 12:t + 1, s], speed=values["speed"][t - 12:t + 1, s],
                                 occupy=values["occupy"][t - 12:t + 1, s]))
        hist[["location", "timestep"]] = hist[["location", "timestep"]].astype("int64")
        errs.append(abs(pred.predict(hist, sensor_id=int(s), timestep=int(t + 1)) - out_pred[t, s]))
    print(f"[ok] contrôle contre son predict() officiel : écart max {max(errs):.2e} sur 300 cas")
    assert max(errs) < 1e-5

    meth = open(os.path.join(goflow, "results", "decision_agent", "decision_agent_methodology.md"),
                encoding="utf-8").read()
    thresholds = json.loads(meth.split("```json", 1)[1].split("```", 1)[0])["thresholds_train_only"]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out, pred_occ15=out_pred, occupancy=values["occupy"].astype(np.float32),
                        flow=values["flow"].astype(np.float32), speed=values["speed"].astype(np.float32),
                        thresholds=json.dumps(thresholds))
    print(f"[ok] {out} : {T} pas x {N} capteurs ; seuils GO-FLOW : {thresholds}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--goflow", default=os.path.join("..", "yassminet", "GO-FLOW"))
    ap.add_argument("--out", default=os.path.join("results", "pems08", "goflow", "goflow_outputs.npz"))
    a = ap.parse_args()
    main(a.goflow, a.out)
