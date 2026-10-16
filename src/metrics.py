"""MAE / RMSE / MAPE masquées : toute cible == 0 (manquante) est exclue."""
import numpy as np

HORIZONS = {"15min": 3, "30min": 6, "60min": 12}


def masked_metrics(pred, true, null_val=0.0):
    mask = true != null_val
    n = mask.sum()
    if n == 0:
        return dict(MAE=np.nan, RMSE=np.nan, MAPE=np.nan, n=0)
    e = (pred - true)[mask]
    return dict(MAE=float(np.abs(e).mean()),
                RMSE=float(np.sqrt((e ** 2).mean())),
                MAPE=float((np.abs(e) / np.abs(true[mask])).mean() * 100),
                n=int(n))


def horizon_table(pred, true, model_name, split_name):
    """pred, true : [S, 12, N] en km/h... (mph pour METR-LA). Une ligne par horizon."""
    rows = []
    for hname, h in HORIZONS.items():
        m = masked_metrics(pred[:, h - 1], true[:, h - 1])
        rows.append(dict(model=model_name, split=split_name, horizon=hname, **m))
    return rows
