"""Chargement METR-LA, split chronologique, fenêtres sans chevauchement, scaler train-only."""
import pickle
from dataclasses import dataclass

import numpy as np
import pandas as pd
import tables

STEPS_PER_DAY = 288  # 5 min


def _read_pandas_hdf_fallback(h5_path):
    """Lit les fichiers HDF pandas générés par le dataset DCRNN sans passer par pandas.read_hdf,
    car cette voie casse avec pandas 3.x + tables 3.11 sur la version de fichier téléchargée."""
    with tables.open_file(h5_path, mode="r") as store:
        candidates = []
        if hasattr(store.root, "_v_children"):
            candidates.extend(store.root._v_children.values())
        candidates.append(store.root)

        for node in candidates:
            if not hasattr(node, "axis0") or not hasattr(node, "axis1") or not hasattr(node, "block0_values"):
                continue
            axis0 = np.asarray(node.axis0[:])
            axis1 = np.asarray(node.axis1[:])
            values = np.asarray(node.block0_values[:])
            columns = [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in axis0]
            index = pd.to_datetime(axis1, unit="ns")
            return pd.DataFrame(values, index=index, columns=columns)

    raise ValueError(f"Impossible de lire le fichier HDF {h5_path!r}: structure inconnue")


def load_metr_la(h5_path, adj_path):
    """Retourne (df vitesses [T, N], adj [N, N], sensor_ids) avec colonnes alignées sur l'ordre du graphe."""
    try:
        df = pd.read_hdf(h5_path)
    except TypeError:
        df = _read_pandas_hdf_fallback(h5_path)

    with open(adj_path, "rb") as f:
        sensor_ids, sensor_id_to_ind, adj = pickle.load(f, encoding="latin1")

    sensor_ids = [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in sensor_ids]
    cols = [str(c) for c in df.columns]
    df.columns = cols
    assert set(cols) == set(sensor_ids), "capteurs du .h5 et du graphe différents"
    df = df[sensor_ids]  # même ordre que la matrice d'adjacence
    assert df.index.is_monotonic_increasing, "index temporel non trié"
    step = df.index.to_series().diff().dropna().unique()
    assert len(step) == 1 and step[0] == np.timedelta64(5, "m"), f"pas de temps irrégulier: {step[:5]}"
    return df, np.asarray(adj, dtype=np.float32), list(sensor_ids)


@dataclass
class Split:
    name: str
    start: int  # index inclus
    end: int    # index exclu


def chrono_split(T, ratios=(0.7, 0.1, 0.2)):
    assert abs(sum(ratios) - 1) < 1e-9
    n_train = int(round(T * ratios[0]))
    n_val = int(round(T * ratios[1]))
    return [Split("train", 0, n_train),
            Split("val", n_train, n_train + n_val),
            Split("test", n_train + n_val, T)]


def window_anchors(split, n_in=12, n_out=12):
    """Indices t0 = dernier pas observé. Entrée [t0-n_in+1, t0], cible [t0+1, t0+n_out].
    Les deux doivent rester DANS le bloc : aucune fenêtre ne traverse une frontière."""
    first = split.start + n_in - 1
    last = split.end - n_out - 1
    return np.arange(first, last + 1)


def make_windows(values, anchors, n_in=12, n_out=12):
    """values [T, N] -> X [S, n_in, N], Y [S, n_out, N] (vue indexée, pas de copie du passé/futur hors fenêtre)."""
    x_idx = anchors[:, None] + np.arange(-n_in + 1, 1)[None, :]
    y_idx = anchors[:, None] + np.arange(1, n_out + 1)[None, :]
    return values[x_idx], values[y_idx], x_idx, y_idx


class MaskedScaler:
    """z-score calculé sur le train uniquement, en ignorant les 0 (= manquants)."""

    def fit(self, train_values):
        v = train_values[train_values > 0]
        self.mean, self.std = float(v.mean()), float(v.std())
        return self

    def transform(self, x):
        return (x - self.mean) / self.std

    def inverse(self, x):
        return x * self.std + self.mean


def time_features(index):
    """slot de la journée (0..287) et jour de la semaine (0..6)."""
    slot = (index.hour * 60 + index.minute) // 5
    return np.asarray(slot), np.asarray(index.dayofweek)
