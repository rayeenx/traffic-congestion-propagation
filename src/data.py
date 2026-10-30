"""Chargement METR-LA ou PEMS08, split chronologique, fenêtres sans chevauchement, scaler train-only."""
import os
import pickle
from dataclasses import dataclass

import numpy as np
import pandas as pd

STEPS_PER_DAY = 288  # 5 min

# Chemins par défaut et dossier de résultats de chaque dataset (PEMS08 à part pour ne pas écraser METR-LA).
DATASETS = {
    "metr-la": dict(data="data/metr-la.h5", adj="data/adj_mx.pkl", out="results",
                    label="METR-LA", city="Los Angeles", gps=True),
    "pems08": dict(data="data/pems08/pems08.npz", adj="data/pems08/distance.csv", out="results/pems08",
                   label="PEMS08", city="San Bernardino", gps=False),
}


def active_dataset():
    """Dataset utilisé par l'agent et le dashboard : variable d'environnement DSO4_DATASET (défaut metr-la)."""
    name = os.environ.get("DSO4_DATASET", "metr-la").lower()
    assert name in DATASETS, f"DSO4_DATASET={name!r} inconnu, choisir parmi {sorted(DATASETS)}"
    return name


def results_dir(name=None):
    return DATASETS[name or active_dataset()]["out"]
# PEMS08 (fichiers ASTGCN, les mêmes que GO-FLOW) ne contient pas de dates. La littérature donne
# juillet-août 2016 : on suppose un début le 01/07/2016 00:00, seulement pour l'heure et le jour de semaine.
PEMS08_START = "2016-07-01"
# Mêmes frontières que GO-FLOW : train = timesteps 1..12500, val = 12501..15176, test = 15177..17856.
PEMS08_SPLIT_ENDS = (12500, 15176)


def _read_pandas_hdf_fallback(h5_path):
    """Lit les fichiers HDF pandas générés par le dataset DCRNN sans passer par pandas.read_hdf,
    car cette voie casse avec pandas 3.x + tables 3.11 sur la version de fichier téléchargée."""
    import tables  # seulement pour METR-LA

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


def load_pems08(npz_path, dist_path, threshold=0.0):
    """Retourne (df vitesses [T, N], adj [N, N], sensor_ids) comme load_metr_la.
    Vitesse = canal 2 du npz (flow=0, occupy=1, speed=2), en mph. sensor_id = index 0..169 du npz,
    le même que `location` / `sensor_id` chez GO-FLOW.
    Graphe : noyau gaussien de DCRNN (gen_adj_mx.py) sur distance.csv, sens des arêtes conservé
    (from -> to), w = exp(-(d / sigma)^2), sigma = écart-type des distances, diagonale à 1 comme adj_mx.pkl.
    Pas de seuil par défaut : distance.csv ne liste que des capteurs voisins, et le seuil 0.1 de DCRNN
    enlèverait la moitié des arêtes (48 capteurs isolés)."""
    speed = np.load(npz_path)["data"][..., 2].astype(np.float64)
    T, N = speed.shape
    sensor_ids = [str(i) for i in range(N)]
    index = pd.date_range(PEMS08_START, periods=T, freq="5min")
    df = pd.DataFrame(speed, index=index, columns=sensor_ids)

    edges = pd.read_csv(dist_path)
    edges = edges.groupby(["from", "to"], as_index=False)["cost"].min()  # arêtes en double -> plus courte
    assert edges[["from", "to"]].to_numpy().max() < N, "id d'arête hors des capteurs du npz"
    dist = np.full((N, N), np.inf)
    dist[edges["from"].to_numpy(), edges["to"].to_numpy()] = edges["cost"].to_numpy()
    sigma = edges["cost"].std()
    adj = np.exp(-np.square(dist / sigma))
    if threshold:
        adj[adj < threshold] = 0.0
    np.fill_diagonal(adj, 1.0)
    return df, adj.astype(np.float32), sensor_ids


def pems08_time(timestep):
    """timestep GO-FLOW (origine 1, pas de 5 min) -> horodatage utilisé par DSO4."""
    return pd.Timestamp(PEMS08_START) + pd.Timedelta(minutes=5 * (int(timestep) - 1))


def schematic_layout(dist_path, n):
    """PEMS08 n'a pas de GPS : positions 2D calculées à partir des seules distances du graphe
    (plus courts chemins puis MDS classique). Ce n'est PAS une carte : seulement un schéma du réseau.
    Colonnes latitude/longitude gardées pour réutiliser le code du dashboard (vue « sans fond »)."""
    e = pd.read_csv(dist_path)
    D = np.full((n, n), np.inf)
    np.fill_diagonal(D, 0.0)
    for f, t, c in e[["from", "to", "cost"]].itertuples(index=False):
        D[f, t] = D[t, f] = min(D[f, t], c)
    for k in range(n):  # Floyd-Warshall (170 noeuds : instantané)
        D = np.minimum(D, D[:, [k]] + D[[k], :])
    D[np.isinf(D)] = np.nanmax(D[np.isfinite(D)]) * 1.5  # composantes séparées : écartées
    J = np.eye(n) - 1.0 / n
    w, v = np.linalg.eigh(-0.5 * J @ (D ** 2) @ J)
    xy = v[:, -2:] * np.sqrt(np.maximum(w[-2:], 0))
    xy = (xy - xy.mean(0)) / np.abs(xy).max()
    return pd.DataFrame(dict(sensor_id=[str(i) for i in range(n)], latitude=xy[:, 0], longitude=xy[:, 1],
                             schematic=True))


def load_dataset(name, data_path=None, adj_path=None):
    """Point d'entrée unique : name = 'metr-la' ou 'pems08' ; chemins par défaut dans DATASETS."""
    cfg = DATASETS[name]
    data_path, adj_path = data_path or cfg["data"], adj_path or cfg["adj"]
    if name == "pems08":
        return load_pems08(data_path, adj_path)
    return load_metr_la(data_path, adj_path)


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


def dataset_splits(name, T):
    """METR-LA : 70/10/20. PEMS08 : frontières de GO-FLOW (environ 70/15/15) pour comparer avec elle."""
    if name == "pems08":
        a, b = PEMS08_SPLIT_ENDS
        assert T > b
        return [Split("train", 0, a), Split("val", a, b), Split("test", b, T)]
    return chrono_split(T)


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
