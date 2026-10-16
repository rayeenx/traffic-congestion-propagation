"""Features tabulaires pour LightGBM. Une ligne = (fenêtre, capteur). Tout est calculé à partir de
l'entrée [t0-11, t0] ou de statistiques du train ; rien ne lit la cible."""
import numpy as np

from src.baselines import HistoricalAverage


def neighbor_mats(adj):
    """adj[i, j] > 0 : arête i -> j (DCRNN : distance routière de i vers j).
    amont de k = {i : adj[i,k] > 0}, aval de k = {j : adj[k,j] > 0}. Diagonale retirée."""
    A = adj.astype(np.float64).copy()
    np.fill_diagonal(A, 0.0)
    return A, A.T.copy()  # (X @ A_up)[k] agrège l'amont de k ; (X @ A_down)[k] l'aval


def weighted_neighbor_mean(X, A):
    """X [S, L, N] (0 = manquant) -> moyenne pondérée des voisins valides [S, L, N], NaN si aucun."""
    valid = (X > 0).astype(np.float64)
    num = np.einsum("sln,nk->slk", X * valid, A)
    den = np.einsum("sln,nk->slk", valid, A)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


class OOFHistoricalAverage:
    """HA comme feature, sans fuite de cible dans le train : pour une ligne du train dont la cible tombe
    dans la semaine w, la HA est recalculée sans la semaine w (out-of-fold). Val/test : HA du train complet."""

    def fit(self, train_values, slot, dow, week):
        self.full = HistoricalAverage().fit(train_values, slot, dow)
        self.by_week = {w: HistoricalAverage().fit(train_values[week != w], slot[week != w], dow[week != w])
                        for w in np.unique(week)}
        return self

    def predict(self, y_flat, slot_all, dow_all, week_all=None):
        if week_all is None:
            return self.full.predict_at(slot_all[y_flat], dow_all[y_flat])
        out = np.empty((len(y_flat), self.full.week.shape[1]))
        w = week_all[y_flat]
        for wk, ha in self.by_week.items():
            sel = w == wk
            if sel.any():
                out[sel] = ha.predict_at(slot_all[y_flat[sel]], dow_all[y_flat[sel]])
        return out


def build_features(X, x_idx, h, slot, dow, ha_feat, A_up, A_down, use_neighbors=True, week_all=None):
    """X [S, 12, N] vitesses brutes. h = horizon en pas (1..12). Retourne (F [S*N, d], noms)."""
    S, L, N = X.shape
    t0 = x_idx[:, -1]
    tgt = t0 + h
    own = np.where(X > 0, X, np.nan)                     # [S, L, N]
    cols, names = [], []
    for lag in range(L):                                  # lag 0 = t0
        cols.append(own[:, L - 1 - lag, :]); names.append(f"own_lag{lag}")
    cols.append(np.nanmean(own, axis=1)); names.append("own_mean12")
    cols.append(np.isnan(own).mean(axis=1)); names.append("own_frac_missing")
    if use_neighbors:
        up, down = weighted_neighbor_mean(X, A_up), weighted_neighbor_mean(X, A_down)
        for lag in range(3):
            cols.append(up[:, L - 1 - lag, :]); names.append(f"up_lag{lag}")
            cols.append(down[:, L - 1 - lag, :]); names.append(f"down_lag{lag}")
        cols.append(np.nanmean(up, axis=1)); names.append("up_mean12")
        cols.append(np.nanmean(down, axis=1)); names.append("down_mean12")
    ha = ha_feat.predict(tgt, slot, dow, week_all).reshape(S, N)
    cols.append(ha); names.append("ha_target")
    cols.append(np.repeat(slot[tgt][:, None], N, 1)); names.append("slot_target")
    cols.append(np.repeat(dow[tgt][:, None], N, 1)); names.append("dow_target")
    cols.append(np.repeat(np.arange(N)[None, :], S, 0)); names.append("sensor")
    F = np.stack([c.astype(np.float32) for c in cols], axis=-1).reshape(S * N, -1)
    return F, names
