"""Baselines sans apprentissage profond. Tout ce qui est 'appris' l'est sur le train uniquement."""
import numpy as np


class HistoricalAverage:
    """Moyenne par (capteur, jour de semaine, créneau 5 min), calculée sur le train en ignorant les 0.
    Repli : moyenne par créneau tous jours confondus, puis moyenne du capteur."""

    def fit(self, train_values, slot, dow):
        T, N = train_values.shape
        valid = train_values > 0
        v = np.where(valid, train_values, 0.0)

        def group_mean(key, n_keys):
            s = np.zeros((n_keys, N)); c = np.zeros((n_keys, N))
            np.add.at(s, key, v); np.add.at(c, key, valid)
            with np.errstate(invalid="ignore", divide="ignore"):
                return np.where(c > 0, s / c, np.nan)

        self.week = group_mean(dow * 288 + slot, 7 * 288)
        self.day = group_mean(slot, 288)
        self.sensor = group_mean(np.zeros(T, dtype=int), 1)[0]
        self.n_fallback = int(np.isnan(self.week).sum())
        return self

    def predict_at(self, slot, dow):
        """slot, dow : tableaux de même forme (n,) -> [n, N]"""
        p = self.week[dow * 288 + slot]
        p = np.where(np.isnan(p), self.day[slot], p)
        p = np.where(np.isnan(p), self.sensor[None, :], p)
        return p

    def predict(self, y_idx, slot_all, dow_all):
        """y_idx [S, n_out] -> [S, n_out, N] (ne dépend que de l'heure cible, pas des observations)."""
        S, H = y_idx.shape
        flat = y_idx.ravel()
        return self.predict_at(slot_all[flat], dow_all[flat]).reshape(S, H, -1)


def last_value(X, fallback):
    """Dernière vitesse non nulle observée DANS la fenêtre d'entrée, répétée sur tous les horizons.
    X [S, n_in, N]. fallback [N] (ex. moyenne train du capteur) si la fenêtre est entièrement manquante."""
    S, n_in, N = X.shape
    valid = X > 0
    # position du dernier pas valide dans la fenêtre (n_in-1 = le plus récent)
    rev = valid[:, ::-1, :]
    has = rev.any(axis=1)
    pos = n_in - 1 - rev.argmax(axis=1)
    last = np.take_along_axis(X, pos[:, None, :], axis=1)[:, 0, :]
    last = np.where(has, last, fallback[None, :])
    return last, (~has).mean()
