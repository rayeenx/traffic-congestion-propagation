"""Vérifications explicites d'absence de fuite. Toutes lèvent AssertionError si violées."""
import numpy as np


def check_splits(splits, T):
    assert splits[0].start == 0 and splits[-1].end == T
    for a, b in zip(splits, splits[1:]):
        assert a.end == b.start, "trou ou chevauchement entre blocs"
    print("[ok] blocs contigus, disjoints, ordonnés :",
          ", ".join(f"{s.name}=[{s.start},{s.end})" for s in splits))


def check_windows(split, x_idx, y_idx):
    assert x_idx.min() >= split.start and y_idx.max() < split.end, f"fenêtre hors du bloc {split.name}"
    assert (x_idx.max(axis=1) < y_idx.min(axis=1)).all(), "entrée et cible se chevauchent"
    print(f"[ok] {split.name}: {len(x_idx)} fenêtres, toutes dans [{split.start},{split.end}), "
          f"entrée strictement avant la cible")


def check_scaler(scaler, values, train_split):
    tr = values[train_split.start:train_split.end]
    v = tr[tr > 0]
    assert np.isclose(scaler.mean, v.mean()) and np.isclose(scaler.std, v.std())
    full = values[values > 0]
    print(f"[ok] scaler = stats du train seul (mean={scaler.mean:.3f}, std={scaler.std:.3f}); "
          f"pour info, stats sur tout = ({full.mean():.3f}, {full.std():.3f}) -> non utilisées")


def check_index_time(index, splits):
    tr, va, te = splits
    assert index[tr.end - 1] < index[va.start] and index[va.end - 1] < index[te.start]
    print(f"[ok] dates : train {index[tr.start]} -> {index[tr.end-1]} | "
          f"val {index[va.start]} -> {index[va.end-1]} | test {index[te.start]} -> {index[te.end-1]}")


class TestGuard:
    """Le test ne s'ouvre qu'une fois : un fichier verrou est créé au premier accès."""

    def __init__(self, lock_path):
        self.lock_path = lock_path

    def open(self, reason):
        import os, datetime
        if os.path.exists(self.lock_path):
            raise RuntimeError(f"Test déjà ouvert (voir {self.lock_path}). Supprimer le verrou = tricher.")
        with open(self.lock_path, "w") as f:
            f.write(f"{datetime.datetime.now().isoformat()} {reason}\n")
        return True
