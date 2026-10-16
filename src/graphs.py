"""Graphes pour l'ablation : vrai graphe routier, graphe aléatoire de même densité, aucune arête."""
import numpy as np


def random_same_density(adj, seed=42):
    """Même nombre d'arêtes orientées hors diagonale, mêmes poids (permutés), positions tirées au hasard.
    La diagonale (self-loops) est conservée telle quelle."""
    rng = np.random.default_rng(seed)
    N = adj.shape[0]
    off = ~np.eye(N, dtype=bool)
    weights = adj[off & (adj > 0)]
    cand = np.flatnonzero(off.ravel())
    pick = rng.choice(cand, size=len(weights), replace=False)
    R = np.zeros_like(adj)
    R.ravel()[pick] = rng.permutation(weights)
    R[np.diag_indices(N)] = np.diag(adj)
    return R


def no_edges(adj):
    R = np.zeros_like(adj)
    R[np.diag_indices(adj.shape[0])] = np.diag(adj)
    return R


def get_graph(adj, kind, seed=42):
    if kind == "real":
        return adj
    if kind == "random":
        return random_same_density(adj, seed)
    if kind == "none":
        return no_edges(adj)
    raise ValueError(kind)
