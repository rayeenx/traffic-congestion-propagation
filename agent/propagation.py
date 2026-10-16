"""Analyse de la propagation RÉELLE des bouchons sur la validation, et de l'anticipation de l'agent.
- début de bouchon d'un capteur : vitesse < 50 % de la vitesse fluide pendant 15 min (3 pas), alors qu'il roulait
  normalement (≥ 75 %) en moyenne sur les 30 min d'avant ;
- arbre de propagation : chaque capteur touché est rattaché au voisin routier (graphe DCRNN) touché juste avant lui
  (au plus 30 min avant). Les capteurs sans parent sont des origines (« patients zéro ») ;
- sens : arête i -> j du graphe = la circulation va de i vers j. Si l'enfant est en amont du parent, le bouchon
  « remonte » la route (sens inverse de la circulation) ;
- vitesse de l'onde : distance à vol d'oiseau entre les deux capteurs / écart des heures de début (sous-estime la
  distance routière, donc la vitesse)."""
from collections import Counter

import numpy as np
import pandas as pd

from agent.core import HEAVY, MODERATE


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def ratios(d):
    return np.where(d["current"] > 0, d["current"] / d["ha_free_flow"], np.nan)


def onsets_in(r, sel):
    T, N = r.shape
    cong = r < HEAVY
    onset = np.full(N, -1)
    for s in range(N):
        for t in sel[1:-2]:
            if cong[t:t + 3, s].all() and not cong[t - 1, s] and np.nanmean(r[max(0, t - 6):t, s]) >= MODERATE:
                onset[s] = t
                break
    return onset


def episode(d, adj, loc, start, end, max_gap=6):
    times = pd.to_datetime(d["times"])
    r = ratios(d)
    N = r.shape[1]
    sel = np.flatnonzero((times >= start) & (times <= end))
    onset = onsets_in(r, sel)
    A = (adj > 0) & ~np.eye(N, dtype=bool)
    hit = np.flatnonzero(onset >= 0)
    parent = {}
    for s in hit:
        nb = [n for n in np.flatnonzero(A[s] | A[:, s]) if onset[n] >= 0 and 0 < onset[s] - onset[n] <= max_gap]
        if nb:
            parent[s] = max(nb, key=lambda n: onset[n])
    lat, lon = loc.latitude.values, loc.longitude.values
    edges = []
    for s, p in parent.items():
        up, down = bool(A[s, p]), bool(A[p, s])
        sens = "remonte (amont)" if up and not down else ("descend (aval)" if down and not up else "double sens")
        dist = float(haversine_km(lat[p], lon[p], lat[s], lon[s]))
        dt_min = (onset[s] - onset[p]) * 5
        edges.append(dict(parent=int(p), child=int(s), dist_km=dist, dt_min=dt_min,
                          speed_kmh=dist / (dt_min / 60), sens=sens))

    def root(s):
        while s in parent:
            s = parent[s]
        return s

    roots = Counter(root(s) for s in hit)
    return dict(onset=onset, hit=hit, parent=parent, edges=pd.DataFrame(edges), roots=roots, times=times)


def lead_times(d, agent, horizon_max=60):
    """Pour chaque début de bouchon réel (toute la validation) : l'agent avait-il alerté ce capteur dans l'heure
    avant, et combien de minutes avant ?"""
    times = pd.to_datetime(d["times"])
    r = ratios(d)
    onset_all = []
    cong = r < HEAVY
    for s in range(r.shape[1]):
        for t in range(6, len(times) - 2):
            if cong[t:t + 3, s].all() and not cong[t - 1, s] and np.nanmean(r[t - 6:t, s]) >= MODERATE:
                onset_all.append((t, s))
    al = pd.DataFrame([dict(t=pd.Timestamp(a.time), s=a.sensor) for a in agent.alerts])
    rows = []
    for t, s in onset_all:
        ot = times[t]
        m = al[(al.s == s) & (al.t < ot) & (al.t >= ot - pd.Timedelta(minutes=horizon_max))]
        rows.append(dict(t=t, sensor=s, onset=ot, alerted=len(m) > 0,
                         lead_min=(ot - m.t.min()).total_seconds() / 60 if len(m) else np.nan))
    return pd.DataFrame(rows)
