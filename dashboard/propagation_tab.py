"""Onglet « Propagation » : d'où vient un bouchon réel, dans quel ordre il touche les voisins, à quelle vitesse,
et combien de minutes avant l'agent l'avait annoncé."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from agent.propagation import episode, lead_times

PERIODES = {"Matin (5 h – 11 h)": (5, 11), "Soir (13 h – 20 h)": (13, 20)}


@st.cache_resource(show_spinner="Analyse de la propagation sur toute la validation…")
def global_stats(_d, _adj, _loc, _agent, days):
    edges = []
    for day in days:
        for h0, h1 in PERIODES.values():
            e = episode(_d, _adj, _loc, day + pd.Timedelta(hours=h0), day + pd.Timedelta(hours=h1))
            edges.append(e["edges"])
    E = pd.concat(edges, ignore_index=True)
    L = lead_times(_d, _agent)
    return E, L


@st.cache_resource(show_spinner="Reconstruction de l'épisode…")
def get_episode(_d, _adj, _loc, day, periode):
    h0, h1 = PERIODES[periode]
    return episode(_d, _adj, _loc, day + pd.Timedelta(hours=h0), day + pd.Timedelta(hours=h1))


def render(d, adj, loc, agent, days, tiles):
    sids = d["sensor_ids"]
    st.markdown("### D'où vient le bouchon, et comment il se propage ?\n"
                "Ici, ce ne sont **pas des prévisions** : ce sont les **vraies données** de la validation. On retrouve "
                "le capteur où le bouchon commence, puis l'ordre dans lequel ses voisins sur la route sont touchés.")
    c1, c2 = st.columns(2)
    day = c1.selectbox("Journée", days, format_func=lambda x: x.strftime("%A %d/%m/%Y"), key="prop_day",
                       index=next((j for j, dd in enumerate(days) if dd == pd.Timestamp("2012-05-24")), 0))
    periode = c2.radio("Période", list(PERIODES), index=1, horizontal=True, key="prop_per")
    ep = get_episode(d, adj, loc, day, periode)
    onset, parent, edges, roots, times = ep["onset"], ep["parent"], ep["edges"], ep["roots"], ep["times"]
    hit = ep["hit"]
    if len(hit) == 0:
        st.info("Aucun début de bouchon détecté sur cette période.")
        return

    main_root, n_main = roots.most_common(1)[0]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Capteurs touchés", len(hit))
    m2.metric("Origine du plus gros bouchon", f"{sids[main_root]}",
              help=f"commence à {times[onset[main_root]]:%H:%M}, se propage à {n_main - 1} voisins")
    if len(edges):
        up = (edges.sens == "remonte (amont)").mean()
        m3.metric("Liens qui remontent la route", f"{up:.0%}", help="le bouchon va dans le sens inverse de la circulation")
        m4.metric("Vitesse de l'onde (médiane)", f"{edges.speed_kmh.median():.1f} km/h",
                  help="distance à vol d'oiseau / écart de temps : sous-estime la vitesse réelle")

    # ---- carte : couleur = heure de début du bouchon, traits = propagation ----
    t_min = np.array([(times[onset[s]] - times[onset[hit].min()]).total_seconds() / 60 if onset[s] >= 0 else np.nan
                      for s in range(len(sids))])
    lat, lon = loc.latitude.values, loc.longitude.values
    T = go.Scattermap if tiles else go.Scatter
    xy = (lambda la, lo: dict(lat=la, lon=lo)) if tiles else (lambda la, lo: dict(x=lo, y=la))
    fig = go.Figure()
    other = [s for s in range(len(sids)) if onset[s] < 0]
    fig.add_trace(T(**xy(lat[other].tolist(), lon[other].tolist()), mode="markers",
                    marker=dict(size=6, color="#c8c8c8"), hoverinfo="skip", name="non touché"))
    l_lat, l_lon = [], []
    for s, p in parent.items():
        l_lat += [float(lat[p]), float(lat[s]), None]; l_lon += [float(lon[p]), float(lon[s]), None]
    fig.add_trace(T(**xy(l_lat or [None], l_lon or [None]), mode="lines", line=dict(width=3, color="#7b2cbf"),
                    hoverinfo="skip", name="propagation"))
    fig.add_trace(T(**xy(lat[hit].tolist(), lon[hit].tolist()), mode="markers",
                    marker=dict(size=13, color=t_min[hit].tolist(), colorscale="Inferno_r", showscale=True,
                                colorbar=dict(title="min après<br>le 1er")),
                    hovertext=[f"capteur {sids[s]}<br>bouchon à {times[onset[s]]:%H:%M}"
                               + (f"<br>vient de {sids[parent[s]]}" if s in parent else "<br>ORIGINE") for s in hit],
                    hoverinfo="text", name="touché"))
    rts = [r for r, n in roots.items() if n >= 3]
    fig.add_trace(T(**xy(lat[rts].tolist() if rts else [None], lon[rts].tolist() if rts else [None]),
                    mode="markers+text" if not tiles else "markers",
                    marker=dict(size=24, color="rgba(0,180,255,0.45)"), hoverinfo="skip",
                    name="origine (≥ 3 capteurs touchés)"))
    fig.update_layout(height=560, margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", y=-0.02))
    if tiles:
        fig.update_layout(map=dict(style="carto-positron", zoom=10, center=dict(lat=34.11, lon=-118.33)))
    else:
        fig.update_yaxes(scaleanchor="x", scaleratio=1 / np.cos(np.radians(34.1)), visible=False)
        fig.update_xaxes(visible=False)
        fig.update_layout(plot_bgcolor="#f4f4f4")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Couleur = minutes après le premier capteur touché (foncé = tôt). Trait violet = le bouchon est passé "
               "de ce voisin à ce capteur. Cercle bleu = origine d'un bouchon qui a touché au moins 3 capteurs.")

    # ---- résultat général sur toute la validation ----
    E, L = global_stats(d, adj, loc, agent, tuple(days))
    st.markdown("### Sur toute la validation (6 jours × matin + soir)")
    g1, g2, g3 = st.columns(3)
    vc = E.sens.value_counts(normalize=True)
    g1.metric("Le bouchon remonte la route", f"{vc.get('remonte (amont)', 0):.0%}",
              help=f"descend : {vc.get('descend (aval)', 0):.0%} · double sens : {vc.get('double sens', 0):.0%} "
                   f"({len(E)} liens de propagation)")
    g2.metric("Vitesse de l'onde (médiane)", f"{E.speed_kmh.median():.1f} km/h",
              help=f"moitié des liens entre {E.speed_kmh.quantile(.25):.1f} et {E.speed_kmh.quantile(.75):.1f} km/h")
    g3.metric("Bouchons annoncés par l'agent", f"{L.alerted.mean():.0%}",
              help=f"{int(L.alerted.sum())} sur {len(L)} débuts de bouchon réels ont eu une alerte dans l'heure d'avant")
    st.info(f"**Résultat :** dans {vc.get('remonte (amont)', 0):.0%} des cas, le bouchon passe d'un capteur à celui "
            f"**situé avant lui sur la route** : il remonte la file de voitures, en sens inverse de la circulation, "
            f"comme le prévoit la théorie du trafic (onde de choc). Quand l'agent a annoncé un bouchon, il l'a fait en "
            f"médiane **{L.lead_min.median():.0f} minutes avant** qu'il arrive.")

    fig2 = px.histogram(L.dropna(subset=["lead_min"]), x="lead_min", nbins=12, height=260,
                        labels={"lead_min": "minutes d'avance de l'alerte sur le bouchon réel"},
                        color_discrete_sequence=["#7b2cbf"])
    fig2.update_layout(margin=dict(l=0, r=0, t=30, b=0), yaxis_title="nombre de bouchons",
                       title="Combien de minutes avant le bouchon l'agent a alerté")
    st.plotly_chart(fig2, use_container_width=True)
    st.caption("Limites : la vitesse de l'onde utilise la distance à vol d'oiseau (plus courte que la route), donc elle "
               "est sous-estimée. L'agent n'alerte que les capteurs encore fluides : un bouchon précédé d'un "
               f"ralentissement n'est pas compté comme annoncé, d'où {L.alerted.mean():.0%} seulement.")
