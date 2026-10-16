"""Onglet « Démo » : une journée de validation rejouée en animation, pour voir l'agent anticiper la congestion."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from agent.core import HEAVY, MODEL_KEYS, MODERATE, STEPS

LVL_COLOR = {"fluide": "#2ca02c", "ralenti": "#ff9f1c", "bouchon": "#d62728", "inconnu": "#b0b0b0"}


def level(ratio):
    return np.where(np.isnan(ratio), "inconnu",
                    np.where(ratio < HEAVY, "bouchon", np.where(ratio < MODERATE, "ralenti", "fluide")))


def chosen_predictions(d, agent, k, hname):
    """Prédiction réellement utilisée par l'agent à chaque pas (le modèle qu'il a choisi)."""
    out = np.empty_like(d["lgbm"][:, k])
    for i, used in enumerate(agent.model_used):
        out[i] = d[MODEL_KEYS[used[hname]]][i, k]
    return out


def valid_days(d):
    """Jours de semaine (hors Memorial Day) avec au moins 12 h de données entre 6 h et 20 h."""
    times = pd.to_datetime(d["times"])
    in_day = times[(times.hour >= 6) & (times.hour < 20)]
    counts = pd.Series(1, index=in_day.normalize()).groupby(level=0).size()
    return [dd for dd, n in counts.items() if n >= 144 and dd.dayofweek < 5 and dd != pd.Timestamp("2012-05-28")]


def render(d, agent, loc, hname, tiles):
    times = pd.to_datetime(d["times"])

    st.markdown(
        f"### Ce que fait l'agent, en une phrase\n"
        f"Toutes les 5 minutes, il regarde les 207 capteurs de Los Angeles, **prévoit où il y aura un bouchon dans "
        f"{hname.replace('min', ' min')}**, et **lance une alerte avant que le bouchon arrive**, en particulier quand "
        f"un voisin sur la route est déjà bloqué (la congestion se propage).")

    days = valid_days(d)
    c1, c2 = st.columns([2, 3])
    day = c1.selectbox("Journée à rejouer (jours de semaine)", days,
                       index=next((j for j, dd in enumerate(days) if dd == pd.Timestamp("2012-05-24")), 0),
                       format_func=lambda x: x.strftime("%A %d/%m/%Y"))
    c2.markdown("<br>Appuie sur **▶ Play** sous la carte. Points = capteurs. "
                "🟢 fluide · 🟠 ralenti · 🔴 bouchon · 🟣 **alerte de l'agent** (émise dans les 30 dernières min).",
                unsafe_allow_html=True)

    fig, fig2, msg = build_figs(d, agent, loc, day, hname, tiles)
    st.plotly_chart(fig, use_container_width=True)
    st.plotly_chart(fig2, use_container_width=True)
    st.info(msg + " Si la courbe violette suit la rouge, c'est que l'agent voit venir les bouchons en avance.")


@st.cache_resource(show_spinner="Préparation de l'animation…")
def build_figs(_d, _agent, _loc, day, hname, tiles):
    d, agent, loc = _d, _agent, _loc
    times = pd.to_datetime(d["times"])
    sids, ff = d["sensor_ids"], d["ha_free_flow"]
    k, h = list(STEPS).index(hname), STEPS[hname]
    sel = np.flatnonzero((times >= day + pd.Timedelta(hours=6)) & (times <= day + pd.Timedelta(hours=20)))
    lat, lon = loc.latitude.values, loc.longitude.values   # converties en listes dans les traces (pas de bdata)
    alert_t = np.array([np.datetime64(a.time) for a in agent.alerts])

    def frame_data(i):
        now = d["current"][i]
        lv = level(np.where(now > 0, now / ff, np.nan))
        win = (alert_t <= np.datetime64(times[i])) & (alert_t > np.datetime64(times[i] - pd.Timedelta(minutes=30)))
        al = [agent.alerts[j] for j in np.flatnonzero(win)]
        a_idx = [a.sensor for a in al]
        l_lat, l_lon = [], []
        for a in al:
            if a.source >= 0:
                l_lat += [float(lat[a.source]), float(lat[a.sensor]), None]
                l_lon += [float(lon[a.source]), float(lon[a.sensor]), None]
        hover = [f"capteur {s}<br>vitesse {v:.0f} mph · {l}" if v > 0 else f"capteur {s}<br>donnée manquante"
                 for s, v, l in zip(sids, now, lv)]
        a_hover = [f"ALERTE capteur {sids[a.sensor]}<br>bouchon prévu dans {a.horizon}<br>"
                   f"{'propagation depuis ' + str(sids[a.source]) if a.source >= 0 else 'congestion à venir'}"
                   for a in al]
        T = go.Scattermap if tiles else go.Scatter
        xy = (lambda la, lo: dict(lat=la, lon=lo)) if tiles else (lambda la, lo: dict(x=lo, y=la))
        return [
            T(**xy(l_lat or [None], l_lon or [None]), mode="lines", line=dict(width=3, color="#7b2cbf"),
              hoverinfo="skip", showlegend=False),
            T(**xy(lat[a_idx].tolist() if a_idx else [None], lon[a_idx].tolist() if a_idx else [None]), mode="markers",
              marker=dict(size=24, color="#9d4edd", opacity=0.45), hovertext=a_hover or None, hoverinfo="text",
              showlegend=False),
            T(**xy(lat.tolist(), lon.tolist()), mode="markers", marker=dict(size=10, color=[LVL_COLOR[x] for x in lv]),
              hovertext=hover, hoverinfo="text", showlegend=False),
        ], len(set(a_idx)), int((lv == "bouchon").sum())

    frames, n_al, n_bo = [], [], []
    for i in sel:
        data, na, nb = frame_data(i)
        n_al.append(na); n_bo.append(nb)
        frames.append(go.Frame(data=data, name=times[i].strftime("%H:%M"),
                               layout=dict(title=dict(text=f"<b>{times[i]:%H:%M}</b> · {nb} capteurs en bouchon · "
                                                           f"{na} alertes actives"))))
    fig = go.Figure(data=frames[0].data, frames=frames)
    steps = [dict(method="animate", label=f.name,
                  args=[[f.name], dict(mode="immediate", frame=dict(duration=0, redraw=True),
                                       transition=dict(duration=0))]) for f in frames]
    fig.update_layout(
        height=620, margin=dict(l=0, r=0, t=50, b=0), title=dict(text=frames[0].layout.title.text),
        updatemenus=[dict(type="buttons", x=0, y=-0.02, xanchor="left", yanchor="top", direction="left",
                          buttons=[dict(label="▶ Play", method="animate",
                                        args=[None, dict(frame=dict(duration=350, redraw=True), fromcurrent=True,
                                                         transition=dict(duration=0))]),
                                   dict(label="⏸ Pause", method="animate",
                                        args=[[None], dict(mode="immediate", frame=dict(duration=0, redraw=False))])])],
        sliders=[dict(active=0, steps=steps, x=0.12, len=0.88, y=-0.02, currentvalue=dict(prefix="heure : "))])
    if tiles:
        fig.update_layout(map=dict(style="carto-positron", zoom=10, center=dict(lat=34.11, lon=-118.33)))
    else:
        fig.update_yaxes(scaleanchor="x", scaleratio=1 / np.cos(np.radians(34.1)), showgrid=False, visible=False)
        fig.update_xaxes(showgrid=False, visible=False)
        fig.update_layout(plot_bgcolor="#f4f4f4")

    # ---- courbe : l'agent voit-il venir le pic ? ----
    pred = chosen_predictions(d, agent, k, hname)
    real_cnt = np.nansum(np.where(d["current"] > 0, d["current"] / ff, np.nan) < HEAVY, axis=1)
    pred_cnt = (pred / ff < HEAVY).sum(axis=1)
    pred_t = times + pd.Timedelta(minutes=5 * h)         # une prédiction faite à t vise t + h
    day_mask = (times >= day + pd.Timedelta(hours=5)) & (times <= day + pd.Timedelta(hours=21))
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=times[day_mask], y=real_cnt[day_mask], name="bouchons réels", line=dict(color="#d62728", width=3)))
    fig2.add_trace(go.Scatter(x=pred_t[day_mask], y=pred_cnt[day_mask], name=f"bouchons prévus {hname} avant",
                              line=dict(color="#7b2cbf", width=2, dash="dash")))
    fig2.add_trace(go.Bar(x=times[sel], y=n_al, name="alertes actives", marker_color="rgba(157,78,221,0.25)",
                          yaxis="y2"))
    fig2.update_layout(height=320, margin=dict(l=0, r=0, t=40, b=0), legend=dict(orientation="h", y=1.12),
                       title=f"Nombre de capteurs en bouchon : réel (rouge) vs prévu par l'agent {hname} à l'avance (violet)",
                       yaxis=dict(title="capteurs en bouchon"),
                       yaxis2=dict(title="alertes", overlaying="y", side="right", showgrid=False))

    i_peak = sel[int(np.argmax(np.array(n_bo)))]
    day_alerts = [a for a in agent.alerts
                  if day + pd.Timedelta(hours=6) <= pd.Timestamp(a.time) <= day + pd.Timedelta(hours=20)]
    ok = [a for a in day_alerts if a.true_speed > 0]
    msg = f"Ce jour-là, le pic est à **{times[i_peak]:%H:%M}** avec **{max(n_bo)} capteurs en bouchon**. "
    if ok:
        hit = np.mean([a.true_speed < HEAVY * a.free_flow for a in ok])
        msg += (f"Entre 6 h et 20 h, l'agent a lancé **{len(day_alerts)} alertes** ; **{hit:.0%}** ont été confirmées "
                f"par un vrai bouchon à l'heure prévue (vérifié après coup sur la validation).")
    return fig, fig2, msg
