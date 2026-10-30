"""Onglet « GO-FLOW → DSO4 » : comment la sortie de GO-FLOW (DSO de la collègue) complète DSO4.
GO-FLOW prévoit l'occupation de chaque capteur à +15 min (son XGBoost). DSO4 s'en sert de deux façons :
entrée du modèle « LightGBM+GO-FLOW », et confirmation des alertes de l'agent. On compare avec DSO4 seul."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from agent.core import HEAVY, MODERATE, STEPS, TrafficAgent, alert_quality


@st.cache_resource(show_spinner="Agent sans GO-FLOW (comparaison)…")
def agent_alone(_d, _adj):
    return TrafficAgent(_d, _adj, use_goflow=False).run()


@st.cache_data(show_spinner=False)
def comparison(_d, _adj, _agent, key):
    from agent.propagation import lead_times
    rows = []
    for name, a in (("DSO4 seul", agent_alone(_d, _adj)), ("DSO4 + GO-FLOW", _agent)):
        q, L = alert_quality(a.alerts), lead_times(_d, a)
        rows.append({"version": name, "alertes": len(a.alerts),
                     "confirmées (bouchon réel)": q["precision"], "confirmées (au moins ralenti)": q["precision_moderate"],
                     "bouchons annoncés": L.alerted.mean(), "n_bouchons": len(L),
                     "n_annoncés": int(L.alerted.sum())})
    ff, mae = _d["ha_free_flow"], []
    for k, h in enumerate(STEPS):
        y = _d["true"][:, k]
        m = (y > 0) & (y < MODERATE * ff)            # moments où ça ralentit vraiment
        for name, key_ in (("LightGBM (DSO4 seul)", "lgbm"), ("LightGBM + sortie GO-FLOW", "lgbm_gf")):
            mae.append(dict(horizon=h, modèle=name, MAE=float(np.abs(_d[key_][:, k] - y)[m].mean())))
    return pd.DataFrame(rows), pd.DataFrame(mae)


def render(d, adj, agent, hname, i):
    sids, times = d["sensor_ids"], pd.to_datetime(d["times"])
    st.markdown(
        "### Ce que GO-FLOW apporte à DSO4\n"
        "**GO-FLOW** (DSO de ma collègue) prévoit, pour chaque capteur, **l'occupation de la route dans 15 min** "
        "(son modèle XGBoost). **DSO4** (le mien) prévoit les **vitesses sur tout le réseau à 15/30/60 min** et la "
        "**propagation** des bouchons. Sa sortie est mon entrée, à deux endroits :\n"
        "1. **Entrée du modèle** : son occupation prévue (capteur + voisins) est une variable du modèle "
        "« LightGBM + GO-FLOW », que l'agent met en concurrence avec les autres.\n"
        "2. **Confirmation des alertes** : l'agent ne lance une alerte que si GO-FLOW prévoit aussi une occupation "
        "élevée sur ce capteur (ses propres seuils, calculés sur le train).")
    st.caption("Chaîne : capteurs PEMS08 → GO-FLOW (occupation +15 min) → DSO4 (vitesses 15/30/60 min + propagation) "
               "→ alertes confirmées → orchestrateur / autres agents du groupe.")

    comp, mae = comparison(d, adj, agent, len(agent.alerts))
    a, b = comp.iloc[0], comp.iloc[1]
    st.markdown("#### Le gain, mesuré sur la validation")
    c1, c2, c3 = st.columns(3)
    c1.metric("Alertes confirmées par un vrai bouchon", f"{b['confirmées (bouchon réel)']:.0%}",
              delta=f"{(b['confirmées (bouchon réel)'] - a['confirmées (bouchon réel)']) * 100:+.0f} pts vs DSO4 seul")
    c2.metric("Alertes suivies d'au moins un ralentissement", f"{b['confirmées (au moins ralenti)']:.0%}",
              delta=f"{(b['confirmées (au moins ralenti)'] - a['confirmées (au moins ralenti)']) * 100:+.0f} pts")
    c3.metric("Bouchons annoncés à l'avance", f"{b['bouchons annoncés']:.0%}",
              delta=f"{(b['bouchons annoncés'] - a['bouchons annoncés']) * 100:+.0f} pts",
              help=f"{b['n_annoncés']} sur {b['n_bouchons']} débuts de bouchon (DSO4 seul : {a['n_annoncés']})")
    show = comp.drop(columns=["n_bouchons", "n_annoncés"]).set_index("version")
    st.dataframe(show.style.format({c: "{:.0%}" for c in show.columns if c != "alertes"}), width="stretch")

    fig = go.Figure()
    for name, color in (("LightGBM (DSO4 seul)", "#8d99ae"), ("LightGBM + sortie GO-FLOW", "#2a9d8f")):
        s = mae[mae["modèle"] == name]
        fig.add_trace(go.Bar(x=s.horizon, y=s.MAE, name=name, marker_color=color, text=s.MAE.round(2)))
    fig.update_layout(barmode="group", height=320, margin=dict(l=0, r=0, t=40, b=0), yaxis_title="MAE (mph)",
                      title="Erreur quand la route ralentit vraiment (vitesse réelle < 75 % de la vitesse fluide)",
                      legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig, width="stretch")

    # ---- un capteur : la prévision GO-FLOW annonce la chute de vitesse prévue par DSO4 ----
    st.markdown("#### Sur un capteur : son occupation prévue et ma vitesse prévue")
    k = list(STEPS).index(hname)
    alerted = sorted({al.sensor for al in agent.alerts})
    default = alerted[0] if alerted else 0
    s = st.selectbox("Capteur", range(len(sids)), index=default, key="gf_sensor",
                     format_func=lambda j: f"{sids[j]}" + (" (alerte de l'agent)" if j in alerted else ""))
    day = times[i].normalize()
    sel = np.flatnonzero((times >= day) & (times < day + pd.Timedelta(days=1)))
    th = __import__("json").loads(str(d["gf_thresholds"]))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=times[sel], y=d["current"][sel, s], name="vitesse réelle (mph)",
                             line=dict(color="#d62728", width=2)))
    fig.add_trace(go.Scatter(x=times[sel] + pd.Timedelta(minutes=5 * STEPS[hname]), y=d["lgbm_gf"][sel, k, s],
                             name=f"vitesse prévue par DSO4 ({hname} avant)", line=dict(color="#7b2cbf", dash="dash")))
    fig.add_trace(go.Scatter(x=times[sel] + pd.Timedelta(minutes=15), y=d["gf_pred_occ15"][sel, s],
                             name="occupation prévue par GO-FLOW (15 min avant)", yaxis="y2",
                             line=dict(color="#2a9d8f")))
    fig.add_hline(y=th["occupancy_high"], yref="y2", line=dict(color="#2a9d8f", dash="dot"),
                  annotation_text="seuil « occupation élevée » de GO-FLOW")
    for al in agent.alerts:
        if al.sensor == s and day <= pd.Timestamp(al.time) < day + pd.Timedelta(days=1):
            fig.add_vline(x=pd.Timestamp(al.time), line=dict(color="#9d4edd", width=1))
    fig.update_layout(height=380, margin=dict(l=0, r=0, t=30, b=0), legend=dict(orientation="h", y=-0.2),
                      yaxis=dict(title="vitesse (mph)"),
                      yaxis2=dict(title="occupation", overlaying="y", side="right", showgrid=False),
                      title=f"Capteur {sids[s]}, {day:%A %d/%m} · traits violets = alertes confirmées par GO-FLOW")
    st.plotly_chart(fig, width="stretch")
    st.caption("Quand l'occupation prévue par GO-FLOW monte au-dessus de son seuil, la vitesse chute peu après : "
               "c'est ce signal qui confirme les alertes de DSO4. Prévisions affichées à l'heure qu'elles visent.")
