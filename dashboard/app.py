"""Dashboard DSO 4 : propagation de la congestion (période de VALIDATION uniquement), METR-LA ou PEMS08.
PEMS08 : définir DSO4_DATASET=pems08 avant de lancer (ou run_dashboard_pems08.bat).
Lancement depuis la racine : streamlit run dashboard/app.py
Prérequis : python -m src.precompute_val (crée results/dashboard/val_preds.npz)."""
import os
import sys
import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from agent.core import HEAVY, MODEL_KEYS, MODERATE, STEPS, TrafficAgent, alert_quality, load  # noqa: E402
from src.data import DATASETS, active_dataset, results_dir  # noqa: E402

DS = DATASETS[active_dataset()]
RES = results_dir()
st.set_page_config(page_title=f"DSO4 · Agent trafic {DS['label']}", layout="wide")
HZ = list(STEPS)


@st.cache_resource
def get_agent():
    d, adj = load()
    return d, adj, TrafficAgent(d, adj).run()


@st.cache_data
def get_locations(sensor_ids):
    if not DS["gps"]:  # PEMS08 : pas de coordonnées, schéma calculé depuis les distances du graphe
        from src.data import schematic_layout
        return schematic_layout(DS["adj"], len(sensor_ids))
    loc = pd.read_csv("data/graph_sensor_locations.csv")
    loc["sensor_id"] = loc["sensor_id"].astype(str)
    return loc.set_index("sensor_id").loc[list(sensor_ids)].reset_index()


@st.cache_data
def get_results():
    base = pd.read_csv(f"{RES}/baselines_val.csv")
    tags = ("LightGBM", "LightGBM_randomGraph", "LightGBM_noNeighbors",
            "GraphWaveNet", "GraphWaveNet_randomGraph", "GraphWaveNet_adaptiveGraph")
    lg = pd.concat([pd.read_csv(f"{RES}/{t}_val.csv") for t in tags if os.path.exists(f"{RES}/{t}_val.csv")])
    res = pd.concat([base, lg])[["model", "horizon", "MAE", "RMSE", "MAPE"]]
    names = {"LightGBM": "LightGBM · vrai graphe", "LightGBM_randomGraph": "LightGBM · graphe aléatoire",
             "LightGBM_noNeighbors": "LightGBM · sans voisins", "GraphWaveNet": "Graph WaveNet · vrai graphe",
             "GraphWaveNet_randomGraph": "Graph WaveNet · graphe aléatoire",
             "GraphWaveNet_adaptiveGraph": "Graph WaveNet · graphe appris seul"}
    res["model"] = res["model"].replace(names)
    shap = pd.read_csv(f"{RES}/LightGBM_shap_val.csv", index_col=0)
    return res, shap


@st.cache_resource
def get_shap_inputs():
    """Tout ce qu'il faut pour recalculer les features d'un instant val (TreeSHAP local)."""
    import lightgbm as lgb
    from src.baselines import HistoricalAverage
    from src.data import dataset_splits, load_dataset, time_features
    from src.features import neighbor_mats
    df, adj, _ = load_dataset(active_dataset())
    values = df.values.astype(np.float64)
    tr = dataset_splits(active_dataset(), len(values))[0]
    slot, dow = time_features(df.index)

    class FullHA:  # en val, la feature HA = HA du train complet (identique à OOFHistoricalAverage.predict)
        full = HistoricalAverage().fit(values[tr.start:tr.end], slot[tr.start:tr.end], dow[tr.start:tr.end])

        def predict(self, y, s, d, w=None):
            return self.full.predict_at(s[y], d[y])

    models = {h: lgb.Booster(model_file=f"{RES}/models/LightGBM_{h}.txt") for h in HZ}
    A_up, A_down = neighbor_mats(adj)
    return values, slot, dow, FullHA(), A_up, A_down, models


def local_shap(anchor, sensor, hname):
    from src.features import build_features
    values, slot, dow, ha, A_up, A_down, models = get_shap_inputs()
    x_idx = (anchor + np.arange(-11, 1))[None, :]
    F, names = build_features(values[x_idx], x_idx, STEPS[hname], slot, dow, ha, A_up, A_down, True)
    contrib = models[hname].predict(F[sensor:sensor + 1], pred_contrib=True)[0]
    return pd.Series(contrib[:-1], index=names), float(contrib[-1]), dict(zip(names, F[sensor].tolist()))


d, adj, agent = get_agent()
times = pd.to_datetime(d["times"])
sids = d["sensor_ids"]
loc = get_locations(tuple(sids))
ff = d["ha_free_flow"]

# ---------------- barre latérale ----------------
st.sidebar.title(f"Agent trafic · {DS['label']}")
st.sidebar.caption(f"Rejoué sur la **validation** ({times[0]:%d/%m} → {times[-1]:%d/%m/%Y}). Le test reste verrouillé."
                   + ("" if DS["gps"] else " Dates PEMS08 supposées (début 01/07/2016)."))
if "i" not in st.session_state:
    # démarre sur l'instant de la val où le plus de capteurs sont congestionnés (pointe réelle)
    st.session_state.i = int(np.argmax(np.nansum(np.where(d["current"] > 0, d["current"] / d["ha_free_flow"], np.nan)
                                                 < HEAVY, axis=1)))
hname = st.sidebar.radio("Horizon de prévision", HZ, index=1, horizontal=True)
if DS["gps"]:
    tiles = st.sidebar.radio("Fond de carte", ["Plan OpenStreetMap", "Sans fond (si la carte reste blanche)"],
                             help="Le plan demande WebGL2 + internet. Le mode sans fond marche partout.")
else:
    tiles = "Sans fond"
    st.sidebar.caption("PEMS08 n'a pas de GPS : les « cartes » sont un **schéma du réseau** calculé à partir des "
                       "distances entre capteurs voisins, pas une vraie carte.")
st.sidebar.divider()
st.sidebar.subheader("Onglets détaillés")
labels = times.strftime("%a %d/%m %H:%M").tolist()
st.session_state.i = labels.index(st.sidebar.select_slider("Instant t0", options=labels,
                                                            value=labels[st.session_state.i]))
c1, c2, c3 = st.sidebar.columns(3)
if c1.button("−1 h"):
    st.session_state.i = max(0, st.session_state.i - 12); st.rerun()
if c2.button("+5 min"):
    st.session_state.i = min(len(times) - 1, st.session_state.i + 1); st.rerun()
if c3.button("+1 h"):
    st.session_state.i = min(len(times) - 1, st.session_state.i + 12); st.rerun()
play = st.sidebar.toggle("▶ Lecture auto (5 min / s)")
color_by = st.sidebar.radio("Couleur de la carte", ["Vitesse (mph)", "Niveau de congestion"])
show_edges = st.sidebar.checkbox("Afficher le graphe routier", value=False)
TILES = tiles.startswith("Plan")
with st.sidebar.expander("🧠 Clé API Claude (explication SHAP)"):
    api_key_in = st.text_input("ANTHROPIC_API_KEY", type="password",
                               help="Gardée seulement pour cette session, jamais écrite sur le disque. "
                                    "Sinon, définir la variable d'environnement ANTHROPIC_API_KEY.")
i = st.session_state.i
k = HZ.index(hname)
t0 = times[i]

used = agent.model_used[i]
roll = agent.rolling[i]
pred_used = d[MODEL_KEYS[used[hname]]][i, k]
now = d["current"][i]
truth = d["true"][i, k]
past_alerts = [a for a in agent.alerts if pd.Timestamp(a.time) <= t0]
alerts_now = [a for a in past_alerts if pd.Timestamp(a.time) > t0 - pd.Timedelta(minutes=30)]

st.title(f"Agent de prévision des bouchons · {DS['city']} ({DS['label']})")
HAS_GF = "gf_pred_occ15" in d   # PEMS08 avec les sorties GO-FLOW (tools/export_goflow.py)
tabs = st.tabs(["🎬 Démo"] + (["🤝 GO-FLOW → DSO4"] if HAS_GF else []) +
               ["🌊 Propagation", "🗺️ Carte détaillée", "🤖 Agent", "📈 Capteur + SHAP", "📊 Résultats"])
tab_demo, (tab_prop, tab_map, tab_agent, tab_sensor, tab_res) = tabs[0], tabs[-5:]

with tab_demo:
    from dashboard import demo
    demo.render(d, agent, loc, hname, TILES)

if HAS_GF:
    with tabs[1]:
        from dashboard import goflow_tab
        goflow_tab.render(d, adj, agent, hname, i)

with tab_prop:
    from dashboard import propagation_tab
    propagation_tab.render(d, adj, loc, agent, demo.valid_days(d), TILES)

# ---------------- carte ----------------
with tab_map:
    st.caption(f"t0 = **{t0:%A %d/%m/%Y %H:%M}** · prédiction pour **{t0 + pd.Timedelta(minutes=5 * STEPS[hname]):%H:%M}** "
               f"(+{hname}) · modèle choisi par l'agent : **{used[hname]}**")

    m1, m2, m3, m4, m5 = st.columns(5)
    cong_now = int(np.nansum(np.where(now > 0, now / ff, np.nan) < HEAVY))
    cong_pred = int((pred_used / ff < HEAVY).sum())
    mask = truth > 0
    m1.metric("Capteurs congestionnés maintenant", cong_now)
    m2.metric(f"Prévus congestionnés à +{hname}", cong_pred, delta=cong_pred - cong_now, delta_color="inverse")
    m3.metric(f"MAE à cet instant (+{hname})",
              f"{np.abs(pred_used - truth)[mask].mean():.2f} mph" if mask.any() else "n/a")
    m4.metric("Alertes (30 dernières min)", len(alerts_now))
    m5.metric("Alertes depuis le début", len(past_alerts))


    def level(r):
        return np.where(np.isnan(r), "inconnu", np.where(r < HEAVY, "congestion", np.where(r < MODERATE, "ralenti", "fluide")))

    df_map = loc.copy()
    df_map["observé maintenant"] = np.where(now > 0, now, np.nan)
    df_map["prédit"] = pred_used
    df_map["réel (validation)"] = np.where(truth > 0, truth, np.nan)
    df_map["vitesse fluide"] = ff
    df_map["niveau prédit"] = level(pred_used / ff)
    df_map["niveau maintenant"] = level(np.where(now > 0, now / ff, np.nan))
    lvl_colors = {"fluide": "#2ca02c", "ralenti": "#ff9f1c", "congestion": "#d62728", "inconnu": "#9e9e9e"}

    def tr(lat, lon, **kw):
        """même trace, sur fond OSM (Scattermap) ou en simple nuage lon/lat (Scatter, sans WebGL)."""
        return go.Scattermap(lat=lat, lon=lon, **kw) if TILES else go.Scatter(x=lon, y=lat, **kw)

    def make_map(col, title, is_level):
        hover = {"sensor_id": True, "observé maintenant": ":.1f", "prédit": ":.1f", "réel (validation)": ":.1f",
                 "vitesse fluide": ":.1f", "latitude": False, "longitude": False}
        if TILES:
            kw = dict(lat="latitude", lon="longitude", zoom=9.6, center=dict(lat=34.10, lon=-118.35),
                      map_style="carto-positron")
            fn = px.scatter_map
        else:
            kw = dict(x="longitude", y="latitude")
            fn = px.scatter
        kw.update(hover_data=hover, height=560)
        if is_level:
            fig = fn(df_map, color=col, color_discrete_map=lvl_colors, **kw)
        else:
            fig = fn(df_map, color=col, color_continuous_scale="RdYlGn", range_color=(10, 70), **kw)
        fig.update_traces(marker=dict(size=11))
        if show_edges:
            A = (adj > (0.5 if DS["gps"] else 0)) & ~np.eye(len(adj), dtype=bool)
            lat, lon = [], []
            for a, b in zip(*np.nonzero(A)):
                lat += [loc.latitude[a], loc.latitude[b], None]; lon += [loc.longitude[a], loc.longitude[b], None]
            fig.add_trace(tr(lat, lon, mode="lines", line=dict(width=1, color="#888"),
                             hoverinfo="skip", name="graphe (poids > 0.5)"))
        for a in alerts_now:
            fig.add_trace(tr([loc.latitude[a.sensor]], [loc.longitude[a.sensor]], mode="markers",
                             marker=dict(size=22, color="#7b2cbf", opacity=0.35), showlegend=False,
                             hovertext=f"ALERTE {a.kind} +{a.horizon}", hoverinfo="text"))
            if a.source >= 0:
                fig.add_trace(tr([loc.latitude[a.source], loc.latitude[a.sensor]],
                                 [loc.longitude[a.source], loc.longitude[a.sensor]], mode="lines",
                                 line=dict(width=4, color="#7b2cbf"), showlegend=False, hoverinfo="skip"))
        fig.update_layout(margin=dict(l=0, r=0, t=30, b=0), title=title,
                          legend=dict(orientation="h", y=-0.02))
        if not TILES:
            fig.update_yaxes(scaleanchor="x", scaleratio=1 / np.cos(np.radians(34.1)), showgrid=False, title="")
            fig.update_xaxes(showgrid=False, title="")
            fig.update_layout(plot_bgcolor="#f4f4f4")
        return fig

    is_level = color_by.startswith("Niveau")
    cA, cB = st.columns(2)
    cA.plotly_chart(make_map("niveau maintenant" if is_level else "observé maintenant",
                             f"Observé à t0 ({t0:%H:%M})", is_level), width="stretch")
    cB.plotly_chart(make_map("niveau prédit" if is_level else "prédit",
                             f"Prédit à +{hname} par l'agent ({used[hname]})", is_level), width="stretch")
    st.caption("Cercles violets = capteurs en alerte (30 dernières min). Trait violet = propagation depuis un voisin "
               f"déjà congestionné. Niveau = vitesse / vitesse fluide du capteur : congestion < {HEAVY:.0%}, "
               f"ralenti < {MODERATE:.0%}.")

# ---------------- agent ----------------
with tab_agent:
    st.subheader("Boucle de l'agent à chaque pas de 5 min")
    st.markdown(f"**1. Perceive** vitesses à t0 + prédictions 15/30/60 ({', '.join(agent.models)}) → "
                "**2. Self-evaluate** MAE glissante 1 h de chaque modèle sur les prédictions déjà vérifiables → "
                "l'agent prend le meilleur par horizon → "
                "**3. Decide** niveau de congestion prédit → **4. Act** alerte + action recommandée (cooldown 30 min).")
    cols = st.columns(3)
    for c, h in zip(cols, HZ):
        c.metric(f"+{h} · modèle utilisé", used[h],
                 help="MAE glissante sur la dernière heure de prédictions vérifiables")
        c.caption(" · ".join(f"{m} {e:.2f}" for m, e in roll[h].items()) + " mph"
                  if any(np.isfinite(list(roll[h].values()))) else "pas encore assez d'historique")

    lo = max(0, i - 288)
    hist = pd.DataFrame({"temps": times[lo:i + 1],
                         **{m: [agent.rolling[j][hname][m] for j in range(lo, i + 1)] for m in agent.models}})
    fig = px.line(hist, x="temps", y=agent.models, height=260,
                  labels={"value": "MAE glissante (mph)", "variable": ""},
                  title=f"Auto-évaluation sur les 24 dernières heures (+{hname})")
    fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig, width="stretch")

    st.subheader("Flux d'alertes")
    show = past_alerts[-40:][::-1]
    if show:
        st.dataframe(pd.DataFrame([dict(
            heure=pd.Timestamp(a.time).strftime("%d/%m %H:%M"),
            type="🔁 propagation" if a.kind == "propagation" else "⚠️ congestion à venir",
            capteur=sids[a.sensor], horizon=a.horizon, **{"vitesse prédite": round(a.pred_speed, 1),
                                                         "vitesse fluide": round(a.free_flow, 1)},
            voisin_congestionné=sids[a.source] if a.source >= 0 else "", modèle=a.model, action=a.action)
            for a in show]), width="stretch", hide_index=True)
    else:
        st.info("Aucune alerte avant cet instant.")
    q = alert_quality(past_alerts)
    if q["n"]:
        st.caption(f"Qualité mesurée a posteriori sur la validation ({q['n']} alertes jusqu'ici) : "
                   f"{q['precision']:.0%} des alertes ont bien vu la vitesse réelle passer sous {HEAVY:.0%} de la "
                   f"vitesse fluide, {q['precision_moderate']:.0%} sous {MODERATE:.0%}. "
                   "La vérité n'est utilisée que pour cette mesure, jamais pour décider.")

# ---------------- capteur + SHAP ----------------
with tab_sensor:
    default = alerts_now[-1].sensor if alerts_now else int(np.nanargmin(np.where(now > 0, now / ff, np.inf)))
    s = st.selectbox("Capteur", range(len(sids)), index=default, format_func=lambda j: f"{sids[j]} (#{j})")
    lo, hi = max(0, i - 144), min(len(times), i + 1)
    h_steps = STEPS[hname]
    tgt_t = times[lo:hi] + pd.Timedelta(minutes=5 * h_steps)
    ts = pd.DataFrame({"temps cible": tgt_t,
                       "réel": np.where(d["true"][lo:hi, k, s] > 0, d["true"][lo:hi, k, s], np.nan),
                       **{m: d[MODEL_KEYS[m]][lo:hi, k, s] for m in agent.models},
                       "LastValue": d["last_value"][lo:hi, k, s]})
    fig = px.line(ts, x="temps cible", y=["réel", *agent.models, "LastValue"], height=320,
                  labels={"value": "vitesse (mph)", "variable": ""},
                  title=f"Capteur {sids[s]} : prédictions à +{hname} sur les 12 dernières heures")
    fig.add_hline(y=HEAVY * ff[s], line_dash="dot", line_color="#d62728", annotation_text="seuil congestion")
    fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig, width="stretch")

    st.subheader(f"Pourquoi cette prédiction ? (TreeSHAP local, capteur {sids[s]}, +{hname})")
    try:
        with st.spinner("Calcul SHAP…"):
            contrib, base, fvals = local_shap(int(d["anchors"][i]), s, hname)
        top = contrib.reindex(contrib.abs().sort_values(ascending=False).index)[:10]
        top = top[::-1]
        fig = go.Figure(go.Bar(x=top.values, y=top.index, orientation="h", text=[f"{v:+.2f}" for v in top.values],
                               marker_color=["#2ca02c" if v > 0 else "#d62728" for v in top.values]))
        fig.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0),
                          xaxis_title=f"contribution (mph) · valeur moyenne {base:.1f} → prédiction {base + contrib.sum():.1f}")
        st.plotly_chart(fig, width="stretch")
        neigh = contrib[[c for c in contrib.index if c.startswith(("up_", "down_"))]].sum()
        st.caption(f"Contribution totale des voisins amont/aval : {neigh:+.2f} mph. "
                   "Note : LightGBM vrai graphe (pas le repli HA).")

        st.subheader("🧠 Explication en langage simple")
        from dashboard import explain
        tv = float(d["true"][i, k, s])
        ctx = dict(sensor=sids[s], time=f"{t0:%A %d/%m %H:%M}", horizon=hname, free_flow=float(ff[s]),
                   true=f"{tv:.1f} mph" if tv > 0 else "inconnue")
        key = explain.get_api_key(api_key_in)
        if st.button("Expliquer ce graphique avec Claude" if key else "Expliquer ce graphique"):
            text, src = None, ""
            if key:
                try:
                    with st.spinner("Claude lit le graphique SHAP…"):
                        text = explain.explain_with_claude(explain.build_prompt(ctx, contrib, fvals, base), key)
                    src = f"Explication générée par Claude ({explain.MODEL}) à partir des valeurs SHAP ci-dessus."
                except Exception as e:
                    st.warning(f"Appel à Claude impossible ({type(e).__name__}) : explication automatique à la place.")
            if not text:
                text = explain.explain_without_llm(ctx, contrib, base)
                src = "Explication automatique sans LLM (règles fixes). Ajoute une clé API Anthropic pour l'explication par Claude."
            st.session_state["shap_expl"] = (text, src, (s, i, hname))
        if st.session_state.get("shap_expl", (None, None, None))[2] == (s, i, hname):
            text, src = st.session_state["shap_expl"][:2]
            st.success(text)
            st.caption(src)
    except Exception as e:  # ne casse pas la démo si le .h5 manque
        st.warning(f"SHAP local indisponible : {e}")

# ---------------- résultats ----------------
with tab_res:
    res, shap = get_results()
    st.subheader("Validation (MAE masquée, mph) : est-ce que le graphe aide ?")
    order = ["LastValue", "HistoricalAverage", "LightGBM · sans voisins", "LightGBM · graphe aléatoire",
             "LightGBM · vrai graphe", "Graph WaveNet · graphe appris seul", "Graph WaveNet · graphe aléatoire",
             "Graph WaveNet · vrai graphe"]
    res["model"] = pd.Categorical(res["model"], [m for m in order if m in set(res["model"])] +
                                  sorted(set(res["model"]) - set(order)))
    fig = px.bar(res.sort_values("model"), x="horizon", y="MAE", color="model", barmode="group", text_auto=".2f",
                 height=420, color_discrete_sequence=["#bdbdbd", "#8d99ae", "#ffb703", "#fb8500", "#219ebc",
                                                     "#cdb4db", "#9d4edd", "#5a189a"])
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), legend_title="")
    st.plotly_chart(fig, width="stretch")
    st.dataframe(res.pivot_table(index="model", columns="horizon", values="MAE", observed=True).round(3),
                 width="stretch")
    st.caption("LightGBM : vrai graphe < graphe aléatoire < sans voisins à chaque horizon, donc les voisins routiers "
               "réels apportent un gain faible mais constant. Graph WaveNet (diffusion sur le graphe + graphe appris) "
               "fait mieux que LightGBM aux 3 horizons, et son vrai graphe bat aussi un graphe aléatoire de même "
               "densité (gain qui diminue avec l'horizon). Chiffres validation (early stopping sur val), le test n'est "
               "pas encore ouvert.")
    st.subheader("Importance globale (mean |SHAP|, LightGBM vrai graphe)")
    sh = shap[hname].sort_values(ascending=True).tail(15)
    fig = px.bar(sh, orientation="h", height=450, labels={"value": "mean |SHAP| (mph)", "index": ""})
    fig.update_traces(marker_color=["#fb8500" if n.startswith(("up_", "down_")) else "#219ebc" for n in sh.index])
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), showlegend=False)
    st.plotly_chart(fig, width="stretch")
    st.caption("Orange = features des voisins (amont/aval).")

if play and i < len(times) - 1:
    time.sleep(1.0)
    st.session_state.i = i + 1
    st.rerun()
