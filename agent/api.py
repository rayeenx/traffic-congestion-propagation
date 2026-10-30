"""API de l'agent rejoué sur la validation. Lancement : uvicorn agent.api:app --port 8004
PEMS08 (aligné sur GO-FLOW) : définir DSO4_DATASET=pems08 avant de lancer.
GET /agent/status       état courant (pas, modèle utilisé par horizon, MAE glissante, nb d'alertes)
POST /agent/step?n=1    avance l'agent de n pas de 5 min
GET /agent/alerts       dernières alertes
POST /agent/predict_goflow  prévision à partir de la sortie GO-FLOW (PEMS08 seulement)"""
from fastapi import FastAPI

from agent.core import STEPS, TrafficAgent, alert_quality, load
from src.data import DATASETS, active_dataset

DATASET = active_dataset()
app = FastAPI(title=f"DSO4 traffic agent (validation {DATASETS[DATASET]['label']})")
_data, _adj = load()
_agent = TrafficAgent(_data, _adj)
_state = {"i": -1, "last": None}


def _clean(x):
    """NaN -> None pour que la réponse soit du JSON valide."""
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    return None if isinstance(x, float) and x != x else x


def _summary():
    last = _state["last"]
    return dict(dataset=DATASET, step=_state["i"], n_steps=len(_data["times"]),
                time=str(last["time"]) if last else None,
                model_used=last["used"] if last else None,
                rolling_mae=_clean(last["rolling"]) if last else None, models=_agent.models,
                n_alerts=len(_agent.alerts), alert_quality_so_far=_clean(alert_quality(_agent.alerts)))


@app.get("/", include_in_schema=False)
def root():
    from fastapi.responses import RedirectResponse
    return RedirectResponse("/docs")


@app.get("/agent/status")
def status():
    return _summary()


@app.post("/agent/step")
def step(n: int = 1):
    for _ in range(n):
        if _state["i"] + 1 >= len(_data["times"]):
            break
        _state["i"] += 1
        _state["last"] = _agent.step(_state["i"])
    return _summary()


@app.get("/agent/alerts")
def alerts(limit: int = 20):
    sid = _data["sensor_ids"]
    return [dict(time=str(a.time), kind=a.kind, sensor=str(sid[a.sensor]), horizon=a.horizon,
                 pred_speed=round(a.pred_speed, 1), free_flow=round(a.free_flow, 1),
                 goflow_pred_occupancy_15min=None if a.goflow_occ15 != a.goflow_occ15 else round(a.goflow_occ15, 4),
                 source=str(sid[a.source]) if a.source >= 0 else None, model=a.model, action=a.action)
            for a in _agent.alerts[-limit:]]


@app.get("/agent/predictions")
def predictions(horizon: str = "30min"):
    """Contrat d'intégration pour les autres DSO : vitesse prédite + niveau par capteur au pas courant."""
    import numpy as np
    from agent.core import HEAVY, MODERATE
    last = _state["last"]
    if last is None or horizon not in STEPS:
        return {"error": "appeler POST /agent/step d'abord ; horizon = 15min | 30min | 60min"}
    k = list(STEPS).index(horizon)
    pred, ff = last["pred"][k], _data["ha_free_flow"]
    level = np.where(pred / ff < HEAVY, "congestion", np.where(pred / ff < MODERATE, "ralenti", "fluide"))
    return dict(time=str(last["time"]), horizon=horizon, model=last["used"][horizon],
                sensors=[dict(sensor_id=str(s), pred_speed_mph=round(float(p), 1), free_flow_mph=round(float(f), 1),
                              level=str(l)) for s, p, f, l in zip(_data["sensor_ids"], pred, ff, level)])


# ---------- données nouvelles (ex. sortie du DSO d'un collègue) ----------
from pydantic import BaseModel  # noqa: E402

_predictor = None


class NewWindow(BaseModel):
    t0: str                              # dernier instant observé, ex. "2026-10-14T08:00"
    speeds: dict[str, list[float | None]]  # sensor_id du dataset actif -> 12 vitesses mph (plus ancienne -> t0)


@app.post("/agent/predict")
def predict_new(w: NewWindow):
    """Prévision 15/30/60 min sur une fenêtre d'1 h fournie par un autre service (modèles déjà entraînés)."""
    global _predictor
    if _predictor is None:
        from agent.predictor import Predictor
        _predictor = Predictor()
    try:
        return _predictor.predict(w.t0, w.speeds)
    except ValueError as e:
        from fastapi import HTTPException
        raise HTTPException(status_code=422, detail=str(e))


# ---------- intégration GO-FLOW (DSO de la collègue, PEMS08) ----------
class GoFlowRow(BaseModel):
    """Une ligne par capteur et par timestep, champs repris de la sortie GO-FLOW."""
    timestep: int                                  # origine 1, pas de 5 min (repère PEMS08 de GO-FLOW)
    sensor_id: int                                 # 0..169
    speed: float                                   # traffic_data.current_speed, mph
    flow: float | None = None                      # traffic_data.current_flow
    occupancy: float | None = None                 # traffic_data.current_occupancy
    predicted_occupancy_t15: float | None = None   # forecast.predicted_occupancy
    traffic_state: str | None = None               # free_flow | moderate | congested | severe


class GoFlowBatch(BaseModel):
    rows: list[GoFlowRow]


@app.post("/agent/predict_goflow")
def predict_goflow(b: GoFlowBatch):
    """Entrée = sortie GO-FLOW sur la dernière heure (12 timesteps x capteurs). Sortie = vitesses prédites
    15/30/60 min pour les 170 capteurs, avec l'état GO-FLOW recopié pour chaque capteur au dernier timestep."""
    from fastapi import HTTPException
    from src.data import pems08_time
    global _predictor
    if DATASET != "pems08":
        raise HTTPException(status_code=409, detail="lancer l'agent avec DSO4_DATASET=pems08 pour lire GO-FLOW")
    if not b.rows:
        raise HTTPException(status_code=422, detail="aucune ligne")
    t_last = max(r.timestep for r in b.rows)
    window = range(t_last - 11, t_last + 1)
    speeds, state, gf = {}, {}, {"occupancy": {}, "flow": {}, "pred_occ15": {}}
    for r in b.rows:
        if r.timestep in window:
            speeds.setdefault(str(r.sensor_id), [None] * 12)[r.timestep - window.start] = r.speed
        if r.timestep == t_last:
            state[str(r.sensor_id)] = dict(traffic_state=r.traffic_state,
                                           predicted_occupancy_t15=r.predicted_occupancy_t15)
            gf["occupancy"][str(r.sensor_id)] = r.occupancy
            gf["flow"][str(r.sensor_id)] = r.flow
            gf["pred_occ15"][str(r.sensor_id)] = r.predicted_occupancy_t15
    if _predictor is None:
        from agent.predictor import Predictor
        _predictor = Predictor()
    try:
        has_gf = any(v is not None for v in gf["pred_occ15"].values())
        out = _predictor.predict(str(pems08_time(t_last)), speeds, goflow=gf if has_gf else None)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    out["t0_timestep"] = t_last
    for row in out["sensors"]:
        row["goflow"] = state.get(row["sensor_id"])
    return out
