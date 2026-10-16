"""Explication en langage simple d'un graphique SHAP local, par Claude (API Anthropic).
Sans clé API, une explication automatique (règles fixes, sans LLM) est produite à la place."""
import os

import numpy as np

MODEL = "claude-opus-5-5"

GLOSSAIRE = {
    "own_lag": "vitesse du capteur il y a {k}×5 min",
    "own_mean12": "vitesse moyenne du capteur sur la dernière heure",
    "own_frac_missing": "part de données manquantes du capteur sur la dernière heure",
    "up_lag": "vitesse moyenne des capteurs EN AMONT (voisins d'où vient le trafic) il y a {k}×5 min",
    "down_lag": "vitesse moyenne des capteurs EN AVAL (voisins où va le trafic) il y a {k}×5 min",
    "up_mean12": "vitesse moyenne des voisins en amont sur la dernière heure",
    "down_mean12": "vitesse moyenne des voisins en aval sur la dernière heure",
    "ha_target": "vitesse habituelle de ce capteur à cette heure et ce jour (moyenne historique)",
    "slot_target": "heure de la journée visée",
    "dow_target": "jour de la semaine visé",
    "sensor": "identité du capteur (ses habitudes propres)",
}


def describe(name):
    for prefix, txt in GLOSSAIRE.items():
        if name.startswith(prefix):
            rest = name[len(prefix):]
            if rest == "0":
                return txt.split(" il y a")[0].replace("vitesse", "vitesse actuelle", 1) + " (maintenant)"
            return txt.format(k=int(rest) * 5).replace("×5", "") if rest.isdigit() else txt
    return name


def _table(contrib, values, top=8):
    idx = contrib.abs().sort_values(ascending=False).index[:top]
    lines = []
    for n in idx:
        v = values.get(n, np.nan)
        v_txt = "manquant" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.1f}"
        lines.append(f"- {n} ({describe(n)}) : valeur {v_txt}, contribution {contrib[n]:+.2f} mph")
    return "\n".join(lines)


def build_prompt(ctx, contrib, values, base):
    neigh = contrib[[c for c in contrib.index if c.startswith(("up_", "down_"))]].sum()
    return f"""Contexte : projet de prévision du trafic sur METR-LA (207 capteurs d'autoroute à Los Angeles, vitesses en mph).
Un modèle LightGBM prédit la vitesse d'un capteur {ctx['horizon']} à l'avance. Le graphique SHAP montre, pour CETTE
prédiction, combien chaque variable a poussé la vitesse prédite vers le haut (+) ou vers le bas (-) par rapport à la
valeur moyenne du modèle.

Capteur {ctx['sensor']}, instant {ctx['time']}, horizon {ctx['horizon']}.
Valeur moyenne du modèle (base) : {base:.1f} mph. Vitesse prédite : {base + contrib.sum():.1f} mph.
Vitesse fluide habituelle de ce capteur : {ctx['free_flow']:.1f} mph. Vitesse réelle observée ensuite : {ctx['true']}.
Contribution totale des voisins routiers (amont + aval) : {neigh:+.2f} mph.

Variables les plus influentes :
{_table(contrib, values)}

Explique ce graphique en français simple à un professeur non spécialiste, en 5 à 7 phrases :
1. ce que le modèle prévoit (bouchon ou pas) ;
2. les 2 ou 3 raisons principales, en mots simples (pas de noms de variables techniques sans les traduire) ;
3. le rôle des voisins routiers (propagation de la congestion), seulement d'après les chiffres ci-dessus ;
4. si la vitesse réelle est connue, dis en une phrase si la prévision était bonne.
N'invente aucun chiffre qui n'est pas ci-dessus. Pas de titre, pas de liste à puces."""


def explain_with_claude(prompt, api_key):
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    response = client.beta.messages.create(
        model=MODEL,
        max_tokens=4000,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        output_config={"effort": "low"},
        messages=[{"role": "user", "content": prompt}],
    )
    if response.stop_reason == "refusal":
        return None
    return "".join(b.text for b in response.content if b.type == "text").strip()


def explain_without_llm(ctx, contrib, base):
    """Repli sans LLM : phrases construites à partir des chiffres SHAP (déterministe)."""
    pred = base + contrib.sum()
    ratio = pred / ctx["free_flow"]
    etat = "un bouchon" if ratio < 0.5 else ("un ralentissement" if ratio < 0.75 else "une circulation fluide")
    order = contrib.abs().sort_values(ascending=False).index
    down = [n for n in order if contrib[n] < 0][:2]
    up = [n for n in order if contrib[n] > 0][:2]
    neigh = contrib[[c for c in contrib.index if c.startswith(("up_", "down_"))]].sum()
    txt = (f"Le modèle prévoit {pred:.0f} mph dans {ctx['horizon']}, soit {etat} "
           f"(vitesse fluide habituelle : {ctx['free_flow']:.0f} mph). ")
    if down:
        txt += "Ce qui fait baisser la prévision : " + " ; ".join(
            f"{describe(n)} ({contrib[n]:+.1f} mph)" for n in down) + ". "
    if up:
        txt += "Ce qui la fait monter : " + " ; ".join(f"{describe(n)} ({contrib[n]:+.1f} mph)" for n in up) + ". "
    txt += (f"Au total, les voisins sur la route comptent pour {neigh:+.1f} mph"
            + (" : la congestion des voisins se propage vers ce capteur." if neigh < -1 else "."))
    return txt


def get_api_key(sidebar_key=""):
    return sidebar_key or os.environ.get("ANTHROPIC_API_KEY", "")
