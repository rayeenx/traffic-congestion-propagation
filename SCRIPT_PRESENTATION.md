# Script de présentation DSO 4 (5 minutes)

**Avant d'entrer :** PC branché, internet actif, double-clic sur `run_dashboard.bat`, page `http://localhost:8501`
ouverte dans Chrome. Horizon **30min** dans la barre latérale. Si la carte est blanche : « Sans fond ».

## 1. Le problème (30 s, sans écran)
> « Mon DSO, c'est prévoir comment un bouchon se propage d'un capteur à ses voisins sur l'autoroute. J'utilise
> METR-LA : 207 capteurs de vitesse à Los Angeles, une mesure toutes les 5 minutes, et le graphe des routes qui
> relie les capteurs. Je prévois la vitesse à 15, 30 et 60 minutes. »

## 2. Onglet 🎬 Démo (1 min 30)
- Choisir **jeudi 24/05/2012**, cliquer **▶ Play**.
> « Chaque point est un capteur : vert fluide, orange ralenti, rouge bouchon. Les cercles violets sont les alertes
> de mon agent. Regardez : le violet apparaît **avant** que les points deviennent rouges. L'agent voit le bouchon
> arriver. »
- Mettre sur Pause vers 16:30–17:30 (pic du soir).
- Descendre à la courbe.
> « En rouge, les bouchons réels. En violet pointillé, ce que l'agent avait prévu 30 minutes avant. Les deux
> courbes se suivent, matin et soir. Ce jour-là : pic à 17:30 avec 95 capteurs en bouchon, 187 alertes,
> 74 % confirmées par un vrai bouchon. »

## 2 bis. Onglet 🌊 Propagation (1 min) — le moment fort
- Jeudi 24/05, période **Soir**. Montrer la carte : couleur foncée = touché tôt, traits violets = passage du bouchon
  d'un capteur à son voisin, cercles bleus = origines.
> « Ici ce sont les vraies données, pas des prévisions. Je retrouve le capteur où le bouchon commence, puis l'ordre
> dans lequel il touche ses voisins. »
- Descendre au résultat général.
> « Sur toute la validation, dans **78 %** des cas le bouchon passe au capteur **situé avant lui sur la route** :
> il remonte la file, en sens inverse de la circulation (seulement 10 % dans le sens de la circulation). C'est
> l'onde de choc de la théorie du trafic, retrouvée dans les données. Elle avance à environ **8 km/h**
> (médiane, à vol d'oiseau donc sous-estimée). Et quand mon agent annonce un bouchon, il le fait en médiane
> **35 minutes avant**. Il en annonce environ la moitié : il n'alerte que les capteurs encore fluides. »

## 3. Onglet 🤖 Agent (1 min)
> « C'est un agent, pas juste un modèle. Toutes les 5 minutes il fait 4 étapes :
> **percevoir** les vitesses, **s'auto-évaluer** (il compare l'erreur de LightGBM, Graph WaveNet et de la moyenne
> historique sur la dernière heure, uniquement sur des prédictions déjà vérifiables), **décider** avec le
> meilleur modèle du moment, et **agir** en lançant une alerte avec une action recommandée. »
- Montrer les 3 cases « modèle utilisé » et le flux d'alertes (type « propagation » = un voisin est déjà bloqué).
> « Il choisit Graph WaveNet la plupart du temps, mais repasse sur LightGBM ou la moyenne historique quand ils
> sont meilleurs. »

## 4. Onglet 📈 Capteur + SHAP (45 s)
- Garder le capteur proposé, montrer le graphique SHAP, cliquer **Expliquer ce graphique**.
> « Ici je montre **pourquoi** le modèle prévoit cette vitesse. Les barres rouges font baisser la prévision. On
> voit que la vitesse des voisins en aval compte : c'est la propagation de la congestion. Le texte en dessous
> explique le graphique en langage simple. »

## 5. Onglet 📊 Résultats (1 min)
> « Les chiffres sont en MAE (erreur moyenne en mph), sur la validation. »

| | 15 min | 30 min | 60 min |
|---|---|---|---|
| Moyenne historique | 4.32 | 4.32 | 4.31 |
| Dernière valeur | 3.22 | 3.81 | 4.78 |
| LightGBM sans voisins | 2.79 | 3.24 | 3.69 |
| LightGBM graphe aléatoire | 2.77 | 3.20 | 3.63 |
| LightGBM vrai graphe | 2.69 | 3.12 | 3.59 |
| Graph WaveNet graphe aléatoire | 2.55 | 2.87 | 3.23 |
| **Graph WaveNet vrai graphe** | **2.48** | **2.82** | **3.19** |

> « Le point clé : avec le **vrai graphe routier**, les deux modèles font mieux qu'avec un **graphe aléatoire**
> de même taille. Donc l'information des voisins aide vraiment : la propagation existe. Le gain est petit et
> diminue à 60 minutes. »

## 6. Conclusion (15 s)
> « Prochaines étapes : ouvrir le test une seule fois pour les chiffres finaux, et brancher mon agent avec les
> DSO du groupe via son API (`/agent/predictions`, `/agent/alerts`, `/agent/predict`). »

---

## Questions probables de la prof

**« Comment tu sais que le bouchon remonte la route ? »**
Le graphe DCRNN est orienté : une arête i -> j veut dire que la circulation va de i vers j. Pour chaque passage du
bouchon d'un capteur à son voisin, je regarde le sens de l'arête : 78 % vont vers l'amont, 10 % vers l'aval,
12 % sont des routes à double sens. Sur 310 passages, 6 jours, matin et soir.


**« Comment tu évites la fuite de données ? »**
Split chronologique 70/10/20 (train avant val avant test), normalisation calculée sur le train seulement, fenêtres
qui ne traversent jamais deux blocs, et le test est verrouillé : il n'a jamais été ouvert.

**« Pourquoi c'est un agent et pas juste un modèle ? »**
Il tourne en boucle seul, s'évalue lui-même sur ses erreurs récentes, change de modèle quand un autre est meilleur,
et produit des alertes avec une action. Il ne contrôle pas les feux : c'est de la prévision + recommandation, pas
du reinforcement learning.

**« Comment tu sais que le graphe sert à quelque chose ? »**
Ablation : même modèle avec le vrai graphe, un graphe aléatoire de même densité, et sans voisins. Le vrai graphe
gagne à chaque horizon, pour LightGBM et pour Graph WaveNet.

**« Tes chiffres sont-ils définitifs ? »**
Non, c'est la validation (et l'arrêt de l'entraînement a été choisi sur la validation, donc un peu optimiste).
Le test sera ouvert une seule fois à la fin.

**« Pourquoi les alertes ne sont pas toutes justes ? »**
Sur toute la validation : 1343 alertes, 65 % voient un vrai bouchon (vitesse < 50 % de la normale), 83 % au moins
un fort ralentissement (< 75 %). Le seuil peut être réglé : plus strict = moins de fausses alertes mais plus de
bouchons ratés.
