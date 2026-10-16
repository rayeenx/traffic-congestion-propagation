"""Graph WaveNet sur METR-LA, même protocole que LightGBM : split chrono 70/10/20, fenêtres 12 -> 12 dans
chaque bloc, scaler train-only (0 = manquant ignoré), loss et métriques masquées, early stopping sur la val.
Le test n'est pas lu. Usage : python -m src.run_gwn [--graph real|random|adaptive] [--epochs 60]"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch

from src.data import MaskedScaler, chrono_split, load_metr_la, time_features, window_anchors
from src.graphs import get_graph
from src.gwn import GraphWaveNet, transition_matrices
from src.leakage import check_scaler, check_splits
from src.metrics import HORIZONS, masked_metrics

SEED = 42


def masked_mae_loss(pred, true):
    m = (true > 0).float()
    m = m / m.mean().clamp_min(1e-8)
    return (torch.abs(pred - true) * m).mean()


def main(h5, adj_path, out_dir, graph, epochs, batch, lr, patience, max_minutes):
    torch.manual_seed(SEED); np.random.seed(SEED)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tag = "GraphWaveNet" + ("" if graph == "real" else f"_{graph}Graph")
    os.makedirs(os.path.join(out_dir, "models"), exist_ok=True)
    df, adj, _ = load_metr_la(h5, adj_path)
    values = df.values.astype(np.float32)
    T, N = values.shape
    splits = chrono_split(T); check_splits(splits, T)
    tr, va = splits[0], splits[1]
    scaler = MaskedScaler().fit(values[tr.start:tr.end]); check_scaler(scaler, values, tr)
    slot, _ = time_features(df.index)

    # entrée : vitesse normalisée (manquant -> 0 = moyenne train) + heure de la journée
    x_speed = np.where(values > 0, scaler.transform(values), 0.0).astype(np.float32)
    x_tod = np.repeat((slot / 288.0)[:, None], N, 1).astype(np.float32)
    feats = torch.tensor(np.stack([x_speed, x_tod], 0), device=dev)      # [2, T, N]
    target = torch.tensor(values, device=dev)                                # vitesses brutes, 0 = manquant

    if graph == "adaptive":
        supports = []
    else:
        supports = transition_matrices(get_graph(adj, graph, SEED))
    model = GraphWaveNet(N, supports).to(dev)
    print(f"{tag} : {sum(p.numel() for p in model.parameters())} paramètres, device={dev}, "
          f"supports fixes={len(supports)} + adaptatif")
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    a_tr = torch.tensor(window_anchors(tr), device=dev)
    a_va = torch.tensor(window_anchors(va), device=dev)
    off_in = torch.arange(-11, 1, device=dev)
    off_out = torch.arange(1, 13, device=dev)
    # garde-fou : toutes les fenêtres restent dans leur bloc
    assert int((a_tr[:, None] + off_out).max()) < tr.end and int((a_va[:, None] + off_in).min()) >= va.start

    def batch_xy(anchors):
        xi = anchors[:, None] + off_in                     # [B, 12]
        x = feats[:, xi].permute(1, 0, 3, 2)               # [B, 2, N, 12]
        y = target[anchors[:, None] + off_out]             # [B, 12, N]
        return x, y

    def predict(anchors):
        model.eval(); out = []
        with torch.no_grad():
            for i in range(0, len(anchors), 128):
                x, _ = batch_xy(anchors[i:i + 128])
                out.append(scaler.inverse(model(x)))
        return torch.cat(out)

    best, best_ep, wait, hist, t0 = np.inf, -1, 0, [], time.time()
    ckpt = os.path.join(out_dir, "models", f"{tag}.pt")
    for ep in range(epochs):
        model.train()
        perm = a_tr[torch.randperm(len(a_tr), device=dev)]
        losses = []
        for i in range(0, len(perm), batch):
            x, y = batch_xy(perm[i:i + batch])
            loss = masked_mae_loss(scaler.inverse(model(x)), y)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); losses.append(loss.item())
        _, yv = batch_xy(a_va)
        pv = predict(a_va)
        val_mae = masked_metrics(pv.cpu().numpy(), yv.cpu().numpy())["MAE"]
        hist.append(dict(epoch=ep, train_mae=float(np.mean(losses)), val_mae=val_mae))
        print(f"epoch {ep:3d} train MAE {np.mean(losses):.3f}  val MAE (12 pas) {val_mae:.3f}  "
              f"({(time.time() - t0) / 60:.1f} min)", flush=True)
        if val_mae < best:
            best, best_ep, wait = val_mae, ep, 0
            torch.save(model.state_dict(), ckpt)
        else:
            wait += 1
        if wait >= patience or (time.time() - t0) / 60 > max_minutes:
            print("arrêt :", "patience" if wait >= patience else "budget temps")
            break

    model.load_state_dict(torch.load(ckpt, map_location=dev))
    _, yv = batch_xy(a_va)
    pv, yv = predict(a_va).cpu().numpy(), yv.cpu().numpy()
    rows = []
    for hname, h in HORIZONS.items():
        m = masked_metrics(pv[:, h - 1], yv[:, h - 1])
        rows.append(dict(model=tag, split="val", horizon=hname, best_epoch=best_ep, **m))
        print(f"{tag} {hname}: MAE={m['MAE']:.3f} RMSE={m['RMSE']:.3f} MAPE={m['MAPE']:.2f}%")
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, f"{tag}_val.csv"), index=False)
    pd.DataFrame(hist).to_csv(os.path.join(out_dir, f"{tag}_history.csv"), index=False)
    np.save(os.path.join(out_dir, "dashboard", f"{tag}_val_pred.npy"), pv.astype(np.float32))  # [S, 12, N]
    with open(os.path.join(out_dir, f"{tag}_config.json"), "w") as f:
        json.dump(dict(graph=graph, epochs_run=len(hist), best_epoch=best_ep, batch=batch, lr=lr,
                       patience=patience, seed=SEED, inputs=["speed_z (0 si manquant)", "time_of_day"]), f, indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", default="data/metr-la.h5")
    ap.add_argument("--adj", default="data/adj_mx.pkl")
    ap.add_argument("--out", default="results")
    ap.add_argument("--graph", choices=["real", "random", "adaptive"], default="real",
                    help="real : graphe routier + adaptatif ; random : graphe aléatoire + adaptatif ; adaptive : adaptatif seul")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--max-minutes", type=float, default=90)
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "dashboard"), exist_ok=True)
    main(a.h5, a.adj, a.out, a.graph, a.epochs, a.batch, a.lr, a.patience, a.max_minutes)
