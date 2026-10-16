"""Graph WaveNet (Wu et al., IJCAI 2019), version compacte fidèle à l'implémentation officielle :
convolutions temporelles dilatées à porte (TCN) + diffusion de graphe sur 2 matrices de transition
(avant / arrière, depuis adj_mx) + 1 matrice adaptative apprise (E1 E2^T)."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def transition_matrices(adj):
    """Double transition DCRNN : P_f = D^-1 A (sens de circulation), P_b = D^-1 A^T."""
    def rw(A):
        d = A.sum(1)
        d_inv = np.where(d > 0, 1.0 / d, 0.0)
        return (d_inv[:, None] * A).astype(np.float32)
    return [rw(adj), rw(adj.T)]


class GraphConv(nn.Module):
    def __init__(self, c_in, c_out, n_supports, order=2, dropout=0.3):
        super().__init__()
        self.order, self.dropout = order, dropout
        self.mlp = nn.Conv2d(c_in * (n_supports * order + 1), c_out, 1)

    def forward(self, x, supports):          # x [B, C, N, L]
        out = [x]
        for A in supports:
            x1 = torch.einsum("bcnl,nm->bcml", x, A)   # diffusion le long des arêtes n -> m
            out.append(x1)
            for _ in range(2, self.order + 1):
                x1 = torch.einsum("bcnl,nm->bcml", x1, A)
                out.append(x1)
        h = self.mlp(torch.cat(out, dim=1))
        return F.dropout(h, self.dropout, training=self.training)


class GraphWaveNet(nn.Module):
    def __init__(self, n_nodes, supports, in_dim=2, out_dim=12, residual=32, dilation=32, skip=256, end=512,
                 blocks=4, layers=2, kernel=2, dropout=0.3, adaptive=True, emb_dim=10):
        super().__init__()
        assert supports or adaptive, "il faut au moins un support (fixe ou adaptatif)"
        self.register_buffer("fixed", torch.stack([torch.tensor(s) for s in supports]) if supports else torch.zeros(0))
        self.adaptive = adaptive
        n_sup = (len(supports) if supports else 0) + int(adaptive)
        if adaptive:
            self.E1 = nn.Parameter(torch.randn(n_nodes, emb_dim))
            self.E2 = nn.Parameter(torch.randn(emb_dim, n_nodes))
        self.start = nn.Conv2d(in_dim, residual, 1)
        self.filt, self.gate, self.skipc, self.gconv, self.bn = (nn.ModuleList() for _ in range(5))
        self.receptive = 1
        for _ in range(blocks):
            d, extra = 1, kernel - 1
            for _ in range(layers):
                self.filt.append(nn.Conv2d(residual, dilation, (1, kernel), dilation=d))
                self.gate.append(nn.Conv2d(residual, dilation, (1, kernel), dilation=d))
                self.skipc.append(nn.Conv2d(dilation, skip, 1))
                self.gconv.append(GraphConv(dilation, residual, n_sup, dropout=dropout))
                self.bn.append(nn.BatchNorm2d(residual))
                self.receptive += extra
                d *= 2; extra *= 2
        self.end1 = nn.Conv2d(skip, end, 1)
        self.end2 = nn.Conv2d(end, out_dim, 1)

    def supports(self):
        s = list(self.fixed) if self.fixed.numel() else []
        if self.adaptive:
            s.append(F.softmax(F.relu(self.E1 @ self.E2), dim=1))
        return s

    def forward(self, x):                     # x [B, in_dim, N, L=12] -> [B, 12, N]
        if x.size(3) < self.receptive:
            x = F.pad(x, (self.receptive - x.size(3), 0, 0, 0))
        x = self.start(x)
        sup = self.supports()
        skip = 0
        for i in range(len(self.filt)):
            res = x
            h = torch.tanh(self.filt[i](x)) * torch.sigmoid(self.gate[i](x))
            s = self.skipc[i](h)
            skip = s + (skip[..., -s.size(3):] if torch.is_tensor(skip) else 0)
            x = self.gconv[i](h, sup)
            x = self.bn[i](x + res[..., -x.size(3):])
        h = F.relu(skip)
        h = F.relu(self.end1(h))
        return self.end2(h)[..., -1]           # [B, 12, N]
