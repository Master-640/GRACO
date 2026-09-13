"""Symmetric-normalized Graph Convolutional Network (Kipf & Welling, 2017).

Native scatter implementation with **implicit self-loops**: every layer folds a
self term into the propagation so that

    h_i = c_ii (W h_i) + sum_{j -> i} c_ij (W h_j),
    c_ij = 1 / sqrt((deg_i + 1)(deg_j + 1)),

which is exactly ``D~^{-1/2} A~ D~^{-1/2} H W`` with ``A~ = A + I``.  Aggregation
is scatter-based over ``edge_index`` (src=[0], dst=[1]).

Reference: T. Kipf, M. Welling, "Semi-Supervised Classification with Graph
Convolutional Networks" (ICLR 2017).
"""

from __future__ import annotations

from typing import Tuple

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from graco.data.batch import BatchedGraph
from graco.models.base import GNNEncoder, Normalizer
from graco.registries import ENCODERS
from graco.utils.scatter import scatter_sum


@ENCODERS.register("gcn")
class GCNEncoder(GNNEncoder):
    """Stacked GCN layers producing per-node embeddings ``[N, out_dim]``."""

    def __init__(self, in_dim: int, hidden_dim: int = 64, num_layers: int = 3, **kwargs):
        super().__init__(in_dim, hidden_dim, num_layers, **kwargs)
        dims = [self.in_dim] + [self.hidden_dim] * (self.num_layers - 1) + [self.out_dim]
        self.lins = nn.ModuleList(
            nn.Linear(dims[i], dims[i + 1]) for i in range(self.num_layers)
        )
        self.norms = nn.ModuleList(
            Normalizer(self.norm_kind, dims[i + 1]) for i in range(self.num_layers)
        )
        self._dims = dims

    def _edge_norm(self, graph: BatchedGraph) -> Tuple[Tensor, Tensor]:
        """Return per-edge coefficient ``c_ij`` [E] and self coefficient ``c_ii`` [N]."""
        deg = graph.degree() + 1.0  # +1 for the implicit self-loop; clamps isolated -> 1
        dinv = deg.pow(-0.5)  # 1 / sqrt(deg + 1)
        src, dst = graph.edge_index[0], graph.edge_index[1]
        c = dinv.index_select(0, dst) * dinv.index_select(0, src)  # [E]
        if graph.edge_weight is not None:
            c = c * graph.edge_weight
        return c, 1.0 / deg  # c_ii = 1 / (deg_i + 1)

    def forward(self, graph: BatchedGraph) -> Tensor:
        x = self.input_x(graph)
        src, dst = graph.edge_index[0], graph.edge_index[1]
        c, self_coef = self._edge_norm(graph)
        for i, lin in enumerate(self.lins):
            wh = lin(x)  # [N, d_out]
            msg = wh.index_select(0, src) * c.unsqueeze(-1)  # [E, d_out]
            out = scatter_sum(msg, dst, graph.num_nodes)
            out = out + self_coef.unsqueeze(-1) * wh  # implicit self-loop
            if i < self.num_layers - 1:
                out = self.act(out)
                out = self.norms[i](out)
                if self.dropout > 0:
                    out = F.dropout(out, self.dropout, self.training)
            if self.residual and self._dims[i] == self._dims[i + 1]:
                out = out + x
            x = out
        return x
