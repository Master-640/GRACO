"""SGC: Simple Graph Convolution (Wu et al., 2019).

Collapses a ``K``-layer GCN into a single linear model by removing all
intermediate nonlinearities: the fixed ``K``-hop diffusion ``P^K x`` is
precomputed with ``K`` sparse aggregations, then a single learned linear maps it
to the output width::

    P    = D~^{-1/2} A~ D~^{-1/2}      (renormalized adjacency, A~ = A + I)
    out  = (P^K x) W

``K`` defaults to ``num_layers``.  Because propagation is parameter-free the only
learnable weights live in the final linear.  Aggregation is scatter-based over
``edge_index`` (src=[0], dst=[1]) and folds in the implicit self-loop.

Reference: F. Wu et al., "Simplifying Graph Convolutional Networks" (ICML 2019).
"""

from __future__ import annotations

from typing import Tuple

import torch.nn as nn
from torch import Tensor

from graco.data.batch import BatchedGraph
from graco.models.base import GNNEncoder
from graco.registries import ENCODERS
from graco.utils.scatter import scatter_sum


@ENCODERS.register("sgc")
class SGCEncoder(GNNEncoder):
    """Simplified graph convolution; embeddings ``[N, hidden_dim]``."""

    def __init__(self, in_dim: int, hidden_dim: int = 64, num_layers: int = 3, **kwargs):
        super().__init__(in_dim, hidden_dim, num_layers, **kwargs)
        self.out_dim = self.hidden_dim
        self.K = self.num_layers  # number of diffusion hops P^K
        self.lin = nn.Linear(self.in_dim, self.hidden_dim)

    def _norm_coef(self, graph: BatchedGraph) -> Tuple[Tensor, Tensor]:
        """Symmetric coefficients ``c_ij`` [E] and self term ``c_ii`` [N]."""
        deg = graph.degree() + 1.0  # +1 implicit self-loop; clamps isolated -> 1
        dinv = deg.pow(-0.5)
        src, dst = graph.edge_index[0], graph.edge_index[1]
        c = dinv.index_select(0, dst) * dinv.index_select(0, src)
        if graph.edge_weight is not None:
            c = c * graph.edge_weight
        return c, 1.0 / deg

    def forward(self, graph: BatchedGraph) -> Tensor:
        x = self.input_x(graph)
        src, dst = graph.edge_index[0], graph.edge_index[1]
        c, self_coef = self._norm_coef(graph)

        h = x
        for _ in range(self.K):  # precompute P^K x, no nonlinearity between hops
            msg = h.index_select(0, src) * c.unsqueeze(-1)
            h = scatter_sum(msg, dst, graph.num_nodes) + self_coef.unsqueeze(-1) * h
        return self.lin(h)
