"""Encoder base class and shared neural building blocks.

A :class:`GNNEncoder` maps a :class:`~graco.data.batch.BatchedGraph` (with
dynamic node features in ``graph.x`` and edge features in ``graph.edge_attr``) to
per-node embeddings ``[N, out_dim]``.  Every concrete encoder is a stack of
scatter-based message-passing layers — see :mod:`graco.models.encoders`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from graco.data.batch import BatchedGraph


def get_activation(name: str) -> Callable[[Tensor], Tensor]:
    name = (name or "relu").lower()
    return {
        "relu": F.relu,
        "elu": F.elu,
        "gelu": F.gelu,
        "leaky_relu": lambda x: F.leaky_relu(x, 0.2),
        "tanh": torch.tanh,
        "identity": lambda x: x,
        "silu": F.silu,
    }[name]


class Normalizer(nn.Module):
    """Optional per-node feature normalization applied between layers.

    ``l2`` reproduces the DIRAC/FINDER ``l2_normalize(axis=1)`` after every ReLU;
    ``layer``/``batch`` use standard norm layers; ``none`` is a no-op.
    """

    def __init__(self, kind: str, dim: int):
        super().__init__()
        self.kind = (kind or "none").lower()
        if self.kind == "layer":
            self.norm = nn.LayerNorm(dim)
        elif self.kind == "batch":
            self.norm = nn.BatchNorm1d(dim)
        else:
            self.norm = None

    def forward(self, x: Tensor) -> Tensor:
        if self.kind == "l2":
            return F.normalize(x, p=2.0, dim=1, eps=1e-12)
        if self.norm is not None:
            return self.norm(x)
        return x


class MLP(nn.Module):
    """A simple configurable multi-layer perceptron."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 2,
        act: str = "relu",
        dropout: float = 0.0,
        norm: str = "none",
    ):
        super().__init__()
        self.act = get_activation(act)
        self.dropout = dropout
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        self.layers = nn.ModuleList(nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1))
        self.norms = nn.ModuleList(
            Normalizer(norm, dims[i + 1]) if i < len(dims) - 2 else nn.Identity()
            for i in range(len(dims) - 1)
        )

    def forward(self, x: Tensor) -> Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.act(x)
                x = self.norms[i](x)
                if self.dropout > 0:
                    x = F.dropout(x, self.dropout, self.training)
        return x


class GNNEncoder(nn.Module, ABC):
    """Abstract graph encoder: ``BatchedGraph -> node embeddings [N, out_dim]``."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        edge_dim: int = 0,
        out_dim: Optional[int] = None,
        act: str = "relu",
        norm: str = "l2",
        dropout: float = 0.0,
        residual: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.edge_dim = int(edge_dim or 0)
        self.out_dim = int(out_dim or hidden_dim)
        self.act = get_activation(act)
        self.norm_kind = norm
        self.dropout = float(dropout)
        self.residual = bool(residual)

    @abstractmethod
    def forward(self, graph: BatchedGraph) -> Tensor:  # -> [N, out_dim]
        ...

    # convenience: safe input features (envs always set graph.x)
    def input_x(self, graph: BatchedGraph) -> Tensor:
        if graph.x is not None:
            return graph.x.float()
        return torch.ones(graph.num_nodes, self.in_dim, device=graph.device)
