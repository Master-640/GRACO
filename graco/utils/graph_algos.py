"""Vectorized graph algorithms used by environments.

The centrepiece is a **batched connected-components** routine that runs entirely
on-device via label propagation (iterated ``scatter_max``).  Network-dismantling
rewards depend on the size of the largest connected component (LCC) after node
removals; recomputing that for a whole batch of graphs on GPU — rather than a
per-graph C++ union-find — is what keeps the vectorized env fast.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from graco.utils.scatter import scatter_max, scatter_sum


@torch.no_grad()
def connected_component_labels(
    edge_index: Tensor,
    num_nodes: int,
    active_mask: Optional[Tensor] = None,
    max_iter: Optional[int] = None,
) -> Tensor:
    """Return a component label ``[N]`` for every node via label propagation.

    Labels are the smallest-reachable-index convention would need scatter_min;
    we use ``scatter_max`` so a component's label is the largest global node id
    in it.  Only edges whose *both* endpoints are active propagate; inactive
    (removed) nodes keep their own id and are excluded downstream.

    Converges in O(diameter) iterations; we detect a fixed point and stop early.
    """
    device = edge_index.device
    src, dst = edge_index[0], edge_index[1]
    labels = torch.arange(num_nodes, device=device)
    if active_mask is None:
        active_mask = torch.ones(num_nodes, dtype=torch.bool, device=device)

    if edge_index.numel() == 0:
        return labels

    # keep only edges with both endpoints active
    edge_ok = active_mask[src] & active_mask[dst]
    src_a = src[edge_ok]
    dst_a = dst[edge_ok]
    if src_a.numel() == 0:
        return labels

    cap = max_iter if max_iter is not None else num_nodes
    for _ in range(int(cap)):
        # message = source label along active edges; max-aggregate into dst
        cand = scatter_max(labels.index_select(0, src_a), dst_a, num_nodes, fill_value=-1)
        new_labels = torch.maximum(labels, cand)
        if torch.equal(new_labels, labels):
            break
        labels = new_labels
    return labels


@torch.no_grad()
def largest_component_size(
    edge_index: Tensor,
    num_nodes: int,
    batch: Tensor,
    num_graphs: int,
    active_mask: Optional[Tensor] = None,
    max_iter: Optional[int] = None,
) -> Tensor:
    """Size of the largest connected component per graph ``[B]`` (active nodes only)."""
    if active_mask is None:
        active_mask = torch.ones(num_nodes, dtype=torch.bool, device=edge_index.device)
    labels = connected_component_labels(edge_index, num_nodes, active_mask, max_iter)
    active_f = active_mask.to(torch.float32)
    # component size = #active nodes sharing a label
    comp_size = scatter_sum(active_f, labels, num_nodes)  # indexed by label id
    size_of_node = comp_size.index_select(0, labels) * active_f  # inactive -> 0
    lcc = scatter_max(size_of_node, batch, num_graphs, fill_value=0.0)
    # match the reference C++ `maxRankCount` floor of 1 for any graph that still
    # has at least one active node (isolated singletons count as size 1).
    has_active = scatter_sum(active_f, batch, num_graphs) > 0
    return torch.where(has_active, lcc.clamp_min(1.0), lcc)


@torch.no_grad()
def num_connected_components(
    edge_index: Tensor,
    num_nodes: int,
    batch: Tensor,
    num_graphs: int,
    active_mask: Optional[Tensor] = None,
    max_iter: Optional[int] = None,
) -> Tensor:
    """Number of connected components per graph ``[B]`` (active nodes only)."""
    if active_mask is None:
        active_mask = torch.ones(num_nodes, dtype=torch.bool, device=edge_index.device)
    labels = connected_component_labels(edge_index, num_nodes, active_mask, max_iter)
    # a node is a "component root" if it carries its component's max label,
    # i.e. label == own id, and it is active.
    is_root = (labels == torch.arange(num_nodes, device=edge_index.device)) & active_mask
    return scatter_sum(is_root.to(torch.float32), batch, num_graphs)


@torch.no_grad()
def pairwise_connectivity(
    edge_index: Tensor,
    num_nodes: int,
    batch: Tensor,
    num_graphs: int,
    active_mask: Optional[Tensor] = None,
    max_iter: Optional[int] = None,
) -> Tensor:
    """Sum over components of ``C(C-1)/2`` per graph ``[B]`` (pairwise connectivity).

    This is the objective used by some dismantling formulations (minimize the
    number of still-connected node pairs).
    """
    if active_mask is None:
        active_mask = torch.ones(num_nodes, dtype=torch.bool, device=edge_index.device)
    labels = connected_component_labels(edge_index, num_nodes, active_mask, max_iter)
    active_f = active_mask.to(torch.float32)
    comp_size = scatter_sum(active_f, labels, num_nodes)  # per-label size
    pairs = comp_size * (comp_size - 1.0) / 2.0
    # attribute each component's pair-count to its label's graph (label id is a
    # real node id, so batch[label] is well-defined)
    label_batch = batch  # labels index nodes; comp_size is indexed by node-id-as-label
    return scatter_sum(pairs * (comp_size > 0), label_batch, num_graphs)
