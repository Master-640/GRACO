"""Heuristic baselines and a shared greedy-rollout evaluator.

Baselines score nodes with a simple rule (random / static degree / adaptive
degree) and run the *same* vectorized env loop as the learned policy, so their
objective values are directly comparable.
"""

from __future__ import annotations

from typing import Callable, Dict

import torch
from torch import Tensor

from graco.data.batch import BatchedGraph
from graco.envs.base import Observation, VectorizedEnv
from graco.utils.scatter import scatter_sum
from graco.utils.segment_ops import segment_argmax

ScoreFn = Callable[[Observation], Tensor]  # obs -> per-node scores [N]


def random_score(obs: Observation) -> Tensor:
    return torch.rand(obs.num_nodes, device=obs.graph.device)


def degree_score(obs: Observation) -> Tensor:
    return obs.graph.degree()


def adaptive_degree_score(obs: Observation) -> Tensor:
    """Degree over the still-active subgraph (uses valid endpoints)."""
    g = obs.graph
    valid = obs.action_mask
    src, dst = g.edge_index[0], g.edge_index[1]
    live = (valid[src] & valid[dst]).float()
    return scatter_sum(live, src, g.num_nodes)


BASELINES: Dict[str, ScoreFn] = {
    "random": random_score,
    "degree": degree_score,
    "adaptive_degree": adaptive_degree_score,
}


@torch.no_grad()
def run_episode(env: VectorizedEnv, graph: BatchedGraph, score_fn: ScoreFn) -> Dict[str, Tensor]:
    """Roll out one batched episode picking argmax(score) each step.

    Returns per-graph episode return ``[B]`` and final objective ``[B]``.
    """
    obs = env.reset(graph)
    ret = torch.zeros(env.num_envs, device=env.device)
    steps = 0
    max_steps = int(graph.graph_num_nodes.max().item()) + 2
    while not obs.done.all():
        scores = score_fn(obs)
        action = segment_argmax(scores, obs.graph.batch, obs.graph.num_graphs, obs.action_mask)
        step = env.step(action)
        ret += step.reward
        obs = step.obs
        steps += 1
        if steps > max_steps:
            break
    objective = env.objective(env.graph, env.state)
    return {"return": ret, "objective": objective, "steps": steps}


def baseline_score_fn(name: str) -> ScoreFn:
    if name not in BASELINES:
        raise KeyError(f"Unknown baseline '{name}'. Available: {sorted(BASELINES)}")
    return BASELINES[name]
