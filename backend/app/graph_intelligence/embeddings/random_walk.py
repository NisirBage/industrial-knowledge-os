"""Biased random walk generator shared by Node2Vec and DeepWalk — DeepWalk
is Node2Vec's own degenerate case (p=q=1, an unbiased walk), not a
separately-implemented algorithm; duplicating the walk logic for what's
mathematically the same procedure would be the kind of "model-specific
implementation" the M9 spec explicitly asks to avoid favoring.
"""

import random

import networkx as nx

_DEFAULT_WALK_LENGTH = 20
_DEFAULT_NUM_WALKS = 10


def generate_walks(
    graph: nx.Graph,
    walk_length: int = _DEFAULT_WALK_LENGTH,
    num_walks: int = _DEFAULT_NUM_WALKS,
    p: float = 1.0,
    q: float = 1.0,
    seed: int = 42,
) -> list[list[str]]:
    """p=q=1.0 (the default) is an unbiased walk — DeepWalk. p<1 biases
    toward returning to the previous node (more local/BFS-like); q<1 biases
    toward moving further away (more DFS-like) — the real Node2Vec
    second-order biased walk when p != 1 or q != 1."""
    rng = random.Random(seed)
    nodes = sorted(graph.nodes())
    walks: list[list[str]] = []

    for _ in range(num_walks):
        for start in nodes:
            walks.append(_single_walk(graph, start, walk_length, p, q, rng))
    return walks


def _single_walk(graph: nx.Graph, start: str, walk_length: int, p: float, q: float, rng: random.Random) -> list[str]:
    walk = [start]
    while len(walk) < walk_length:
        current = walk[-1]
        neighbors = list(graph.neighbors(current))
        if not neighbors:
            break
        if len(walk) == 1:
            walk.append(rng.choice(neighbors))
            continue

        previous = walk[-2]
        weights = []
        for neighbor in neighbors:
            if neighbor == previous:
                weights.append(1.0 / p)
            elif graph.has_edge(neighbor, previous):
                weights.append(1.0)
            else:
                weights.append(1.0 / q)
        total = sum(weights)
        probabilities = [w / total for w in weights]
        walk.append(rng.choices(neighbors, weights=probabilities, k=1)[0])
    return walk
