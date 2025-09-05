"""
Adapted from Pham et al. (2024):
Root Cause Analysis for Microservices based on Causal Inference: How Far Are We?
Original implementation: http://anonymous.4open.science/r/ase24-cfm/cfm/graph_heads/random_walk.py

This file adapts the original random_walk implementation to remove dependencies on cfm-specific classes
(CaseData, Graph, MemoryGraph, Node) and pandas. It uses only numpy and networkx, and works with node names as strings.

Changes:
- Removed cfm.classes.data, cfm.classes.graph, and pandas dependencies
- Uses networkx.DiGraph and node names as strings
- Simplified random walk logic
"""

import numpy as np
import networkx as nx
from typing import List, Optional, Tuple


def _times(num: int, multiplier: int = 10) -> int:
    return num * multiplier


def random_walk(
    adj: np.ndarray,
    node_names: Optional[List[str]] = None,
    num_loop: Optional[int] = None,
    seed: int = 42,
) -> List[Tuple[str, float]]:
    """
    Adapted random walk algorithm for root cause ranking.
    - adj: adjacency matrix (numpy array)
    - node_names: list of node names (strings)
    - num_loop: number of walk steps (default: 10 * number of nodes)
    - seed: random seed
    Returns: list of (node_name, score) tuples sorted by score descending
    """
    if node_names is None:
        node_names = [f"X{i}" for i in range(len(adj))]
    node_num = len(node_names)
    if num_loop is None:
        num_loop = node_num * 10
    # Build directed graph
    G = nx.DiGraph()
    for i, name in enumerate(node_names):
        G.add_node(name)
    for a in range(node_num):
        for b in range(node_num):
            if adj[a, b] != 0:
                G.add_edge(node_names[b], node_names[a])  # Invert direction to match original
    G = G.reverse()
    # Random walk
    rng = np.random.RandomState(seed)
    current_node = node_names[rng.randint(0, node_num)]
    visits = {node: 0 for node in node_names}
    for _ in range(num_loop):
        neighbors = list(G.successors(current_node))
        if not neighbors:
            current_node = node_names[rng.randint(0, node_num)]
        else:
            current_node = neighbors[rng.randint(0, len(neighbors))]
        visits[current_node] += 1
    scores = [(node, count / num_loop) for node, count in visits.items()]
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores
