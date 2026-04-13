"""
Component graph construction, pruning, and MST.

Steps:
  owner_map + dist_map + prev_y + prev_x
    → find_meeting_points()   : {(A,B): {cost, y, x, y_b, x_b, path}}
    → build_component_graph() : nx.Graph with edge weights and paths
    → prune_edges()           : remove high-cost edges
    → minimum_spanning_forest(): nx.Graph MST
"""

from typing import Optional

import networkx as nx
import numpy as np


def _backtrack(
    y: int,
    x: int,
    prev_y: np.ndarray,
    prev_x: np.ndarray,
) -> list[tuple[int, int]]:
    """Trace the Dijkstra path from pixel (y, x) back to its annotation seed."""
    path = []
    while y >= 0 and x >= 0:
        path.append((int(y), int(x)))
        py, px = int(prev_y[y, x]), int(prev_x[y, x])
        if py < 0:
            break
        y, x = py, px
    return path


def find_meeting_points(
    owner_map: np.ndarray,
    dist_map: np.ndarray,
    prev_y: Optional[np.ndarray] = None,
    prev_x: Optional[np.ndarray] = None,
) -> dict[tuple[int, int], dict]:
    """
    Find the minimum-cost meeting point for every adjacent component pair.

    A meeting point is any pair of adjacent pixels owned by different
    components. The meeting cost is dist[A] + dist[B] (total path cost
    from both sides). For each pair (A, B), only the cheapest meeting
    point is kept.

    If prev_y / prev_x are provided (from multi_source_dijkstra), the full
    backtracked path seed_A → meeting → seed_B is stored in connections['path'].

    Args:
        owner_map: int32 (H, W) — component ownership (0=unvisited)
        dist_map:  float32 (H, W) — accumulated Dijkstra cost
        prev_y:    int32 (H, W) — predecessor row (-1 for seeds); optional
        prev_x:    int32 (H, W) — predecessor col (-1 for seeds); optional

    Returns:
        connections: dict {(A, B): {'cost': float, 'y': int, 'x': int,
                                    'y_b': int, 'x_b': int,
                                    'path': list[tuple[int,int]]}}
                     where A < B; 'path' only present when prev_y/prev_x given
    """
    shifts = [(0, 1), (1, 0), (1, 1), (1, -1)]
    connections: dict[tuple[int, int], dict] = {}

    for dy, dx in shifts:
        if dy > 0:
            oa = owner_map[:-dy, :]
            ob = owner_map[dy:, :]
            da = dist_map[:-dy, :]
            db = dist_map[dy:, :]
            ys_offset = 0
        else:
            oa = ob = owner_map
            da = db = dist_map
            ys_offset = 0

        if dx > 0:
            oa = oa[:, :-dx]
            ob = ob[:, dx:]
            da = da[:, :-dx]
            db = db[:, dx:]
            xs_offset = 0
        elif dx < 0:
            oa = oa[:, -dx:]
            ob = ob[:, :dx]
            da = da[:, -dx:]
            db = db[:, :dx]
            xs_offset = -dx
        else:
            xs_offset = 0

        meet = (oa > 0) & (ob > 0) & (oa != ob)
        ys_local, xs_local = np.where(meet)
        if len(ys_local) == 0:
            continue

        a_ids = oa[ys_local, xs_local].astype(int)
        b_ids = ob[ys_local, xs_local].astype(int)
        costs = da[ys_local, xs_local] + db[ys_local, xs_local]
        global_y = ys_local + ys_offset
        global_x = xs_local + xs_offset
        # B-side pixel is always (A_y + dy, A_x + dx)
        global_y_b = global_y + dy
        global_x_b = global_x + dx

        for a, b, cost, gy, gx, gyb, gxb in zip(
            a_ids.tolist(),
            b_ids.tolist(),
            costs.tolist(),
            global_y.tolist(),
            global_x.tolist(),
            global_y_b.tolist(),
            global_x_b.tolist(),
        ):
            key = (min(a, b), max(a, b))
            if key not in connections or cost < connections[key]["cost"]:
                connections[key] = {
                    "cost": float(cost),
                    "y": int(gy),
                    "x": int(gx),
                    "y_b": int(gyb),
                    "x_b": int(gxb),
                }

    # Backtrack paths if predecessor maps are provided
    if prev_y is not None and prev_x is not None:
        for key, info in connections.items():
            path_a = _backtrack(info["y"],   info["x"],   prev_y, prev_x)
            path_b = _backtrack(info["y_b"], info["x_b"], prev_y, prev_x)
            # seed_A → ... → meeting_A → meeting_B → ... → seed_B
            info["path"] = list(reversed(path_a)) + path_b

    return connections


def build_component_graph(
    connections: dict[tuple[int, int], dict],
    n_components: int,
) -> nx.Graph:
    """
    Build a component-level graph from meeting points.

    Nodes  = annotation component IDs (1 .. n_components)
    Edges  = meeting points, weighted by minimum meeting cost

    Args:
        connections:  output of find_meeting_points()
        n_components: total number of annotation components

    Returns:
        G: nx.Graph with edge attributes 'weight', 'y', 'x', 'path'
    """
    G = nx.Graph()
    G.add_nodes_from(range(1, n_components + 1))

    for (a, b), info in connections.items():
        if G.has_edge(a, b):
            if info["cost"] < G[a][b]["weight"]:
                G[a][b].update(
                    weight=info["cost"],
                    y=info["y"],
                    x=info["x"],
                    path=info.get("path", []),
                )
        else:
            G.add_edge(
                a, b,
                weight=info["cost"],
                y=info["y"],
                x=info["x"],
                path=info.get("path", []),
            )

    return G


def prune_edges(
    G: nx.Graph,
    threshold: Optional[float] = None,
    prune_k: Optional[float] = None,
) -> nx.Graph:
    """
    Remove edges whose weight exceeds a threshold.

    Provide either a fixed `threshold` or `prune_k` to compute
    threshold = mean + prune_k * std of all edge weights.

    Args:
        G:         component graph
        threshold: fixed cost threshold (overrides prune_k if given)
        prune_k:   std multiplier for automatic threshold

    Returns:
        G_pruned: copy of G with high-cost edges removed
    """
    edge_costs = np.array([d["weight"] for _, _, d in G.edges(data=True)])

    if threshold is None:
        if prune_k is None:
            raise ValueError("Provide either threshold or prune_k")
        threshold = float(edge_costs.mean() + prune_k * edge_costs.std())

    G_pruned = G.copy()
    edges_to_remove = [
        (a, b) for a, b, d in G_pruned.edges(data=True) if d["weight"] > threshold
    ]
    G_pruned.remove_edges_from(edges_to_remove)
    return G_pruned


def minimum_spanning_forest(G: nx.Graph) -> nx.Graph:
    """
    Compute the minimum spanning forest of a graph.

    Works on disconnected graphs (produces one MST per connected component).

    Args:
        G: pruned component graph

    Returns:
        mst: nx.Graph — minimum spanning forest
    """
    return nx.minimum_spanning_tree(G, weight="weight")
