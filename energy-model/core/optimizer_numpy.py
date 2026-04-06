from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from core.build_edges import build_edges_principal_curves

Array = np.ndarray


@dataclass(frozen=True)
class NumpyParams:
    p: float = 2.0
    lambda1: float = 0.1  # edge length
    lambda_curv: float = 0.2  # curvature regularization
    lambda_train: float = 0.1  # train-graph max shortest-path regularization
    n_train_lines: int = 3
    line_overlap_penalty: float = 1.25
    line_candidate_pool: int = 28
    min_line_hops: int = 2
    eps: float = 1e-8
    lr: float = 2e-2
    iters: int = 200
    fd_eps: float = 1e-4
    disconnected_penalty: float = 100.0
    enforce_principal_curve_topology: bool = False
    n_principal_curves: int = 1


def assign_points(X: Array, Y: Array) -> Tuple[Array, Array]:
    d2 = np.sum((Y[:, None, :] - X[None, :, :]) ** 2, axis=-1)
    assign_idx = np.argmin(d2, axis=1).astype(np.int32)
    d2_min = np.min(d2, axis=1)
    return assign_idx, d2_min


def _build_weighted_adj(X: Array, edges: Array) -> List[List[Tuple[int, float]]]:
    n = int(X.shape[0])
    adj: List[List[Tuple[int, float]]] = [[] for _ in range(n)]
    for a, b in np.asarray(edges, dtype=np.int32):
        w = float(np.linalg.norm(X[a] - X[b]))
        adj[a].append((int(b), w))
        adj[b].append((int(a), w))
    return adj


def _dijkstra(adj: List[List[Tuple[int, float]]], start: int) -> Tuple[Array, Array]:
    n = len(adj)
    dist = np.full(n, np.inf, dtype=np.float64)
    prev = np.full(n, -1, dtype=np.int32)
    dist[start] = 0.0

    pq: List[Tuple[float, int]] = [(0.0, start)]
    while pq:
        d_u, u = heapq.heappop(pq)
        if d_u > dist[u]:
            continue
        for v, w in adj[u]:
            nd = d_u + w
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))

    return dist, prev


def graph_diameter_cost(
    X: Array,
    edges: Array,
    disconnected_penalty: float,
) -> Tuple[float, Dict[str, object]]:
    if edges.size == 0 or X.shape[0] <= 1:
        info: Dict[str, object] = {"start": 0, "end": 0, "path": [0], "diameter": 0.0}
        return 0.0, info

    adj = _build_weighted_adj(X, edges)
    n = len(adj)

    max_dist = -1.0
    best_start = 0
    best_end = 0
    best_prev: Optional[Array] = None
    disconnected = False

    for s in range(n):
        dist, prev = _dijkstra(adj, s)
        finite_mask = np.isfinite(dist)
        if not np.all(finite_mask):
            disconnected = True
        if np.any(finite_mask):
            t = int(np.argmax(np.where(finite_mask, dist, -1.0)))
            d = float(dist[t])
            if d > max_dist:
                max_dist = d
                best_start = s
                best_end = t
                best_prev = prev

    if max_dist < 0.0:
        max_dist = float(disconnected_penalty)

    penalty = max_dist + (disconnected_penalty if disconnected else 0.0)

    path: List[int] = []
    if best_prev is not None:
        cur = best_end
        path.append(cur)
        while cur != best_start and cur != -1:
            cur = int(best_prev[cur])
            if cur != -1:
                path.append(cur)
        path.reverse()

    info = {
        "start": best_start,
        "end": best_end,
        "path": path,
        "diameter": max_dist,
        "disconnected": disconnected,
    }
    return float(penalty), info


def _reconstruct_path(prev: Array, start: int, end: int) -> List[int]:
    path: List[int] = []
    cur = end
    path.append(cur)
    while cur != start and cur != -1:
        cur = int(prev[cur])
        if cur != -1:
            path.append(cur)
    path.reverse()
    if not path or path[0] != start:
        return []
    return path


def multi_train_lines_cost(
    X: Array,
    edges: Array,
    n_lines: int,
    disconnected_penalty: float,
    line_overlap_penalty: float,
) -> Tuple[float, Dict[str, object]]:
    if edges.size == 0 or X.shape[0] <= 1:
        info: Dict[str, object] = {
            "lines": [{"start": 0, "end": 0, "path": [0], "distance": 0.0}],
            "mean_line_distance": 0.0,
            "disconnected": False,
        }
        return 0.0, info

    adj = _build_weighted_adj(X, edges)
    n = len(adj)
    n_lines = max(1, int(n_lines))

    prev_map: Dict[int, Array] = {}
    pairs: List[Tuple[float, int, int]] = []
    disconnected = False

    for s in range(n):
        dist, prev = _dijkstra(adj, s)
        prev_map[s] = prev
        finite_mask = np.isfinite(dist)
        if not np.all(finite_mask):
            disconnected = True
        for t in range(s + 1, n):
            if np.isfinite(dist[t]):
                pairs.append((float(dist[t]), s, t))

    if not pairs:
        penalty = float(disconnected_penalty)
        info = {
            "lines": [],
            "mean_line_distance": 0.0,
            "disconnected": True,
        }
        return penalty, info

    pairs.sort(reverse=True)

    selected: List[Dict[str, object]] = []
    used_edges: set[Tuple[int, int]] = set()

    for dist, s, t in pairs:
        if len(selected) >= n_lines:
            break

        path = _reconstruct_path(prev_map[s], s, t)
        if len(path) < 2:
            continue

        edge_set: set[Tuple[int, int]] = set()
        for i in range(len(path) - 1):
            a = int(path[i])
            b = int(path[i + 1])
            edge_set.add((a, b) if a < b else (b, a))

        overlap = 0.0
        if edge_set:
            overlap = len(edge_set & used_edges) / len(edge_set)

        score = dist * (1.0 - line_overlap_penalty * overlap)
        if score <= 0.0:
            continue

        selected.append(
            {
                "start": s,
                "end": t,
                "distance": float(dist),
                "score": float(score),
                "path": path,
                "overlap": float(overlap),
            }
        )
        used_edges |= edge_set

    if not selected:
        best_dist, s, t = pairs[0]
        selected.append(
            {
                "start": s,
                "end": t,
                "distance": float(best_dist),
                "score": float(best_dist),
                "path": _reconstruct_path(prev_map[s], s, t),
                "overlap": 0.0,
            }
        )

    mean_line_distance = float(np.mean([line["distance"] for line in selected]))
    mean_overlap = float(np.mean([line["overlap"] for line in selected]))
    penalty = mean_line_distance + line_overlap_penalty * mean_overlap
    if disconnected:
        penalty += float(disconnected_penalty)

    info = {
        "lines": selected,
        "mean_line_distance": mean_line_distance,
        "mean_overlap": mean_overlap,
        "disconnected": disconnected,
    }
    return float(penalty), info


def energy_fit(X: Array, Y: Array, assign_idx: Array, params: NumpyParams) -> float:
    X_assigned = X[assign_idx]
    r2 = np.sum((Y - X_assigned) ** 2, axis=-1)
    if params.p == 2.0:
        dist_p = r2
    else:
        dist = np.sqrt(r2 + params.eps)
        dist_p = dist**params.p
    return float(np.mean(dist_p))


def energy_len(X: Array, edges: Array, params: NumpyParams) -> float:
    if edges.size == 0:
        return 0.0
    a = edges[:, 0]
    b = edges[:, 1]
    r = np.sqrt(np.sum((X[a] - X[b]) ** 2, axis=-1) + params.eps)
    return float(params.lambda1 * np.sum(r))


def energy_curvature(X: Array, edges: Array, params: NumpyParams) -> float:
    if edges.size == 0:
        return 0.0

    n = int(X.shape[0])
    neigh: List[List[int]] = [[] for _ in range(n)]
    for a, b in np.asarray(edges, dtype=np.int32):
        neigh[int(a)].append(int(b))
        neigh[int(b)].append(int(a))

    curv = 0.0
    count = 0
    for i in range(n):
        if not neigh[i]:
            continue
        mean_nb = np.mean(X[neigh[i]], axis=0)
        d = X[i] - mean_nb
        curv += float(np.dot(d, d))
        count += 1

    if count == 0:
        return 0.0
    return float(params.lambda_curv * (curv / count))


def energy_train_maxdist(X: Array, edges: Array, params: NumpyParams) -> Tuple[float, Dict[str, object]]:
    dcost, info = multi_train_lines_cost(
        X=X,
        edges=edges,
        n_lines=params.n_train_lines,
        disconnected_penalty=params.disconnected_penalty,
        line_overlap_penalty=params.line_overlap_penalty,
    )
    return float(params.lambda_train * dcost), info


def energy_total(
    X: Array,
    edges: Array,
    Y: Array,
    assign_idx: Array,
    params: NumpyParams,
) -> Tuple[float, Dict[str, float], Dict[str, object]]:
    ef = energy_fit(X, Y, assign_idx, params)
    el = energy_len(X, edges, params)
    ec = energy_curvature(X, edges, params)
    etrain, train_info = energy_train_maxdist(X, edges, params)

    total = ef + el + ec + etrain
    terms = {
        "fit": ef,
        "len": el,
        "curvature": ec,
        "train_maxdist": etrain,
        "total": total,
    }
    return total, terms, train_info


def finite_difference_grad(
    X: Array,
    edges: Array,
    Y: Array,
    assign_idx: Array,
    params: NumpyParams,
) -> Array:
    grad = np.zeros_like(X, dtype=np.float64)
    h = float(params.fd_eps)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Xp = X.copy()
            Xm = X.copy()
            Xp[i, j] += h
            Xm[i, j] -= h

            lp, _, _ = energy_total(Xp, edges, Y, assign_idx, params)
            lm, _, _ = energy_total(Xm, edges, Y, assign_idx, params)
            grad[i, j] = (lp - lm) / (2.0 * h)

    return grad.astype(X.dtype)


def optimize_numpy(
    X0: Array,
    edges: Array,
    Y: Array,
    params: NumpyParams,
) -> Tuple[Array, list[dict[str, float]], Dict[str, object]]:
    """
    Alternating minimization without JAX:
      1) hard assign data points to nearest nodes
      2) finite-difference gradient descent step on node positions
    """
    X = np.asarray(X0, dtype=np.float64).copy()
    edges = np.asarray(edges, dtype=np.int32)
    Y = np.asarray(Y, dtype=np.float64)

    history: list[dict[str, float]] = []
    last_train_info: Dict[str, object] = {}
    edges_cur = edges.copy()

    for t in range(params.iters):
        if params.enforce_principal_curve_topology:
            edges_cur = build_edges_principal_curves(
                np.asarray(X, dtype=np.float32),
                n_curves=params.n_principal_curves,
            )

        assign_idx, _ = assign_points(X, Y)
        grad = finite_difference_grad(X, edges_cur, Y, assign_idx, params)
        X -= params.lr * grad

        if t % 10 == 0 or t == params.iters - 1:
            _, terms, train_info = energy_total(X, edges_cur, Y, assign_idx, params)
            row = {k: float(v) for k, v in terms.items()}
            row["iter"] = float(t)
            history.append(row)
            last_train_info = train_info

    return X.astype(np.float32), history, last_train_info
