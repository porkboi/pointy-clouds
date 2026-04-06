from __future__ import annotations

from typing import Literal

import numpy as np


def _canonicalize_edges(edges: np.ndarray, n: int) -> np.ndarray:
    """Return unique undirected edges in int32 with a < b."""
    if edges.size == 0:
        return np.zeros((0, 2), dtype=np.int32)

    e = np.asarray(edges, dtype=np.int64)
    a = np.minimum(e[:, 0], e[:, 1])
    b = np.maximum(e[:, 0], e[:, 1])
    keep = (a >= 0) & (b >= 0) & (a < n) & (b < n) & (a != b)
    out = np.stack([a[keep], b[keep]], axis=1)
    if out.size == 0:
        return np.zeros((0, 2), dtype=np.int32)

    out = np.unique(out, axis=0)
    return out.astype(np.int32)


def build_edges_knn(X: np.ndarray, k: int = 3) -> np.ndarray:
    """
    Build undirected kNN edges for points X in R^d.
    Returns edges with shape (|E|, 2), int32, canonicalized (a < b).
    """
    n = int(X.shape[0])
    if n <= 1 or k <= 0:
        return np.zeros((0, 2), dtype=np.int32)

    k = min(k, n - 1)
    d2 = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=-1)
    np.fill_diagonal(d2, np.inf)
    nbrs = np.argpartition(d2, kth=k - 1, axis=1)[:, :k]

    src = np.repeat(np.arange(n, dtype=np.int32), k)
    dst = nbrs.reshape(-1).astype(np.int32)
    edges = np.stack([src, dst], axis=1)
    return _canonicalize_edges(edges, n)


def build_edges_mst(X: np.ndarray) -> np.ndarray:
    """
    Build Euclidean MST edges with Prim's algorithm in O(n^2).
    Good default for connected sparse graphs without extra dependencies.
    """
    n = int(X.shape[0])
    if n <= 1:
        return np.zeros((0, 2), dtype=np.int32)

    d2 = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=-1)
    np.fill_diagonal(d2, np.inf)

    in_tree = np.zeros(n, dtype=bool)
    in_tree[0] = True

    parent = np.zeros(n, dtype=np.int32)
    best = d2[0].copy()

    edges = []
    for _ in range(n - 1):
        candidates = np.where(~in_tree)[0]
        u = candidates[np.argmin(best[candidates])]
        v = parent[u]
        edges.append((u, v))
        in_tree[u] = True

        better = d2[u] < best
        best[better] = d2[u][better]
        parent[better] = u

    return _canonicalize_edges(np.asarray(edges, dtype=np.int32), n)


def build_edges_principal_curve(X: np.ndarray) -> np.ndarray:
    """
    Build a strict principal-curve spine as a single chain:
      1) project nodes onto first principal component
      2) connect consecutive nodes in projection order
    Returns undirected edges (a < b).
    """
    n = int(X.shape[0])
    if n <= 1:
        return np.zeros((0, 2), dtype=np.int32)

    Xf = np.asarray(X, dtype=np.float64)
    mu = np.mean(Xf, axis=0, keepdims=True)
    Xc = Xf - mu

    cov = (Xc.T @ Xc) / max(1, n - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    pc1 = eigvecs[:, int(np.argmax(eigvals))]

    t = Xc @ pc1
    order = np.argsort(t, kind="mergesort")

    src = order[:-1].astype(np.int32)
    dst = order[1:].astype(np.int32)
    edges = np.stack([src, dst], axis=1)
    return _canonicalize_edges(edges, n)


def build_edges_principal_curves(X: np.ndarray, n_curves: int = 1) -> np.ndarray:
    """
    Build multiple disjoint principal-curve chains:
      - order nodes by projection onto first principal component
      - split the ordered sequence into n_curves contiguous segments at largest gaps
      - connect within each segment
      - add a shared hub stop so curves meet at least once
    """
    n = int(X.shape[0])
    n_curves = max(1, int(n_curves))
    if n <= 1:
        return np.zeros((0, 2), dtype=np.int32)
    if n_curves <= 1:
        return build_edges_principal_curve(X)

    Xf = np.asarray(X, dtype=np.float64)
    mu = np.mean(Xf, axis=0, keepdims=True)
    Xc = Xf - mu
    cov = (Xc.T @ Xc) / max(1, n - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    pc1 = eigvecs[:, int(np.argmax(eigvals))]

    t = Xc @ pc1
    order = np.argsort(t, kind="mergesort")
    t_sorted = t[order]

    # Split at largest projection jumps to form separate principal curves.
    jumps = np.diff(t_sorted)
    split_budget = min(n_curves - 1, max(0, n - 1))
    if split_budget == 0:
        return build_edges_principal_curve(X)
    split_pos = np.argpartition(-jumps, kth=split_budget - 1)[:split_budget]
    split_after = sorted(int(p) for p in split_pos)

    edges_list: list[tuple[int, int]] = []
    segments: list[np.ndarray] = []
    start = 0
    for end_after in split_after + [n - 1]:
        end = int(end_after) + 1
        seg = order[start:end]
        segments.append(seg)
        if seg.shape[0] >= 2:
            src = seg[:-1].astype(np.int32)
            dst = seg[1:].astype(np.int32)
            edges_list.extend((int(a), int(b)) for a, b in zip(src, dst))
        start = end

    # Common stop: pick a hub around the median PC1 position, then connect each
    # other segment to the hub via its nearest node.
    hub = int(order[n // 2])
    Xf_hub = Xf[hub]
    for seg in segments:
        if seg.size == 0:
            continue
        if np.any(seg == hub):
            continue
        pts = Xf[seg]
        d2 = np.sum((pts - Xf_hub[None, :]) ** 2, axis=1)
        join = int(seg[int(np.argmin(d2))])
        edges_list.append((hub, join))

    if not edges_list:
        return np.zeros((0, 2), dtype=np.int32)
    return _canonicalize_edges(np.asarray(edges_list, dtype=np.int32), n)


def build_edges(
    X: np.ndarray,
    method: Literal["knn", "mst", "principal_curve"] = "principal_curve",
    k: int = 3,
    n_principal_curves: int = 1,
) -> np.ndarray:
    if method == "knn":
        return build_edges_knn(X, k=k)
    if method == "mst":
        return build_edges_mst(X)
    if method == "principal_curve":
        return build_edges_principal_curves(X, n_curves=n_principal_curves)
    raise ValueError(f"Unknown edge build method: {method}")
