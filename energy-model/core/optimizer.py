from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from core.build_edges import build_edges_principal_curves

Array = jnp.ndarray


@dataclass(frozen=True)
class Params:
    p: float = 2.0
    lambda1: float = 0.1  # edge length
    lambda2: float = 0.0  # components (kept out of grad path)
    lambda_curv: float = 0.0
    lambda_T: float = 0.0
    w_access: float = 1.0
    w_train: float = 0.2
    pair_count: int = 4096
    pair_mode: str = "mixed"  # local | global | mixed
    local_k: int = 32
    outer_iters: int = 500
    inner_steps: int = 1
    disconnected_penalty: float = 1e3
    enforce_principal_curve_topology: bool = False
    n_principal_curves: int = 1
    eps: float = 1e-8
    lr: float = 1e-2
    iters: int = 500  # backward-compatible alias; ignored when outer_iters > 0


HookEnergy = Callable[[Array, Array, Array, Array, Params], Dict[str, Array]]
# signature: hook_energy(X, edges, Y, assign_idx, params) -> {"term_name": scalar, ...}


def sample_pairs(
    Y: Array,
    pair_count: int,
    mode: str = "mixed",
    local_k: int = 32,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    m = int(Y.shape[0])
    if m <= 1 or pair_count <= 0:
        return np.zeros((0, 2), dtype=np.int32)

    Y_np = np.asarray(Y, dtype=np.float64)
    rng = rng if rng is not None else np.random.default_rng(0)

    def _sample_global(count: int) -> np.ndarray:
        i = rng.integers(0, m, size=count, endpoint=False)
        k = rng.integers(0, m, size=count, endpoint=False)
        same = i == k
        if np.any(same):
            k[same] = (k[same] + 1) % m
        return np.stack([i, k], axis=1).astype(np.int32)

    def _sample_local(count: int) -> np.ndarray:
        if count <= 0:
            return np.zeros((0, 2), dtype=np.int32)
        d2 = np.sum((Y_np[:, None, :] - Y_np[None, :, :]) ** 2, axis=-1)
        np.fill_diagonal(d2, np.inf)
        k_eff = max(1, min(int(local_k), m - 1))
        neigh = np.argpartition(d2, kth=k_eff - 1, axis=1)[:, :k_eff]

        i = rng.integers(0, m, size=count, endpoint=False)
        pick = rng.integers(0, k_eff, size=count, endpoint=False)
        k = neigh[i, pick]
        return np.stack([i, k], axis=1).astype(np.int32)

    mode = mode.lower()
    if mode == "global":
        return _sample_global(pair_count)
    if mode == "local":
        return _sample_local(pair_count)
    if mode != "mixed":
        raise ValueError(f"Unknown pair sampling mode: {mode}")

    local_count = pair_count // 2
    global_count = pair_count - local_count
    return np.concatenate([_sample_local(local_count), _sample_global(global_count)], axis=0)


def assign_points(X: Array, Y: Array, eps: float = 1e-8) -> Tuple[Array, Array]:
    """
    Hard assignment: for each y_i pick nearest x_j in squared Euclidean distance.

    Returns:
      assign_idx: (m,) int32
      access_dist: (m,) float
    """
    d2 = jnp.sum((Y[:, None, :] - X[None, :, :]) ** 2, axis=-1)  # (m, n)
    assign_idx = jnp.argmin(d2, axis=1).astype(jnp.int32)
    d2_min = jnp.min(d2, axis=1)
    access_dist = jnp.sqrt(d2_min + eps)
    return assign_idx, access_dist


def build_weighted_adj(X: np.ndarray, edges: np.ndarray, eps: float) -> List[List[Tuple[int, float]]]:
    n = int(X.shape[0])
    adj: List[List[Tuple[int, float]]] = [[] for _ in range(n)]
    for a, b in np.asarray(edges, dtype=np.int32):
        w = float(np.sqrt(np.sum((X[a] - X[b]) ** 2) + eps))
        adj[int(a)].append((int(b), w))
        adj[int(b)].append((int(a), w))
    return adj


def _dijkstra(adj: List[List[Tuple[int, float]]], start: int) -> np.ndarray:
    n = len(adj)
    dist = np.full(n, np.inf, dtype=np.float64)
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
                heapq.heappush(pq, (nd, v))

    return dist


def dijkstra_multi_source(
    X: np.ndarray,
    edges: np.ndarray,
    sources: np.ndarray,
    targets: np.ndarray,
    eps: float,
    disconnected_penalty: float,
) -> np.ndarray:
    if sources.size == 0:
        return np.zeros((0,), dtype=np.float32)
    if X.shape[0] <= 1:
        return np.zeros_like(sources, dtype=np.float32)

    adj = build_weighted_adj(np.asarray(X, dtype=np.float64), np.asarray(edges, dtype=np.int32), eps)
    out = np.full(sources.shape[0], float(disconnected_penalty), dtype=np.float64)

    unique_sources = np.unique(sources.astype(np.int32))
    dist_map: Dict[int, np.ndarray] = {}
    for s in unique_sources:
        dist_map[int(s)] = _dijkstra(adj, int(s))

    for t in range(sources.shape[0]):
        s = int(sources[t])
        d = int(targets[t])
        val = dist_map[s][d]
        out[t] = float(val) if np.isfinite(val) else float(disconnected_penalty)

    return out.astype(np.float32)


def energy_fit(X: Array, Y: Array, assign_idx: Array, params: Params) -> Array:
    X_assigned = X[assign_idx]  # (m, d)
    r2 = jnp.sum((Y - X_assigned) ** 2, axis=-1)
    if params.p == 2.0:
        dist_p = r2
    else:
        dist = jnp.sqrt(r2 + params.eps)
        dist_p = dist**params.p
    return jnp.mean(dist_p)


def energy_len(X: Array, edges: Array, params: Params) -> Array:
    if edges.size == 0:
        return jnp.array(0.0, dtype=X.dtype)

    a = edges[:, 0]
    b = edges[:, 1]
    d = X[a] - X[b]
    r = jnp.sqrt(jnp.sum(d * d, axis=-1) + params.eps)
    return params.lambda1 * jnp.sum(r)


def energy_curvature(X: Array, edges: Array, params: Params) -> Array:
    if params.lambda_curv <= 0.0 or edges.size == 0:
        return jnp.array(0.0, dtype=X.dtype)

    n = X.shape[0]
    a = edges[:, 0]
    b = edges[:, 1]

    row = jnp.concatenate([a, b], axis=0)
    col = jnp.concatenate([b, a], axis=0)
    data = jnp.ones((row.shape[0],), dtype=X.dtype)
    adj = jnp.zeros((n, n), dtype=X.dtype).at[row, col].set(data)

    deg = jnp.sum(adj, axis=1, keepdims=True)
    neigh_sum = adj @ X
    mean_nb = neigh_sum / jnp.maximum(deg, 1.0)

    valid = (deg[:, 0] > 0.0).astype(X.dtype)
    diff2 = jnp.sum((X - mean_nb) ** 2, axis=1)
    denom = jnp.maximum(jnp.sum(valid), 1.0)
    return params.lambda_curv * jnp.sum(diff2 * valid) / denom


def _transport_terms_from_frozen(
    X: Array,
    Y: Array,
    assign_idx: Array,
    pair_idx: Array,
    train_dist_const: Array,
    params: Params,
) -> Dict[str, Array]:
    if pair_idx.size == 0 or params.lambda_T <= 0.0:
        zero = jnp.array(0.0, dtype=X.dtype)
        return {
            "transport": zero,
            "mean_access_cost": zero,
            "mean_train_cost": zero,
            "mean_train_dist": zero,
        }

    i = pair_idx[:, 0]
    k = pair_idx[:, 1]

    access_i = jnp.sqrt(jnp.sum((Y[i] - X[assign_idx[i]]) ** 2, axis=-1) + params.eps)
    access_k = jnp.sqrt(jnp.sum((Y[k] - X[assign_idx[k]]) ** 2, axis=-1) + params.eps)

    access_cost = params.w_access * (access_i + access_k)
    train_cost = params.w_train * train_dist_const
    pair_cost = access_cost + train_cost

    return {
        "transport": params.lambda_T * jnp.mean(pair_cost),
        "mean_access_cost": jnp.mean(access_cost),
        "mean_train_cost": jnp.mean(train_cost),
        "mean_train_dist": jnp.mean(train_dist_const),
    }


def transport_term(
    X: Array,
    Y: Array,
    edges: Array,
    pair_idx: Array,
    params: Params,
) -> Tuple[Array, Dict[str, Array]]:
    """
    MCP transport term:
      c(i,k) = w_access*(access_i + access_k) + w_train*SP(src, dst)
      E_trans = lambda_T * mean(c(i,k))
    """
    if pair_idx.size == 0 or params.lambda_T <= 0.0:
        zero = jnp.array(0.0, dtype=X.dtype)
        return zero, {
            "mean_access_cost": zero,
            "mean_train_cost": zero,
            "mean_train_dist": zero,
        }

    assign_idx, _ = assign_points(X, Y, eps=params.eps)
    pair_idx_np = np.asarray(pair_idx, dtype=np.int32)
    assign_np = np.asarray(assign_idx, dtype=np.int32)
    src = assign_np[pair_idx_np[:, 0]]
    dst = assign_np[pair_idx_np[:, 1]]
    train_dist_np = dijkstra_multi_source(
        X=np.asarray(X, dtype=np.float32),
        edges=np.asarray(edges, dtype=np.int32),
        sources=src,
        targets=dst,
        eps=params.eps,
        disconnected_penalty=params.disconnected_penalty,
    )
    train_dist = jnp.asarray(train_dist_np, dtype=X.dtype)
    terms = _transport_terms_from_frozen(X, Y, assign_idx, pair_idx, train_dist, params)
    return terms["transport"], {
        "mean_access_cost": terms["mean_access_cost"],
        "mean_train_cost": terms["mean_train_cost"],
        "mean_train_dist": terms["mean_train_dist"],
    }


def transport_delta_for_edges(
    X: Array,
    Y: Array,
    edges_old: Array,
    edges_new: Array,
    pair_idx: Array,
    params: Params,
) -> Dict[str, Array]:
    e_old, diag_old = transport_term(X=X, Y=Y, edges=edges_old, pair_idx=pair_idx, params=params)
    e_new, diag_new = transport_term(X=X, Y=Y, edges=edges_new, pair_idx=pair_idx, params=params)
    return {
        "E_trans_old": e_old,
        "E_trans_new": e_new,
        "delta_E_trans": e_new - e_old,
        "old_mean_train_dist": diag_old["mean_train_dist"],
        "new_mean_train_dist": diag_new["mean_train_dist"],
    }


def _compute_frozen_train_dist(
    X: Array,
    Y: Array,
    edges: Array,
    pair_idx_np: np.ndarray,
    params: Params,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if pair_idx_np.size == 0 or params.lambda_T <= 0.0:
        assign_idx_np = np.zeros((Y.shape[0],), dtype=np.int32)
        access_dist_np = np.zeros((Y.shape[0],), dtype=np.float32)
        train_dist = np.zeros((pair_idx_np.shape[0],), dtype=np.float32)
        return assign_idx_np, access_dist_np, train_dist

    assign_idx, access_dist = assign_points(X, Y, eps=params.eps)
    assign_idx_np = np.asarray(assign_idx, dtype=np.int32)
    access_dist_np = np.asarray(access_dist, dtype=np.float32)

    i = pair_idx_np[:, 0]
    k = pair_idx_np[:, 1]
    src = assign_idx_np[i]
    dst = assign_idx_np[k]

    train_dist = dijkstra_multi_source(
        X=np.asarray(X, dtype=np.float32),
        edges=np.asarray(edges, dtype=np.int32),
        sources=src,
        targets=dst,
        eps=params.eps,
        disconnected_penalty=params.disconnected_penalty,
    )
    return assign_idx_np, access_dist_np, train_dist


def energy_total(
    X: Array,
    edges: Array,
    Y: Array,
    assign_idx: Array,
    params: Params,
    hook_energy: Optional[HookEnergy] = None,
    pair_idx: Optional[Array] = None,
    train_dist_const: Optional[Array] = None,
) -> Tuple[Array, Dict[str, Array]]:
    ef = energy_fit(X, Y, assign_idx, params)
    el = energy_len(X, edges, params)
    ec = energy_curvature(X, edges, params)

    terms: Dict[str, Array] = {
        "fit": ef,
        "len": el,
        "curvature": ec,
    }

    extra = jnp.array(0.0, dtype=X.dtype)

    if pair_idx is not None and train_dist_const is not None and params.lambda_T > 0.0:
        trans_terms = _transport_terms_from_frozen(X, Y, assign_idx, pair_idx, train_dist_const, params)
        for k, v in trans_terms.items():
            terms[k] = v
        extra = extra + trans_terms["transport"]

    if hook_energy is not None:
        hook_terms = hook_energy(X, edges, Y, assign_idx, params)
        for k, v in hook_terms.items():
            terms[k] = v
            extra = extra + v

    total = ef + el + ec + extra
    terms["total"] = total
    return total, terms


def optimize(
    key: Array,
    X0: Array,
    edges: Array,
    Y: Array,
    params: Params,
    hook_energy: Optional[HookEnergy] = None,
) -> Tuple[Array, list[dict[str, float]]]:
    """
    Alternating minimization with frozen shortest paths per outer iteration:
      1) update hard assignments
      2) recompute shortest-path distances on current graph
      3) run K gradient steps with train distances frozen as constants
    """
    seed = int(np.asarray(key, dtype=np.uint32).reshape(-1)[0])
    rng = np.random.default_rng(seed)

    outer_iters = int(params.outer_iters if params.outer_iters > 0 else params.iters)
    inner_steps = max(1, int(params.inner_steps))

    pair_idx_np = sample_pairs(
        Y=Y,
        pair_count=params.pair_count,
        mode=params.pair_mode,
        local_k=params.local_k,
        rng=rng,
    )
    pair_idx_j = jnp.asarray(pair_idx_np, dtype=jnp.int32)

    opt = optax.adam(params.lr)
    opt_state = opt.init(X0)

    @jax.jit
    def step(
        X: Array,
        edges_cur: Array,
        opt_state: optax.OptState,
        assign_idx: Array,
        train_dist_const: Array,
    ):
        def loss_fn(X_: Array):
            loss, _ = energy_total(
                X_,
                edges_cur,
                Y,
                assign_idx,
                params,
                hook_energy,
                pair_idx=pair_idx_j,
                train_dist_const=train_dist_const,
            )
            return loss

        loss, grads = jax.value_and_grad(loss_fn)(X)
        updates, opt_state2 = opt.update(grads, opt_state, X)
        X2 = optax.apply_updates(X, updates)
        return X2, opt_state2, loss

    history: list[dict[str, float]] = []
    X = X0
    edges_cur = jnp.asarray(edges, dtype=jnp.int32)

    for t in range(outer_iters):
        if params.enforce_principal_curve_topology:
            edges_np = build_edges_principal_curves(
                np.asarray(X, dtype=np.float32),
                n_curves=params.n_principal_curves,
            )
            edges_cur = jnp.asarray(edges_np, dtype=jnp.int32)

        assign_idx_np, _access_dist_np, train_dist_np = _compute_frozen_train_dist(X, Y, edges_cur, pair_idx_np, params)
        assign_idx = jnp.asarray(assign_idx_np, dtype=jnp.int32)
        train_dist_const = jnp.asarray(train_dist_np, dtype=X.dtype)

        for _ in range(inner_steps):
            X, opt_state, _ = step(X, edges_cur, opt_state, assign_idx, train_dist_const)

        if t % 10 == 0 or t == outer_iters - 1:
            _, terms = energy_total(
                X,
                edges_cur,
                Y,
                assign_idx,
                params,
                hook_energy,
                pair_idx=pair_idx_j,
                train_dist_const=train_dist_const,
            )
            row = {k: float(v) for k, v in terms.items()}
            row["iter"] = float(t)
            history.append(row)

    return X, history
