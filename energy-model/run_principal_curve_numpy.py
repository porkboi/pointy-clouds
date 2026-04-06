from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Tuple

import numpy as np

from core.optimizer_numpy import NumpyParams, optimize_numpy
from run_experiment_numpy import make_city_like_cloud


def _canonicalize_edges(edges: np.ndarray, n: int) -> np.ndarray:
    if edges.size == 0:
        return np.zeros((0, 2), dtype=np.int32)
    e = np.asarray(edges, dtype=np.int64)
    a = np.minimum(e[:, 0], e[:, 1])
    b = np.maximum(e[:, 0], e[:, 1])
    keep = (a >= 0) & (b >= 0) & (a < n) & (b < n) & (a != b)
    out = np.stack([a[keep], b[keep]], axis=1)
    if out.size == 0:
        return np.zeros((0, 2), dtype=np.int32)
    return np.unique(out, axis=0).astype(np.int32)


def init_principal_curve_nodes(
    Y: np.ndarray,
    n_spine_nodes: int,
    n_branches: int,
    branch_length: int,
    branch_scale: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build strict principal-curve topology:
      - one connected spine (chain)
      - optional branch chains attached to interior spine nodes
    """
    Y = np.asarray(Y, dtype=np.float64)
    n_spine_nodes = max(2, int(n_spine_nodes))
    n_branches = max(0, int(n_branches))
    branch_length = max(1, int(branch_length)) if n_branches > 0 else 0

    mu = np.mean(Y, axis=0)
    Yc = Y - mu
    cov = (Yc.T @ Yc) / max(1, Y.shape[0] - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, int(np.argmax(eigvals))]

    t = Yc @ axis
    tmin = float(np.min(t))
    tmax = float(np.max(t))
    tq = np.linspace(tmin, tmax, n_spine_nodes)
    sigma = max((tmax - tmin) / (2.0 * n_spine_nodes), 1e-3)

    spine = np.zeros((n_spine_nodes, Y.shape[1]), dtype=np.float64)
    for i, q in enumerate(tq):
        w = np.exp(-0.5 * ((t - q) / sigma) ** 2)
        wsum = np.sum(w)
        if wsum < 1e-12:
            j = int(np.argmin(np.abs(t - q)))
            spine[i] = Y[j]
        else:
            spine[i] = (w[:, None] * Y).sum(axis=0) / wsum

    nodes: List[np.ndarray] = [spine]
    edge_list: List[Tuple[int, int]] = []
    spine_edge_list: List[Tuple[int, int]] = []

    for i in range(n_spine_nodes - 1):
        e = (i, i + 1)
        edge_list.append(e)
        spine_edge_list.append(e)

    if n_branches > 0:
        # Attach branches to interior spine nodes spaced along the chain.
        if n_spine_nodes <= 2:
            root_ids = np.array([0] * n_branches, dtype=np.int32)
        else:
            root_ids = np.linspace(1, n_spine_nodes - 2, n_branches, dtype=np.int32)

        residual_scale = float(np.sqrt(np.mean(np.sum((Y - np.mean(Y, axis=0)) ** 2, axis=1))))
        step = branch_scale * residual_scale / max(1, branch_length)

        for root in root_ids:
            root = int(root)
            root_pt = spine[root]

            if 0 < root < n_spine_nodes - 1:
                tangent = spine[root + 1] - spine[root - 1]
            elif root == 0:
                tangent = spine[1] - spine[0]
            else:
                tangent = spine[-1] - spine[-2]
            tnorm = np.linalg.norm(tangent) + 1e-12
            tangent = tangent / tnorm

            # Use local residuals orthogonal to tangent to estimate branch direction.
            d2 = np.sum((Y - root_pt[None, :]) ** 2, axis=1)
            near_idx = np.argpartition(d2, kth=min(40, len(d2) - 1))[: min(40, len(d2))]
            local = Y[near_idx] - root_pt[None, :]
            proj = (local @ tangent)[:, None] * tangent[None, :]
            perp = local - proj
            direction = np.mean(perp, axis=0)
            dnorm = np.linalg.norm(direction)

            if dnorm < 1e-8:
                # Deterministic fallback perpendicular in 2D.
                if Y.shape[1] >= 2:
                    direction = np.array([-tangent[1], tangent[0]], dtype=np.float64)
                else:
                    direction = np.array([1.0], dtype=np.float64)
                dnorm = np.linalg.norm(direction)
            direction = direction / (dnorm + 1e-12)

            branch_nodes = np.zeros((branch_length, Y.shape[1]), dtype=np.float64)
            for k in range(branch_length):
                branch_nodes[k] = root_pt + (k + 1) * step * direction

            base = sum(arr.shape[0] for arr in nodes)
            nodes.append(branch_nodes)
            edge_list.append((root, base))
            for k in range(branch_length - 1):
                edge_list.append((base + k, base + k + 1))

    X0 = np.vstack(nodes).astype(np.float32)
    edges = _canonicalize_edges(np.asarray(edge_list, dtype=np.int32), n=X0.shape[0])
    spine_edges = _canonicalize_edges(np.asarray(spine_edge_list, dtype=np.int32), n=X0.shape[0])
    return X0, edges, spine_edges


def write_history_csv(path: Path, history: list[dict[str, float]]) -> None:
    keys = sorted({k for row in history for k in row.keys()})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(history)


def plot_result(
    path: Path,
    Y: np.ndarray,
    X: np.ndarray,
    edges: np.ndarray,
    spine_edges: np.ndarray,
    train_lines: list[dict[str, object]],
) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
    ax.scatter(Y[:, 0], Y[:, 1], s=10, alpha=0.24, color="#8a8a8a", label="Point cloud")

    if edges.size > 0:
        spine_edge_set = {(int(a), int(b)) for a, b in np.asarray(spine_edges, dtype=np.int32)}
        for a, b in edges:
            xa, xb = X[a], X[b]
            e = (int(min(a, b)), int(max(a, b)))
            is_spine = e in spine_edge_set
            color = "#0d6efd" if is_spine else "#6c757d"
            lw = 1.8 if is_spine else 1.2
            ax.plot([xa[0], xb[0]], [xa[1], xb[1]], linewidth=lw, alpha=0.9, color=color)

    line_colors = ["#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#17becf", "#8c564b"]
    for i, line in enumerate(train_lines):
        path_nodes = line.get("path", [])
        if len(path_nodes) < 2:
            continue
        route = X[np.asarray(path_nodes, dtype=np.int32)]
        ax.plot(
            route[:, 0],
            route[:, 1],
            color=line_colors[i % len(line_colors)],
            linewidth=2.5,
            alpha=0.95,
            label=f"Train line {i + 1}",
        )

    ax.scatter(X[:, 0], X[:, 1], s=22, color="#111111", label="Principal-curve nodes")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Strict Principal Curve (Spine + Optional Branches)")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Strict principal-curve optimizer (NumPy)")
    p.add_argument("--out", type=Path, default=Path("out_principal_curve"), help="Output directory")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-spine-nodes", type=int, default=30)
    p.add_argument("--n-branches", type=int, default=2)
    p.add_argument("--branch-length", type=int, default=5)
    p.add_argument("--branch-scale", type=float, default=0.45)

    # Same penalties/terms as the existing NumPy optimizer.
    p.add_argument("--iters", type=int, default=180)
    p.add_argument("--lr", type=float, default=1.2e-2)
    p.add_argument("--p", type=float, default=2.0)
    p.add_argument("--lambda1", type=float, default=0.08)
    p.add_argument("--lambda-curv", type=float, default=0.28)
    p.add_argument("--lambda-train", type=float, default=0.18)
    p.add_argument("--n-train-lines", type=int, default=3)
    p.add_argument("--line-overlap-penalty", type=float, default=1.25)
    p.add_argument("--line-candidate-pool", type=int, default=28)
    p.add_argument("--min-line-hops", type=int, default=2)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    Y = make_city_like_cloud(seed=args.seed)
    X0, edges, spine_edges = init_principal_curve_nodes(
        Y=Y,
        n_spine_nodes=args.n_spine_nodes,
        n_branches=args.n_branches,
        branch_length=args.branch_length,
        branch_scale=args.branch_scale,
    )
    params = NumpyParams(
        p=args.p,
        lambda1=args.lambda1,
        lambda_curv=args.lambda_curv,
        lambda_train=args.lambda_train,
        n_train_lines=args.n_train_lines,
        line_overlap_penalty=args.line_overlap_penalty,
        line_candidate_pool=args.line_candidate_pool,
        min_line_hops=args.min_line_hops,
        lr=args.lr,
        iters=args.iters,
    )

    X_final, history, train_info = optimize_numpy(
        X0=X0,
        edges=edges,
        Y=Y,
        params=params,
    )

    write_history_csv(out / "history.csv", history)
    np.save(out / "X0.npy", X0)
    np.save(out / "X_final.npy", X_final)
    np.save(out / "edges.npy", edges.astype(np.int32))
    plot_result(out / "plot.png", Y, X_final, edges, spine_edges, train_lines=list(train_info.get("lines", [])))

    print(f"Wrote: {out / 'history.csv'}")
    print(f"Wrote: {out / 'X0.npy'}")
    print(f"Wrote: {out / 'X_final.npy'}")
    print(f"Wrote: {out / 'edges.npy'}")
    print(f"Wrote: {out / 'plot.png'}")
    print(f"Nodes: {X_final.shape[0]}, Edges: {edges.shape[0]}, Spine edges: {spine_edges.shape[0]}")
    print(f"Train lines selected: {len(train_info.get('lines', []))}")
    print(f"Mean line distance: {train_info.get('mean_line_distance')}")
    print(f"Overlap ratio: {train_info.get('overlap_ratio')}")
    print(f"Line objective: {train_info.get('line_objective')}")


if __name__ == "__main__":
    main()
