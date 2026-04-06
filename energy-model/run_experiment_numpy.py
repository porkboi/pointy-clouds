from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from core.build_edges import build_edges
from core.optimizer_numpy import NumpyParams, optimize_numpy


def make_city_like_cloud(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)

    # Dense downtown core.
    downtown = 0.18 * rng.normal(size=(260, 2))

    # Neighborhood clusters around the core with different spreads.
    centers = np.array(
        [
            [-1.5, 0.9],
            [-1.2, -0.9],
            [-0.3, 1.4],
            [0.8, 1.2],
            [1.4, 0.3],
            [1.1, -0.8],
            [-0.2, -1.5],
        ],
        dtype=np.float64,
    )
    sizes = [110, 90, 85, 95, 100, 88, 80]
    neighborhoods = []
    for c, n in zip(centers, sizes):
        ang = rng.uniform(0.0, np.pi)
        rot = np.array(
            [[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]],
            dtype=np.float64,
        )
        scales = np.diag([rng.uniform(0.08, 0.2), rng.uniform(0.06, 0.16)])
        cov = rot @ scales @ scales @ rot.T
        neighborhoods.append(rng.multivariate_normal(mean=c, cov=cov, size=n))

    # Two high-density transit corridors crossing the city.
    t1 = rng.uniform(-2.0, 2.0, size=(170, 1))
    corridor_1 = np.hstack([0.9 * t1, 0.55 * t1 + 0.15 * rng.normal(size=(170, 1))])
    t2 = rng.uniform(-1.7, 1.7, size=(130, 1))
    corridor_2 = np.hstack([0.4 * t2 + 0.2 * rng.normal(size=(130, 1)), -0.95 * t2])

    Y = np.vstack([downtown, *neighborhoods, corridor_1, corridor_2])

    # Slight axis scaling to resemble non-uniform urban footprint.
    Y = Y * np.array([1.1, 0.95], dtype=np.float64)
    return Y.astype(np.float32)


def init_particles(seed: int, Y: np.ndarray, n: int = 22) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = rng.choice(Y.shape[0], size=n, replace=False)
    return Y[idx].astype(np.float32)


def write_history_csv(path: Path, history: list[dict[str, float]]) -> None:
    keys = sorted({k for row in history for k in row.keys()})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(history)


def plot_result(path: Path, Y: np.ndarray, X: np.ndarray, edges: np.ndarray, train_lines: list[dict[str, object]]) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)

    ax.scatter(Y[:, 0], Y[:, 1], s=10, alpha=0.28, color="#888888", label="Point cloud")

    if edges.size > 0:
        for a, b in edges:
            xa, xb = X[a], X[b]
            ax.plot([xa[0], xb[0]], [xa[1], xb[1]], linewidth=1.1, alpha=0.75, color="#1f77b4")

    line_colors = ["#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#17becf", "#8c564b"]
    for i, line in enumerate(train_lines):
        path_nodes = line.get("path", [])
        if len(path_nodes) < 2:
            continue
        route = X[np.asarray(path_nodes, dtype=np.int32)]
        color = line_colors[i % len(line_colors)]
        ax.plot(
            route[:, 0],
            route[:, 1],
            color=color,
            linewidth=2.4,
            alpha=0.95,
            label=f"Train line {i + 1}",
        )

    ax.scatter(X[:, 0], X[:, 1], s=20, color="#111111", label="Fitted nodes")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("NumPy Energy Fit with Curvature + Multi-Line Dijkstra Cost")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run NumPy-only energy optimization on a small point cloud")
    p.add_argument("--out", type=Path, default=Path("out_numpy"), help="Output directory")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-particles", type=int, default=22)
    p.add_argument("--iters", type=int, default=140)
    p.add_argument("--lr", type=float, default=1.2e-2)
    p.add_argument("--p", type=float, default=2.0)
    p.add_argument("--lambda1", type=float, default=0.08)
    p.add_argument("--lambda-curv", type=float, default=0.28)
    p.add_argument("--lambda-train", type=float, default=0.18)
    p.add_argument("--n-train-lines", type=int, default=3)
    p.add_argument("--line-overlap-penalty", type=float, default=0.25)
    p.add_argument("--edge-method", choices=["principal_curve", "knn", "mst"], default="principal_curve")
    p.add_argument("--n-principal-curves", type=int, default=1)
    p.add_argument("--k", type=int, default=2, help="k for kNN edges")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    Y = make_city_like_cloud(seed=args.seed)
    X0 = init_particles(seed=args.seed + 1, Y=Y, n=args.n_particles)
    edges = build_edges(X0, method=args.edge_method, k=args.k, n_principal_curves=args.n_principal_curves)

    params = NumpyParams(
        p=args.p,
        lambda1=args.lambda1,
        lambda_curv=args.lambda_curv,
        lambda_train=args.lambda_train,
        n_train_lines=args.n_train_lines,
        line_overlap_penalty=args.line_overlap_penalty,
        enforce_principal_curve_topology=(args.edge_method == "principal_curve"),
        n_principal_curves=args.n_principal_curves,
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
    np.save(out / "X_final.npy", X_final)
    edges_final = build_edges(
        X_final,
        method=args.edge_method,
        k=args.k,
        n_principal_curves=args.n_principal_curves,
    )
    np.save(out / "edges.npy", edges_final.astype(np.int32))
    plot_result(out / "plot.png", Y, X_final, edges_final, train_lines=list(train_info.get("lines", [])))

    print(f"Wrote: {out / 'history.csv'}")
    print(f"Wrote: {out / 'X_final.npy'}")
    print(f"Wrote: {out / 'edges.npy'}")
    print(f"Wrote: {out / 'plot.png'}")
    print(f"Train lines selected: {len(train_info.get('lines', []))}")
    print(f"Mean line distance: {train_info.get('mean_line_distance')}")
    for i, line in enumerate(train_info.get("lines", [])):
        print(
            f"Line {i + 1}: {line.get('start')} -> {line.get('end')}, "
            f"distance={line.get('distance')}, path={line.get('path')}"
        )


if __name__ == "__main__":
    main()
