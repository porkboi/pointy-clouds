from __future__ import annotations

import argparse
import csv
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from core.build_edges import build_edges
from core.optimizer import Params, optimize


def make_ring_branches(
    key: jax.Array,
    m_ring: int = 1000,
    m_branch: int = 300,
    radius: float = 1.0,
    noise: float = 0.03,
) -> np.ndarray:
    """Synthetic 2D benchmark: noisy ring + three radial branches."""
    key_theta, key_noise, key_br = jax.random.split(key, 3)

    theta = jax.random.uniform(key_theta, (m_ring,), minval=0.0, maxval=2.0 * jnp.pi)
    ring = jnp.stack([radius * jnp.cos(theta), radius * jnp.sin(theta)], axis=1)

    branch_angles = jnp.array([0.2, 2.3, 4.4], dtype=jnp.float32)
    branch_ids = jax.random.randint(key_br, (m_branch,), minval=0, maxval=3)
    t = jax.random.uniform(key_theta, (m_branch,), minval=0.0, maxval=1.0)
    rr = radius + 0.9 * t
    ang = branch_angles[branch_ids]
    branches = jnp.stack([rr * jnp.cos(ang), rr * jnp.sin(ang)], axis=1)

    Y = jnp.concatenate([ring, branches], axis=0)
    Y = Y + noise * jax.random.normal(key_noise, Y.shape)
    return np.asarray(Y, dtype=np.float32)


def init_particles(key: jax.Array, Y: np.ndarray, n: int = 120) -> np.ndarray:
    m = Y.shape[0]
    idx = np.asarray(jax.random.choice(key, m, shape=(n,), replace=False), dtype=np.int32)
    return Y[idx].astype(np.float32)


def write_history_csv(path: Path, history: list[dict[str, float]]) -> None:
    keys = sorted({k for row in history for k in row.keys()})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(history)


def plot_result(path: Path, Y: np.ndarray, X: np.ndarray, edges: np.ndarray) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 7), dpi=140)
    ax.scatter(Y[:, 0], Y[:, 1], s=4, alpha=0.25, label="Y points")

    if edges.size > 0:
        for a, b in edges:
            xa, xb = X[a], X[b]
            ax.plot([xa[0], xb[0]], [xa[1], xb[1]], linewidth=1.0, alpha=0.8, color="#1f77b4")

    ax.scatter(X[:, 0], X[:, 1], s=14, color="#d62728", label="X particles")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Energy Model Result")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run JAX baseline energy optimization experiment")
    p.add_argument("--out", type=Path, default=Path("out"), help="Output directory")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--m-ring", type=int, default=1000)
    p.add_argument("--m-branch", type=int, default=300)
    p.add_argument("--n-particles", type=int, default=120)
    p.add_argument("--outer-iters", type=int, default=500)
    p.add_argument("--inner-steps", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--p", type=float, default=2.0)
    p.add_argument("--lambda1", type=float, default=0.1)
    p.add_argument("--lambda-curv", type=float, default=0.0)
    p.add_argument("--lambda-T", type=float, default=0.1)
    p.add_argument("--w-access", type=float, default=1.0)
    p.add_argument("--w-train", type=float, default=0.2)
    p.add_argument("--pair-count", type=int, default=4096)
    p.add_argument("--pair-mode", choices=["local", "global", "mixed"], default="mixed")
    p.add_argument("--local-k", type=int, default=32)
    p.add_argument("--disconnected-penalty", type=float, default=1e3)
    p.add_argument("--edge-method", choices=["principal_curve", "knn", "mst"], default="principal_curve")
    p.add_argument("--n-principal-curves", type=int, default=1)
    p.add_argument("--k", type=int, default=3, help="k for kNN edges")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    key = jax.random.PRNGKey(args.seed)
    key_y, key_x = jax.random.split(key)

    Y = make_ring_branches(key_y, m_ring=args.m_ring, m_branch=args.m_branch)
    X0 = init_particles(key_x, Y, n=args.n_particles)
    edges = build_edges(X0, method=args.edge_method, k=args.k, n_principal_curves=args.n_principal_curves)

    params = Params(
        p=args.p,
        lambda1=args.lambda1,
        lambda_curv=args.lambda_curv,
        lambda_T=args.lambda_T,
        w_access=args.w_access,
        w_train=args.w_train,
        pair_count=args.pair_count,
        pair_mode=args.pair_mode,
        local_k=args.local_k,
        outer_iters=args.outer_iters,
        inner_steps=args.inner_steps,
        disconnected_penalty=args.disconnected_penalty,
        enforce_principal_curve_topology=(args.edge_method == "principal_curve"),
        n_principal_curves=args.n_principal_curves,
        lambda2=0.0,
        eps=1e-8,
        lr=args.lr,
        iters=args.outer_iters,
    )

    X_final, history = optimize(
        key=key,
        X0=jnp.asarray(X0),
        edges=jnp.asarray(edges, dtype=jnp.int32),
        Y=jnp.asarray(Y),
        params=params,
        hook_energy=None,
    )

    X_np = np.asarray(X_final, dtype=np.float32)

    write_history_csv(out / "history.csv", history)
    np.save(out / "X_final.npy", X_np)
    edges_final = build_edges(X_np, method=args.edge_method, k=args.k, n_principal_curves=args.n_principal_curves)
    np.save(out / "edges.npy", edges_final.astype(np.int32))
    plot_result(out / "plot.png", Y, X_np, edges_final)

    print(f"Wrote: {out / 'history.csv'}")
    print(f"Wrote: {out / 'X_final.npy'}")
    print(f"Wrote: {out / 'edges.npy'}")
    print(f"Wrote: {out / 'plot.png'}")


if __name__ == "__main__":
    main()
