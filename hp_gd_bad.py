import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
from PIL import Image


RNG = np.random.default_rng(42)

# Hyperparameters
N_HUBS      = 7        # number of hubs (green nodes)
N_DATA      = 70       # number of data points
LAM         = 0.5      # HP regulariser weight  λ
ETA         = 0.025    # gradient descent step size  η
CW          = 1.0      # edge cost weight
REPULSION   = 30.0     # hub repulsion strength
GRAD_CLIP   = 50.0     # gradient clipping threshold
EDGE_EVERY  = 10       # run edge pruning every this many GD steps
N_ITER      = 120      # total gradient descent iterations

GIF_FRAMES  = 40       # number of frames in the GIF
GIF_FPS     = 8        # frames per second

BG_COLOR    = "white"
FG_COLOR    = "#111"


def make_scene():
    centers = RNG.uniform(0.1, 0.9, (N_HUBS, 2))
    assign  = RNG.integers(0, N_HUBS, N_DATA)
    data    = centers[assign] + RNG.normal(0, 0.08, (N_DATA, 2))
    data    = np.clip(data, 0.02, 0.98)
    hubs    = centers + RNG.normal(0, 0.1, (N_HUBS, 2))
    hubs    = np.clip(hubs, 0.02, 0.98)
    return data, hubs

# graph utilities
def prim_mst(hubs):
    n = len(hubs)
    in_tree, edges = {0}, []
    while len(in_tree) < n:
        best, bd = None, np.inf
        for i in in_tree:
            for j in range(n):
                if j not in in_tree:
                    d = np.linalg.norm(hubs[i] - hubs[j])
                    if d < bd:
                        bd, best = d, (i, j)
        in_tree.add(best[1])
        edges.append(best)
    return edges

def build_candidate_edges(hubs, thresh=0.7):
    n, edges = len(hubs), []
    for i in range(n):
        for j in range(i+1, n):
            if np.linalg.norm(hubs[i]-hubs[j]) < thresh:
                edges.append((i, j))
    mst = prim_mst(hubs)
    for e in mst:
        if e not in edges and (e[1], e[0]) not in edges:
            edges.append(e)
    # add a few random extra edges for cycles
    for _ in range(n // 2):
        a, b = RNG.integers(0, n, 2)
        if a != b and (a,b) not in edges and (b,a) not in edges:
            edges.append((int(a), int(b)))
    return list(edges)

def is_connected_without(active, skip_idx, n):
    adj = [[] for _ in range(n)]
    for k, (i, j) in enumerate(active):
        if k != skip_idx:
            adj[i].append(j); adj[j].append(i)
    vis, q = set([0]), deque([0])
    while q:
        cur = q.popleft()
        for nb in adj[cur]:
            if nb not in vis:
                vis.add(nb); q.append(nb)
    return len(vis) == n

def get_assignments(data, hubs):
    dists = np.linalg.norm(data[:, None] - hubs[None], axis=2)
    return np.argmin(dists, axis=1)

def bfs_path(active, n, src, dst):
    adj = [[] for _ in range(n)]
    for i, j in active:
        adj[i].append(j); adj[j].append(i)
    prev, vis, q = [-1]*n, [False]*n, deque([src])
    vis[src] = True
    while q:
        cur = q.popleft()
        if cur == dst: break
        for nb in adj[cur]:
            if not vis[nb]:
                vis[nb] = True; prev[nb] = cur; q.append(nb)
    path, cur = [], dst
    while cur != -1:
        path.insert(0, cur); cur = prev[cur]
    return path if path[0] == src else []

# Gradient
def compute_gradients(hubs, data, active, assignments):
    n    = len(hubs)
    grads = np.zeros_like(hubs)
    wi   = np.array([np.sum(assignments == i) for i in range(n)], dtype=float)

    # Fidelity  –∂‖y−xk‖/∂xk = −(y−xk)/‖y−xk‖
    for k in range(n):
        pts = data[assignments == k]
        if len(pts) == 0: continue
        diff = hubs[k] - pts          # (m, 2)
        d    = np.linalg.norm(diff, axis=1, keepdims=True).clip(1e-9)
        grads[k] += (diff / d).sum(axis=0)

    # Repulsion
    ms = 0.05
    for a in range(n):
        for b in range(a+1, n):
            diff = hubs[a] - hubs[b]
            d    = np.linalg.norm(diff).clip(1e-9)
            if d < 3*ms:
                f = REPULSION * ms**2 / d**3
                grads[a] -= f * (hubs[b] - hubs[a])
                grads[b] += f * (hubs[b] - hubs[a])

    # Complexity  1/4 w(i)w(j) path-length gradient
    N2 = max(1, len(data)**2)
    for i in range(n):
        for j in range(i+1, n):
            path = bfs_path(active, n, i, j)
            if len(path) < 2: continue
            w = 0.25 * wi[i] * wi[j] / N2
            for idx, k in enumerate(path):
                for l in ([path[idx-1]] if idx>0 else []) + ([path[idx+1]] if idx<len(path)-1 else []):
                    diff = hubs[k] - hubs[l]
                    d    = np.linalg.norm(diff).clip(1e-9)
                    grads[k] += w * 2 * diff / d

    # HP regulariser  2λ/n · (xk − xother) per active edge
    for i, j in active:
        diff = hubs[i] - hubs[j]
        grads[i] += (2 * LAM / n) * diff
        grads[j] -= (2 * LAM / n) * diff

    # Cost  2·cw/|E| · (xk−xj)/‖xk−xj‖
    nE = max(1, len(active))
    for i, j in active:
        diff = hubs[i] - hubs[j]
        d    = np.linalg.norm(diff).clip(1e-9)
        grads[i] += (2 * CW / nE) * diff / d
        grads[j] -= (2 * CW / nE) * diff / d

    return grads

def compute_loss(hubs, data, active, assignments):
    n  = len(hubs)
    wi = np.array([np.sum(assignments == i) for i in range(n)], dtype=float)
    F  = 0.0
    for k in range(n):
        pts = data[assignments == k]
        if len(pts): F += np.linalg.norm(hubs[k] - pts, axis=1).sum()
    for i in range(n):
        for j in range(i+1, n):
            path = bfs_path(active, n, i, j)
            if len(path) < 2: continue
            L = sum(np.linalg.norm(hubs[path[t]] - hubs[path[t+1]]) for t in range(len(path)-1))
            F += 0.25 * wi[i] * wi[j] * L
    for i, j in active:
        diff = hubs[i] - hubs[j]
        F += LAM * np.dot(diff, diff)
        F += CW  * np.linalg.norm(diff)
    return F

def edge_score(hubs, active, assignments, i, j):
    n  = len(hubs)
    wi = np.sum(assignments == i); wj = np.sum(assignments == j)
    nE = max(1, len(active)); N2 = max(1, len(assignments)**2)
    dist = np.linalg.norm(hubs[i] - hubs[j])
    return (LAM / n) * dist**2 + (CW / nE) * dist + 0.25 * wi * wj * dist / N2

def prune_edges(hubs, active, assignments):
    n       = len(hubs)
    max_rm  = max(1, len(active) // 4)
    active  = list(active)
    for _ in range(max_rm):
        if len(active) <= n - 1: break
        scored = sorted(enumerate(active), key=lambda x: -edge_score(hubs, active, assignments, *x[1]))
        removed = False
        for idx, (i, j) in scored:
            if is_connected_without(active, idx, n):
                active.pop(idx); removed = True; break
        if not removed: break
    # guarantee MST connectivity
    for e in prim_mst(hubs):
        if e not in active and (e[1], e[0]) not in active:
            active.append(e)
    return active

# Optimization
def run_optimization():
    data, hubs = make_scene()
    candidate  = build_candidate_edges(hubs)
    active     = prim_mst(hubs)[:]   # start with MST
    asn        = get_assignments(data, hubs)
    losses     = []
    snapshots  = []   # (hubs_copy, active_copy, asn_copy, loss)

    frame_iters = set(np.linspace(0, N_ITER-1, GIF_FRAMES, dtype=int))

    for it in range(N_ITER):
        grads = compute_gradients(hubs, data, active, asn)
        gnorm = np.linalg.norm(grads)
        scale = min(1.0, GRAD_CLIP / gnorm) if gnorm > GRAD_CLIP else 1.0
        hubs  = np.clip(hubs - ETA * grads * scale, 0.02, 0.98)
        asn   = get_assignments(data, hubs)
        if (it + 1) % EDGE_EVERY == 0:
            active = prune_edges(hubs, active, asn)
        loss = compute_loss(hubs, data, active, asn)
        losses.append(loss)
        if it in frame_iters:
            snapshots.append((hubs.copy(), list(active), asn.copy(), loss, it))

    return data, snapshots, losses


COLORS = plt.cm.Set2.colors   # one colour per hub cluster

def draw_frame(ax, data, hubs, active, asn, loss, it, show_loss_note=True):
    ax.clear()
    ax.set_facecolor("white")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")

    n = len(hubs)

    # candidate edges (dim)
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    for i, j in all_pairs:
        if (i,j) not in active and (j,i) not in active:
            ax.plot([hubs[i,0], hubs[j,0]], [hubs[i,1], hubs[j,1]],
                    color="#1f2a3e", lw=0.8, ls="--", zorder=1)

    # active edges (bright)
    for i, j in active:
        ax.plot([hubs[i,0], hubs[j,0]], [hubs[i,1], hubs[j,1]],
                color="#2563eb", lw=2, alpha=0.75, zorder=2)

    # data points coloured by cluster
    for k in range(n):
        pts = data[asn == k]
        if len(pts):
            ax.scatter(pts[:,0], pts[:,1], s=12, color=COLORS[k % len(COLORS)],
                       alpha=0.45, zorder=3, linewidths=0)

    # assignment lines
    for idx, pt in enumerate(data):
        h = hubs[asn[idx]]
        ax.plot([pt[0], h[0]], [pt[1], h[1]],
                color="#f87171", lw=0.4, alpha=0.15, zorder=2)

    # hub nodes
    wi = np.array([np.sum(asn == i) for i in range(n)])
    sizes = 80 + wi * 18
    ax.scatter(hubs[:,0], hubs[:,1], s=sizes, color="#6ee7b7",
               edgecolors="#0c0f17", linewidths=1.5, zorder=5)
    for i, (x, y) in enumerate(hubs):
        ax.text(x, y - 0.045, f"x{i}", color="#6ee7b7",
                fontsize=6, ha="center", va="top",
                fontfamily="monospace", zorder=6)

    # title / metadata
    ax.set_title(f"Modified HP Filter  ·  iter {it+1}  ·  F = {loss:.1f}  ·  |E| = {len(active)}",
                 color="#111", fontsize=8, fontfamily="monospace",
                 pad=6, loc="left")

def add_loss_panel(ax2, losses, current_idx):
    ax2.clear()
    ax2.set_facecolor("#0c0f17")
    visible = losses[:current_idx]
    if visible:
        ax2.plot(range(1, len(visible)+1), visible, color="#6ee7b7", lw=1.3)
        ax2.fill_between(range(1, len(visible)+1), visible,
                         alpha=0.08, color="#6ee7b7")
        ymin, ymax = min(visible), max(visible)
        pad = max((ymax-ymin)*0.1, 1.0)
        ax2.set_ylim(ymin-pad, ymax+pad)
    ax2.set_xlim(1, N_ITER)
    ax2.tick_params(colors="#475569", labelsize=6)
    for sp in ax2.spines.values(): sp.set_edgecolor("#1f2a3e")
    ax2.set_facecolor("#0c0f17")
    ax2.set_xlabel("iter", color="#475569", fontsize=7)
    ax2.set_ylabel("F",    color="#475569", fontsize=7)
    ax2.set_title("Loss F(X,E)", color="#6ee7b7", fontsize=7, pad=4)


def save_static(data, snapshots, losses):
    hubs, active, asn, loss, it = snapshots[-1]
    fig = plt.figure(figsize=(11, 7), facecolor=BG_COLOR)
    ax1 = fig.add_axes([0.02, 0.02, 0.64, 0.88])   # main plot  (left 65%)
    ax2 = fig.add_axes([0.70, 0.10, 0.28, 0.30])   # loss panel (right, lower third)
    draw_frame(ax1, data, hubs, active, asn, loss, it)
    add_loss_panel(ax2, losses, len(losses))
    fig.savefig("hp_final_bad.png", dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print("Saved hp_final_bad.png")


def save_gif(data, snapshots, losses):
    fig = plt.figure(figsize=(10, 6), facecolor=BG_COLOR)
    ax1 = fig.add_axes([0.02, 0.02, 0.64, 0.88])   # main plot  (left 65%)
    ax2 = fig.add_axes([0.70, 0.10, 0.28, 0.30])   # loss panel (right, lower third)
    frames = []

    for hubs, active, asn, loss, it in snapshots:
        draw_frame(ax1, data, hubs, active, asn, loss, it)
        add_loss_panel(ax2, losses, it)
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h,w,3)
        frames.append(Image.fromarray(buf))

    frames[0].save(
        "hp_anim_bad.gif",
        save_all=True, append_images=frames[1:],
        duration=int(1000/GIF_FPS), loop=0,
    )
    plt.close(fig)
    print("Saved hp_anim_bad.gif")


if __name__ == "__main__":
    print("Running optimisation...")
    data, snapshots, losses = run_optimization()
    print(f"Done. Final loss: {losses[-1]:.2f}  |  Frames: {len(snapshots)}")
    save_static(data, snapshots, losses)
    save_gif(data, snapshots, losses)
    print("All done!")