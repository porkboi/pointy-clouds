"""
OPTIMAL TRANSIT NETWORK — FICTIONAL POLYCENTRIC REGION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ℓ₁ Trend Filtering → distinct colored transit lines.

Population/activity is synthetic and multi-centre:
several CBDs, sub-centres, and corridor-oriented growth.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator
from matplotlib.patches import Polygon as MplPolygon
from collections import defaultdict
import heapq
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ═══════════════════════════════════════════════════════════
# 1.  FICTIONAL REGION GEOGRAPHY (uses same coordinate box)
#     Lon-like axis: 139.55 → 139.95
#     Lat-like axis:  35.55 →  35.82
#     Mapped to display grid 0..8 × 0..5.4
# ═══════════════════════════════════════════════════════════
LON0, LON1 = 139.55, 139.95
LAT0, LAT1 =  35.55,  35.82

GX, GY = 240, 162

xc = np.linspace(0, 8.0, GX)
yc = np.linspace(0, 5.4, GY)
XX, YY = np.meshgrid(xc, yc)

def lx(lon): return (lon - LON0) / (LON1 - LON0) * 8.0
def ly(lat): return (lat - LAT0) / (LAT1 - LAT0) * 5.4

def blob(cx, cy, sx, sy, peak):
    return peak * np.exp(-((XX-cx)**2/(2*sx**2) + (YY-cy)**2/(2*sy**2)))

def corridor(x0, y0, x1, y1, width=0.12, peak=10):
    dx, dy = x1-x0, y1-y0
    L = np.hypot(dx, dy) + 1e-9
    t = np.clip(((XX-x0)*dx+(YY-y0)*dy)/L**2, 0, 1)
    d = np.hypot(XX-(x0+t*dx), YY-(y0+t*dy))
    return peak * np.exp(-d**2/(2*width**2))

# ── Synthetic multi-centre density nodes ────────────────────
# (name, lon, lat, sx_deg, sy_deg, peak)
PLACES = {
    # Primary centres
    'Aster Core':        (139.752, 35.687, 0.020, 0.016, 108),
    'Nova Junction':     (139.708, 35.692, 0.018, 0.014,  96),
    'Harbor Delta':      (139.804, 35.654, 0.018, 0.014,  150),
    'Northgate':         (139.744, 35.752, 0.016, 0.013,  200),
    'Northeast Arc':     (139.822, 35.754, 0.013, 0.010,  180),

    # Secondary centres
    'Westpoint':         (139.622, 35.704, 0.016, 0.012,  82),
    'Eastport':          (139.846, 35.706, 0.017, 0.013,  84),
    'Southbank':         (139.716, 35.612, 0.016, 0.012,  150),
    'Rivercross':        (139.784, 35.716, 0.014, 0.011,  78),
    'University Belt':   (139.734, 35.718, 0.014, 0.010,  74),
    'Tech Spine':        (139.776, 35.690, 0.014, 0.010,  76),

    # Tertiary clusters
    'Hillside':          (139.664, 35.672, 0.012, 0.010,  66),
    'Old Town':          (139.792, 35.679, 0.012, 0.010,  64),
    'Garden District':   (139.686, 35.734, 0.012, 0.010,  62),
    'Lakeside':          (139.592, 35.686, 0.013, 0.010,  60),
    'Freight Yard':      (139.828, 35.642, 0.013, 0.010,  61),
    'Aerotown':          (139.690, 35.575, 0.013, 0.010,  58),
    'Outer Ring West':   (139.574, 35.706, 0.013, 0.010,  57),
    'Outer Ring East':   (139.886, 35.690, 0.013, 0.010,  57),
    'Foothill North':    (139.706, 35.784, 0.012, 0.010,  55),
}

density = np.zeros((GY, GX))
for name, (lon, lat, sx_d, sy_d, peak) in PLACES.items():
    cx = lx(lon); cy = ly(lat)
    sx = sx_d / (LON1-LON0) * 8.0
    sy = sy_d / (LAT1-LAT0) * 5.4
    density += blob(cx, cy, sx, sy, peak)

# ── Corridor density boosts between centres ────────────────
CORR = [
    # Core orbital arcs
    (lx(139.708), ly(35.692), lx(139.744), ly(35.752), 0.09, 20),
    (lx(139.744), ly(35.752), lx(139.804), ly(35.654), 0.09, 20),
    (lx(139.804), ly(35.654), lx(139.708), ly(35.692), 0.09, 19),
    # West-East backbone
    (lx(139.574), ly(35.706), lx(139.886), ly(35.690), 0.08, 17),
    # North-South spine
    (lx(139.706), ly(35.784), lx(139.690), ly(35.575), 0.08, 16),
    # Harbour connector and logistics branch
    (lx(139.716), ly(35.612), lx(139.846), ly(35.706), 0.07, 14),
    (lx(139.804), ly(35.654), lx(139.828), ly(35.642), 0.06, 13),
    # Northeast growth corridor
    (lx(139.744), ly(35.752), lx(139.822), ly(35.754), 0.07, 13),
]
for c in CORR:
    density += corridor(*c)

density += np.abs(np.random.randn(GY, GX)) * 1.7
density  = gaussian_filter(density, sigma=2.0)
density  = np.clip(density, 0, None)

interp_density = RegularGridInterpolator(
    (yc, xc), density, method='linear',
    bounds_error=False, fill_value=0
)

# ═══════════════════════════════════════════════════════════
# 2.  ℓ₁ TREND FILTER
# ═══════════════════════════════════════════════════════════
LAM = 18
PATH_MU = 7.5
OUTER_MAX_ITERS = 6
INNER_MAX_ITERS = 70
SOLVER_TOL = 1e-4

def make_D2(n):
    D = np.zeros((n-2, n))
    for i in range(n-2):
        D[i,i]=1; D[i,i+1]=-2; D[i,i+2]=1
    return D

def make_D1_sub(n, a_idx, b_idx):
    m = max(0, b_idx - a_idx)
    D = np.zeros((m, n))
    for i in range(m):
        t = a_idx + i
        D[i, t] = -1.0
        D[i, t+1] = 1.0
    return D

def soft_threshold(v, tau):
    return np.sign(v) * np.maximum(np.abs(v) - tau, 0.0)

def l1_admm(y, lam=LAM, max_iter=600, tol=1e-5):
    n = len(y)
    if n < 3: return y.copy()
    D = make_D2(n)
    rho = lam
    M = np.eye(n) + rho*(D.T@D)
    x = y.copy(); z = D@x; u = np.zeros(n-2)
    for _ in range(max_iter):
        x   = np.linalg.solve(M, y + rho*D.T@(z-u))
        Dxu = D@x + u
        zn  = np.sign(Dxu)*np.maximum(np.abs(Dxu)-lam/rho, 0)
        u  += D@x - zn
        if np.linalg.norm(zn-z) < tol: break
        z = zn
    return x

def _anchor_grad_hess(xi, vi, c, mu_path, eps=1e-8):
    s = xi - vi
    den = np.sqrt(c*c + s*s + eps)
    grad = mu_path * (s / den)
    hess = mu_path * (c*c + eps) / (den**3)
    return grad, hess

def _find_nn_anchor_indices(x, t_a, v_a, t_b, v_b):
    idx = np.arange(len(x), dtype=float)
    da = (idx - t_a)**2 + (x - v_a)**2
    db = (idx - t_b)**2 + (x - v_b)**2
    a_idx = int(np.argmin(da))
    b_idx = int(np.argmin(db))
    return a_idx, b_idx

def path_cost_inner_admm(
    y, x0, lam, mu_path, t_a, v_a, t_b, v_b, a_idx, b_idx,
    max_iter=INNER_MAX_ITERS, tol=SOLVER_TOL
):
    n = len(y)
    if n < 3:
        return y.copy()

    D2 = make_D2(n)
    D1s = make_D1_sub(n, a_idx, b_idx)
    rho2 = max(1.0, lam)
    rho1 = max(1.0, 0.5 * mu_path)

    x = x0.copy()
    z = D2 @ x
    u = np.zeros_like(z)

    if D1s.shape[0] > 0:
        w = D1s @ x
        r = np.zeros_like(w)
    else:
        w = np.zeros(0)
        r = np.zeros(0)

    M = np.eye(n) + rho2 * (D2.T @ D2)
    if D1s.shape[0] > 0:
        M += rho1 * (D1s.T @ D1s)

    c_a = float(a_idx - t_a)
    c_b = float(b_idx - t_b)

    for _ in range(max_iter):
        x_prev = x.copy()

        grad_a, hess_a = _anchor_grad_hess(x[a_idx], v_a, c_a, mu_path)
        grad_b, hess_b = _anchor_grad_hess(x[b_idx], v_b, c_b, mu_path)

        H = M.copy()
        H[a_idx, a_idx] += hess_a
        H[b_idx, b_idx] += hess_b

        rhs = y + rho2 * D2.T @ (z - u)
        if D1s.shape[0] > 0:
            rhs += rho1 * D1s.T @ (w - r)

        # First-order consistent local quadratic model for the two anchor terms.
        rhs[a_idx] += hess_a * x[a_idx] - grad_a
        rhs[b_idx] += hess_b * x[b_idx] - grad_b

        x = np.linalg.solve(H, rhs)

        D2x = D2 @ x
        z_old = z
        z = soft_threshold(D2x + u, lam / rho2)
        u += D2x - z

        if D1s.shape[0] > 0:
            D1x = D1s @ x
            w_old = w
            w = soft_threshold(D1x + r, (0.5 * mu_path) / rho1)
            r += D1x - w
            prim_1 = np.linalg.norm(D1x - w)
            dual_1 = np.linalg.norm(rho1 * (w - w_old))
        else:
            prim_1 = 0.0
            dual_1 = 0.0

        prim_2 = np.linalg.norm(D2x - z)
        dual_2 = np.linalg.norm(rho2 * (z - z_old))
        x_step = np.max(np.abs(x - x_prev))

        if max(prim_1, dual_1, prim_2, dual_2, x_step) < tol:
            break

    return x

def path_cost_l1_trend_nn(
    y, lam=LAM, mu_path=PATH_MU, anchor_a=None, anchor_b=None,
    tol=SOLVER_TOL, max_outer_iters=OUTER_MAX_ITERS
):
    n = len(y)
    if n < 3:
        return y.copy(), {'a_idx': 0, 'b_idx': max(0, n-1), 'outer_iters': 0}

    if anchor_a is None:
        anchor_a = (0, float(y[0]))
    if anchor_b is None:
        anchor_b = (n-1, float(y[-1]))

    t_a, v_a = float(anchor_a[0]), float(anchor_a[1])
    t_b, v_b = float(anchor_b[0]), float(anchor_b[1])

    x = l1_admm(y, lam=lam, max_iter=220, tol=max(1e-5, tol*0.1))
    prev_a = prev_b = -1
    used_outer = 0

    for k in range(max_outer_iters):
        used_outer = k + 1
        a_idx, b_idx = _find_nn_anchor_indices(x, t_a, v_a, t_b, v_b)

        # Enforce ordering by swapping anchor labels if needed.
        if a_idx > b_idx:
            t_a, t_b = t_b, t_a
            v_a, v_b = v_b, v_a
            a_idx, b_idx = b_idx, a_idx

        x_new = path_cost_inner_admm(
            y=y, x0=x, lam=lam, mu_path=mu_path,
            t_a=t_a, v_a=v_a, t_b=t_b, v_b=v_b,
            a_idx=a_idx, b_idx=b_idx, max_iter=INNER_MAX_ITERS, tol=tol
        )

        delta = np.max(np.abs(x_new - x))
        x = x_new

        if (a_idx == prev_a) and (b_idx == prev_b) and (delta < tol):
            break
        prev_a, prev_b = a_idx, b_idx

    return x, {'a_idx': prev_a, 'b_idx': prev_b, 'outer_iters': used_outer}

def route_score(smt, dist):
    return np.trapezoid(smt**2) / (dist**0.6 + 1e-6)

# ═══════════════════════════════════════════════════════════
# 3.  AUTO LINE SEARCH FROM POPULATION DENSITY
# ═══════════════════════════════════════════════════════════
AUTO_LINE_COUNT = 4
AUTO_MIN_PEAK_SEP = 16
AUTO_PEAK_CANDIDATES = 34
AUTO_PATH_DENSITY_BIAS = 4.2
AUTO_REUSE_PENALTY = 0.55

LINE_COLORS = [
    '#0039A6',
    '#8888F6',
    '#FFD9A6',
    '#F039A6'
]

def pick_density_peaks(field, count=18, min_sep=AUTO_MIN_PEAK_SEP):
    flat = np.argsort(field.ravel())[::-1]
    peaks = []
    for idx in flat:
        yi, xi = np.unravel_index(idx, field.shape)
        if all((yi-py)**2 + (xi-px)**2 >= min_sep**2 for py, px in peaks):
            peaks.append((yi, xi))
            if len(peaks) >= count:
                break
    return peaks

def choose_endpoint_pair(peaks, field):
    if len(peaks) < 2:
        return (field.shape[0]//2, field.shape[1]//4), (field.shape[0]//2, 3*field.shape[1]//4)
    best = None
    best_score = -1e18
    for i in range(len(peaks)):
        y0, x0 = peaks[i]
        d0 = field[y0, x0]
        for j in range(i+1, len(peaks)):
            y1, x1 = peaks[j]
            d1 = field[y1, x1]
            dist = np.hypot(y1-y0, x1-x0)
            score = 0.65 * dist + 0.35 * (d0 + d1)
            if score > best_score:
                best_score = score
                best = ((y0, x0), (y1, x1))
    return best

def density_weighted_path(field, start, end, density_bias=AUTO_PATH_DENSITY_BIAS):
    h, w = field.shape
    eps = 1e-6
    norm = field / (field.max() + eps)
    nbrs = [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]

    def heuristic(y, x):
        return np.hypot(end[0]-y, end[1]-x) / (1.0 + density_bias * 0.5)

    g = np.full((h, w), np.inf)
    prev = {}
    g[start] = 0.0
    pq = [(heuristic(start[0], start[1]), 0.0, start)]

    while pq:
        _, cur_g, (y, x) = heapq.heappop(pq)
        if cur_g > g[y, x]:
            continue
        if (y, x) == end:
            break
        for dy, dx in nbrs:
            ny, nx = y + dy, x + dx
            if ny < 0 or ny >= h or nx < 0 or nx >= w:
                continue
            step = np.hypot(dy, dx)
            dens = 0.5 * (norm[y, x] + norm[ny, nx])
            step_cost = step / (1.0 + density_bias * dens)
            ng = cur_g + step_cost
            if ng < g[ny, nx]:
                g[ny, nx] = ng
                prev[(ny, nx)] = (y, x)
                heapq.heappush(pq, (ng + heuristic(ny, nx), ng, (ny, nx)))

    if (end not in prev) and (end != start):
        return [start, end]

    path = [end]
    cur = end
    while cur != start:
        cur = prev[cur]
        path.append(cur)
    path.reverse()
    return path

def path_to_xy(path_cells):
    xs, ys = [], []
    for yi, xi in path_cells:
        xs.append(xc[xi])
        ys.append(yc[yi])
    return np.array(xs), np.array(ys)

def sample_waypoints(xs, ys, count=18):
    if len(xs) <= count:
        idx = np.arange(len(xs))
    else:
        idx = np.linspace(0, len(xs)-1, count).astype(int)
    out = []
    for k, i in enumerate(idx):
        out.append((float(xs[i]), float(ys[i]), f"W{k+1:02d}"))
    return out

# ═══════════════════════════════════════════════════════════
# 4.  DISCOVER OPTIMAL LINES DIRECTLY FROM DENSITY FIELD
# ═══════════════════════════════════════════════════════════

print("Discovering optimal lines directly from population density…")
selected_lines = []
coverage = np.zeros_like(density)

for li in range(AUTO_LINE_COUNT):
    # Penalize already-used corridors to discover diverse high-value routes.
    search_field = density / (1.0 + AUTO_REUSE_PENALTY * coverage)
    peaks = pick_density_peaks(search_field, count=AUTO_PEAK_CANDIDATES, min_sep=AUTO_MIN_PEAK_SEP)
    start_cell, end_cell = choose_endpoint_pair(peaks, search_field)
    path_cells = density_weighted_path(search_field, start_cell, end_cell, density_bias=AUTO_PATH_DENSITY_BIAS)
    xs, ys = path_to_xy(path_cells)
    waypoints = sample_waypoints(xs, ys, count=18)
    pts = np.column_stack([ys, xs])
    raw = interp_density(pts)
    anchor_a = (0, float(raw[0]))
    anchor_b = (len(raw)-1, float(raw[-1]))
    smt, pc_meta = path_cost_l1_trend_nn(
        raw, lam=LAM, mu_path=PATH_MU,
        anchor_a=anchor_a, anchor_b=anchor_b,
        tol=SOLVER_TOL, max_outer_iters=OUTER_MAX_ITERS
    )
    dists = np.hypot(np.diff(xs), np.diff(ys))
    total_dist = dists.sum()
    score = route_score(smt, total_dist)

    color = LINE_COLORS[li % len(LINE_COLORS)]
    line_name = f"Optimised {li+1}"
    selected_lines.append({
        'name':  line_name,
        'color': color,
        'label': f"Auto line {li+1}",
        'loop':  False,
        'chain': {
            'xs': xs, 'ys': ys,
            'raw': raw, 'smooth': smt,
            'dist': total_dist, 'score': score,
            'anchors': pc_meta,
            'waypoints': waypoints,
        }
    })

    xi = np.clip((xs / xc[-1] * (GX-1)).astype(int), 0, GX-1)
    yi = np.clip((ys / yc[-1] * (GY-1)).astype(int), 0, GY-1)
    for px, py in zip(xi, yi):
        coverage[max(0, py-2):min(GY, py+3), max(0, px-2):min(GX, px+3)] += 1.0

    print(
        f"  {line_name:22s}  {len(path_cells)} path pts  "
        f"score={score:.0f}  anchors=({pc_meta['a_idx']},{pc_meta['b_idx']})  "
        f"outer={pc_meta['outer_iters']}"
    )

# Interchange detection
seen_ix = set(); unique_ix = []
for i, lnA in enumerate(selected_lines):
    for j, lnB in enumerate(selected_lines):
        if j <= i: continue
        for (ax, ay, an) in lnA['chain']['waypoints']:
            for (bx, by, bn) in lnB['chain']['waypoints']:
                if np.hypot(ax-bx, ay-by) < 0.14:
                    k = (round((ax+bx)/2, 1), round((ay+by)/2, 1))
                    if k not in seen_ix:
                        seen_ix.add(k); unique_ix.append(((ax+bx)/2, (ay+by)/2))

# ═══════════════════════════════════════════════════════════
# 5.  VISUALISATION
# ═══════════════════════════════════════════════════════════
DARK  = '#060a0f'; PANEL = '#0b1118'; GC = '#141e28'
TEXT  = '#dce8f8'; MUTED = '#4a5e72'

cmap_city = LinearSegmentedColormap.from_list('tokyo', [
    '#060a0f','#071828','#083858','#0c5c80',
    '#158888','#3dbb85','#a0e840','#ffc000','#ff4800','#ff0020'
])

fig = plt.figure(figsize=(26, 19), facecolor=DARK)
gs  = GridSpec(3, 5, figure=fig,
               hspace=0.40, wspace=0.30,
               left=0.03, right=0.97, top=0.935, bottom=0.05)

def sax(ax, title='', sub=''):
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values(): sp.set_edgecolor(GC)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.xaxis.label.set_color(MUTED); ax.yaxis.label.set_color(MUTED)
    ax.set_title(title+(f'\n{sub}' if sub else ''),
                 color=TEXT, fontsize=9, fontweight='bold',
                 fontfamily='monospace', pad=6)
    ax.grid(True, color=GC, lw=0.4, alpha=0.5)

# ── Panel A: Density ────────────────────────────────────────
ax_d = fig.add_subplot(gs[:2, :2])
ax_d.set_facecolor(DARK)
for sp in ax_d.spines.values(): sp.set_edgecolor(GC)
ax_d.tick_params(colors=MUTED, labelsize=8)

im = ax_d.imshow(density, origin='lower', cmap=cmap_city,
                 extent=[0,8,0,5.4], aspect='auto',
                 interpolation='bilinear', vmin=0, vmax=density.max()*0.88)
cb = plt.colorbar(im, ax=ax_d, fraction=0.018, pad=0.01)
cb.ax.tick_params(colors=MUTED, labelsize=7)
cb.set_label('Population / activity density', color=MUTED, fontsize=8)

for name, (lon, lat, sx_d, sy_d, peak) in PLACES.items():
    cx, cy = lx(lon), ly(lat)
    col = '#ff9500' if peak >= 95 else TEXT
    ms  = 160 if peak >= 95 else (60 if peak >= 70 else 25)
    ax_d.scatter(cx, cy, s=ms, color=col, zorder=5, edgecolors='black', lw=0.3)
    if peak >= 62:
        ax_d.annotate(name, (cx,cy), xytext=(cx+0.04, cy+0.06),
                      color=TEXT, fontsize=4.8, fontfamily='monospace',
                      fontweight='bold' if peak>=85 else 'normal')

ax_d.set_title('① FICTIONAL MULTI-CENTRE POPULATION DENSITY\nPolycentric hubs · corridors · suburbs',
               color=TEXT, fontsize=10, fontweight='bold', fontfamily='monospace', pad=8)
ax_d.set_xlabel('← West Fringe  →  East Fringe')
ax_d.set_ylabel('← Southern Belt  →  Northern Belt')
ax_d.set_xlim(0,8); ax_d.set_ylim(0,5.4)

# ── Panel B: Transit Network ─────────────────────────────────
ax_s = fig.add_subplot(gs[:2, 2:4])
ax_s.set_facecolor('#060d16')
for sp in ax_s.spines.values(): sp.set_edgecolor(GC)
ax_s.tick_params(colors=MUTED, labelsize=8)

ax_s.imshow(density, origin='lower', cmap=cmap_city,
            extent=[0,8,0,5.4], aspect='auto',
            interpolation='bilinear', alpha=0.12, vmin=0, vmax=density.max()*0.88)

for li, line in enumerate(selected_lines):
    col   = line['color']
    chain = line['chain']
    xs, ys = chain['xs'], chain['ys']

    ax_s.plot(xs, ys, color='white', lw=7.0, alpha=0.09, zorder=4,
              solid_capstyle='round', solid_joinstyle='round')
    ax_s.plot(xs, ys, color=col, lw=4.5, alpha=0.18, zorder=4,
              solid_capstyle='round', solid_joinstyle='round')
    ax_s.plot(xs, ys, color=col, lw=2.8, alpha=0.97, zorder=5,
              solid_capstyle='round', solid_joinstyle='round')

    # Station dots
    for (wx, wy, wname) in chain['waypoints']:
        ax_s.plot(wx, wy, 'o', color=col, ms=3.8, zorder=7,
                  markeredgecolor='white', markeredgewidth=0.9)

    # Terminals
    for (wx, wy, _) in [chain['waypoints'][0], chain['waypoints'][-1]]:
        if not line['loop']:
            ax_s.plot(wx, wy, 's', color=col, ms=6.0, zorder=8,
                      markeredgecolor='white', markeredgewidth=1.2)

    # Label near midpoint
    mid = len(xs) // 2
    ax_s.text(xs[mid]+0.04, ys[mid], line['name'],
              color=col, fontsize=5.0, fontfamily='monospace', fontweight='bold',
              zorder=9, bbox=dict(facecolor='#060d16', edgecolor='none', alpha=0.7, pad=0.5))

# Interchange nodes
for (cx, cy) in unique_ix:
    ax_s.scatter(cx, cy, s=80, color='white', zorder=10, edgecolors='#333', linewidths=1.0)
    ax_s.scatter(cx, cy, s=22, color='#111', zorder=11)

# Major labels
for name, (lon, lat, sx_d, sy_d, peak) in PLACES.items():
    if peak >= 78:
        ax_s.annotate(name, (lx(lon), ly(lat)), xytext=(lx(lon)+0.05, ly(lat)+0.08),
                      color=TEXT, fontsize=5.0, fontfamily='monospace',
                      fontweight='bold', zorder=12)

legend_h = [mpatches.Patch(color=ln['color'], label=ln['label'])
            for ln in selected_lines]
ax_s.legend(handles=legend_h, fontsize=6.2, facecolor='#06101a',
            edgecolor=GC, labelcolor=TEXT, loc='lower left', framealpha=0.93)

ax_s.set_title('② ℓ₁  OPTIMAL TRANSIT NETWORK  ·  FICTIONAL REGION\n'
               'Continuous piecewise-linear lines · ℓ₁ density filtering',
               color=TEXT, fontsize=10, fontweight='bold', fontfamily='monospace', pad=8)
ax_s.set_xlabel('← West  →  East'); ax_s.set_ylabel('← South  →  North')
ax_s.set_xlim(0,8); ax_s.set_ylim(0,5.4)

# ── Panel C: Coverage ────────────────────────────────────────
ax_cov = fig.add_subplot(gs[:2, 4])
ax_cov.set_facecolor(DARK)
for sp in ax_cov.spines.values(): sp.set_edgecolor(GC)
ax_cov.tick_params(colors=MUTED, labelsize=7)
cmap_cov = LinearSegmentedColormap.from_list('cov', [
    '#060a0f','#0a1e38','#0f5090','#22a8c0','#88ef60','#FF0000'])
cov_display = np.zeros((GY, GX), dtype=float)
for ln in selected_lines:
    chain = ln['chain']
    xi = np.clip((chain['xs']/xc[-1]*(GX-1)).astype(int), 0, GX-1)
    yi = np.clip((chain['ys']/yc[-1]*(GY-1)).astype(int), 0, GY-1)
    for px, py in zip(xi, yi):
        cov_display[max(0,py-2):min(GY,py+2), max(0,px-2):min(GX,px+2)] += 1
ax_cov.imshow(np.log1p(cov_display), origin='lower', cmap=cmap_cov,
              extent=[0,8,0,5.4], aspect='auto', interpolation='bilinear')
for name,(lon,lat,*_) in PLACES.items():
    ax_cov.scatter(lx(lon), ly(lat), s=12, color='white', zorder=4,
                   edgecolors='black', linewidths=0.3)
ax_cov.set_title('③ TRANSIT\nCOVERAGE\n(log scale\ndark=none\nred=max)',
                 color=TEXT, fontsize=8, fontweight='bold', fontfamily='monospace', pad=5)
ax_cov.set_xlabel('W → E'); ax_cov.set_ylabel('S → N')

# ── Panel D: ℓ₁ profiles ─────────────────────────────────────
ax_prof = fig.add_subplot(gs[2, :3])
sax(ax_prof, '④ ℓ₁-FILTERED RIDERSHIP PROFILES  —  DENSITY ALONG EACH CORRIDOR',
    f'Path-cost ℓ₁ trend + NN anchors (λ={LAM}, μ={PATH_MU}) · higher = denser corridor')
for li, line in enumerate(selected_lines):
    chain = line['chain']
    km = np.linspace(0, len(chain['smooth'])*0.12, len(chain['smooth']))
    lbl = f"{line['name']}  ({len(chain['waypoints'])} stops)"
    ax_prof.plot(km, chain['smooth'], color=line['color'], lw=2.0, label=lbl, alpha=0.92)
    ax_prof.fill_between(km, chain['smooth'], alpha=0.07, color=line['color'])
ax_prof.legend(fontsize=6.5, facecolor=DARK, edgecolor=GC, labelcolor=TEXT,
               loc='upper right', ncol=2)
ax_prof.set_xlabel('Approximate distance (display units)')
ax_prof.set_ylabel('Population density (ℓ₁ smoothed)')

# ── Panel E: Bar chart ───────────────────────────────────────
ax_bar = fig.add_subplot(gs[2, 3:])
sax(ax_bar, '⑤ LINE SCORES\ndensity² / length^0.6')
line_scores = [ln['chain']['score'] for ln in selected_lines]
line_names  = [ln['label'] for ln in selected_lines]
line_colors = [ln['color'] for ln in selected_lines]
order = np.argsort(line_scores)
ax_bar.barh([line_names[i] for i in order],
            [line_scores[i] for i in order],
            color=[line_colors[i] for i in order],
            edgecolor=GC, linewidth=0.5, height=0.72)
ax_bar.set_xlabel('chain score  (density² / length^0.6)')
ax_bar.tick_params(labelsize=6.2)

fig.suptitle(
    'ℓ₁  PATH-COST  TREND  FILTER  ·  FICTIONAL  POLYCENTRIC  TRANSIT  NETWORK',
    color=TEXT, fontsize=13, fontweight='bold', fontfamily='monospace', y=0.977
)

total_stops = sum(len(ln['chain']['waypoints']) for ln in selected_lines)
footer = (
    f"Fictional polycentric transit simulation  ·  {len(selected_lines)} lines  ·  "
    f"{total_stops} total stops  ·  {len(unique_ix)} interchange nodes  ·  "
    f"Algorithm: Path-Cost ℓ₁ Trend Filter + NN anchors (λ={LAM}, μ={PATH_MU})  ·  "
    f"Score: density²/length^0.6  ·  "
    f"Lines discovered by density-weighted path search (no manual station coordinates)"
)
fig.text(0.5, 0.003, footer, ha='center', color=MUTED, fontsize=7.0,
         fontfamily='monospace',
         bbox=dict(facecolor=PANEL, edgecolor=GC, boxstyle='round,pad=0.35'))

preferred_dir = Path('/mnt/user-data/outputs')
fallback_dir = Path('out_numpy')
try:
    preferred_dir.mkdir(parents=True, exist_ok=True)
    output_path = preferred_dir / 'fictional_polycentric_transit_network.png'
except OSError:
    fallback_dir.mkdir(parents=True, exist_ok=True)
    output_path = fallback_dir / 'fictional_polycentric_transit_network.png'

plt.savefig(output_path, dpi=165, bbox_inches='tight', facecolor=DARK)
print(f"Saved: {output_path}")
