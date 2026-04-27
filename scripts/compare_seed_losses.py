#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass


TAU = math.pi * 2.0


@dataclass(frozen=True)
class Vec:
    x: float
    y: float


def u32(value: int) -> int:
    return value & 0xFFFFFFFF


def imul(a: int, b: int) -> int:
    product = (a & 0xFFFFFFFF) * (b & 0xFFFFFFFF)
    product &= 0xFFFFFFFF
    return product if product < 0x80000000 else product - 0x100000000


def mulberry32(seed: int):
    t = u32(seed)

    def rng() -> float:
        nonlocal t
        t = u32(t + 0x6D2B79F5)
        value = imul(u32(t ^ (t >> 15)), u32(t | 1))
        value = u32(value ^ u32(value + imul(u32(value ^ (value >> 7)), u32(value | 61))))
        return u32(value ^ (value >> 14)) / 4294967296.0

    return rng


def random_normal(rng) -> float:
    u1 = max(rng(), 1e-9)
    u2 = rng()
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(TAU * u2)


def add(a: Vec, b: Vec) -> Vec:
    return Vec(a.x + b.x, a.y + b.y)


def sub(a: Vec, b: Vec) -> Vec:
    return Vec(a.x - b.x, a.y - b.y)


def scale(a: Vec, s: float) -> Vec:
    return Vec(a.x * s, a.y * s)


def dot(a: Vec, b: Vec) -> float:
    return a.x * b.x + a.y * b.y


def squared_dist(a: Vec, b: Vec) -> float:
    dx = a.x - b.x
    dy = a.y - b.y
    return dx * dx + dy * dy


def dist(a: Vec, b: Vec) -> float:
    return math.sqrt(squared_dist(a, b))


def norm(a: Vec) -> float:
    return math.sqrt(a.x * a.x + a.y * a.y)


def normalize(a: Vec) -> Vec:
    length = norm(a)
    if length < 1e-12:
        return Vec(0.0, 0.0)
    return scale(a, 1.0 / length)


def average(points: list[Vec]) -> Vec:
    if not points:
        return Vec(0.0, 0.0)
    return Vec(
        sum(point.x for point in points) / len(points),
        sum(point.y for point in points) / len(points),
    )


def seeded_point_cloud(seed: int = 12) -> list[Vec]:
    rng = mulberry32(seed)
    points: list[Vec] = []

    def push_cluster(center: Vec, count: int, sx: float, sy: float) -> None:
        for _ in range(count):
            points.append(Vec(center.x + random_normal(rng) * sx, center.y + random_normal(rng) * sy))

    push_cluster(Vec(0.0, 0.0), 320, 0.24, 0.19)
    push_cluster(Vec(-1.45, 0.9), 135, 0.19, 0.13)
    push_cluster(Vec(-1.2, -0.95), 128, 0.18, 0.14)
    push_cluster(Vec(-0.2, 1.4), 114, 0.17, 0.12)
    push_cluster(Vec(1.18, 1.08), 138, 0.19, 0.14)
    push_cluster(Vec(1.46, 0.28), 132, 0.16, 0.11)
    push_cluster(Vec(1.12, -0.85), 126, 0.17, 0.13)
    push_cluster(Vec(-0.1, -1.52), 120, 0.17, 0.16)

    for _ in range(240):
        t = rng() * 4.2 - 2.1
        points.append(Vec(0.98 * t + random_normal(rng) * 0.08, 0.5 * t + random_normal(rng) * 0.09))
    for _ in range(210):
        t = rng() * 3.8 - 1.9
        points.append(Vec(0.48 * t + random_normal(rng) * 0.08, -1.02 * t + random_normal(rng) * 0.08))

    return points


def farthest_point_sampling(points: list[Vec], count: int, seed: int = 77) -> list[Vec]:
    rng = mulberry32(seed + count * 17)
    first = int(rng() * len(points))
    chosen = [points[first]]
    used = {first}

    while len(chosen) < count:
        best_index = -1
        best_score = -1.0
        for index, point in enumerate(points):
            if index in used:
                continue
            min_d2 = min(squared_dist(point, center) for center in chosen)
            if min_d2 > best_score:
                best_score = min_d2
                best_index = index
        if best_index == -1:
            break
        used.add(best_index)
        chosen.append(points[best_index])

    return chosen[:count]


def sample_initial_hubs(points: list[Vec], count: int, seed: int = 77) -> list[Vec]:
    hubs = farthest_point_sampling(points, count, seed)
    result = []
    for index, point in enumerate(hubs):
        jitter = Vec(((index % 3) - 1) * 0.03, (((index + 1) % 3) - 1) * 0.03)
        result.append(add(point, jitter))
    return result


def nearest_center_index(point: Vec, centers: list[Vec]) -> int:
    best = 0
    best_dist = float("inf")
    for index, center in enumerate(centers):
        d2 = squared_dist(point, center)
        if d2 < best_dist:
            best = index
            best_dist = d2
    return best


def kmeans(points: list[Vec], k: int, iterations: int = 12, seed: int = 91, initial_centers: list[Vec] | None = None) -> tuple[list[Vec], list[int]]:
    if not points or k <= 0:
        return [], []
    center_count = min(k, len(points))
    centers = list(initial_centers[:center_count]) if initial_centers else sample_initial_hubs(points, center_count, seed)
    assignments = [0] * len(points)

    for _ in range(iterations):
        for index, point in enumerate(points):
            assignments[index] = nearest_center_index(point, centers)
        buckets = [[] for _ in range(center_count)]
        for index, point in enumerate(points):
            buckets[assignments[index]].append(point)
        centers = [average(bucket) if bucket else centers[index] for index, bucket in enumerate(buckets)]

    return centers, assignments


def canonicalize_edges(edges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    seen = set()
    canonical = []
    for a_raw, b_raw in edges:
        a = min(a_raw, b_raw)
        b = max(a_raw, b_raw)
        if a == b:
            continue
        key = (a, b)
        if key not in seen:
            seen.add(key)
            canonical.append(key)
    return canonical


def build_knn_edges(points: list[Vec], k: int = 2) -> list[tuple[int, int]]:
    edges = []
    for i, point in enumerate(points):
        neighbors = []
        for j, other in enumerate(points):
            if i == j:
                continue
            neighbors.append((dist(point, other), j))
        neighbors.sort()
        for _, j in neighbors[: min(k, len(neighbors))]:
            edges.append((i, j))
    return canonicalize_edges(edges)


def build_graph(points: list[Vec], edges: list[tuple[int, int]]) -> list[list[tuple[int, float]]]:
    adjacency = [[] for _ in points]
    for a, b in canonicalize_edges(edges):
        w = dist(points[a], points[b])
        adjacency[a].append((b, w))
        adjacency[b].append((a, w))
    return adjacency


def dijkstra(adjacency: list[list[tuple[int, float]]], start: int) -> tuple[list[float], list[int]]:
    n = len(adjacency)
    distances = [float("inf")] * n
    prev = [-1] * n
    visited = [False] * n
    distances[start] = 0.0

    for _ in range(n):
        best = float("inf")
        node = -1
        for i in range(n):
            if not visited[i] and distances[i] < best:
                best = distances[i]
                node = i
        if node == -1:
            break
        visited[node] = True
        for nxt, weight in adjacency[node]:
            candidate = distances[node] + weight
            if candidate < distances[nxt]:
                distances[nxt] = candidate
                prev[nxt] = node
    return distances, prev


def reconstruct_path(prev: list[int], start: int, end: int) -> list[int]:
    if start == end:
        return [start]
    path = []
    current = end
    while current != -1:
        path.append(current)
        if current == start:
            break
        current = prev[current]
    path.reverse()
    return path if path and path[0] == start else []


def assign_points(hubs: list[Vec], points: list[Vec]) -> tuple[list[list[Vec]], list[int], float]:
    assignments = [[] for _ in hubs]
    weights = [0] * len(hubs)
    sse = 0.0
    for point in points:
        best = nearest_center_index(point, hubs)
        assignments[best].append(point)
        weights[best] += 1
        sse += 0.5 * squared_dist(point, hubs[best])
    return assignments, weights, sse


def compute_path_smoothness(points: list[Vec], weights: list[int], edges: list[tuple[int, int]]) -> float:
    adjacency = build_graph(points, edges)
    total_mass = max(sum(weights), 1)
    smoothness = 0.0

    for i in range(len(points)):
        distances, prev = dijkstra(adjacency, i)
        for j in range(i + 1, len(points)):
            if not math.isfinite(distances[j]):
                continue
            path = reconstruct_path(prev, i, j)
            if not path:
                continue
            path_penalty = 0.0
            for p in range(len(path) - 1):
                path_penalty += dist(points[path[p]], points[path[p + 1]])
            smoothness += 0.5 * (weights[i] * weights[j]) / (total_mass * total_mass) * path_penalty
    return smoothness


def compute_graph_variation(points: list[Vec], edges: list[tuple[int, int]], mu: float) -> float:
    return sum(mu * dist(points[a], points[b]) for a, b in edges)


def point_to_segment_distance_sq(point: Vec, a: Vec, b: Vec) -> float:
    ab = sub(b, a)
    ap = sub(point, a)
    denom = dot(ab, ab)
    if denom < 1e-12:
        return squared_dist(point, a)
    t = max(0.0, min(1.0, dot(ap, ab) / denom))
    projection = add(a, scale(ab, t))
    return squared_dist(point, projection)


def principal_axis(points: list[Vec]) -> Vec:
    mean = average(points)
    sxx = syy = sxy = 0.0
    for point in points:
        dx = point.x - mean.x
        dy = point.y - mean.y
        sxx += dx * dx
        syy += dy * dy
        sxy += dx * dy
    if abs(sxy) < 1e-12:
        return Vec(1.0, 0.0) if sxx >= syy else Vec(0.0, 1.0)
    trace = sxx + syy
    det_term = math.sqrt(max((sxx - syy) * (sxx - syy) + 4.0 * sxy * sxy, 0.0))
    eigenvalue = 0.5 * (trace + det_term)
    axis = Vec(sxy, eigenvalue - sxx)
    return normalize(axis)


def order_chain(points: list[Vec]) -> list[Vec]:
    axis = principal_axis(points)
    center = average(points)
    return sorted(points, key=lambda point: dot(sub(point, center), axis))


def chain_length(chain: list[Vec]) -> float:
    return sum(dist(chain[i], chain[i + 1]) for i in range(len(chain) - 1))


def mppc_energies(chain: list[Vec], cloud: list[Vec], lambda1: float, lambda2: float) -> tuple[float, float, float]:
    if not chain:
        return 0.0, 0.0, 0.0
    discrete_fit = 0.0
    continuous_fit = 0.0
    for point in cloud:
        discrete_fit += min(squared_dist(point, node) for node in chain) / len(cloud)
        segment_terms = [point_to_segment_distance_sq(point, chain[i], chain[i + 1]) for i in range(len(chain) - 1)]
        best_segment = min(segment_terms) if segment_terms else min(squared_dist(point, node) for node in chain)
        continuous_fit += best_segment / len(cloud)
    length_term = chain_length(chain)
    component_term = lambda2 * 0.0
    discrete = lambda1 * (length_term + component_term) + discrete_fit
    continuous = lambda1 * (length_term + component_term) + continuous_fit
    return discrete, continuous, length_term


def app_objective(hubs: list[Vec], cloud: list[Vec], edges: list[tuple[int, int]], lambda_value: float, mu: float, station_cost: float) -> tuple[float, dict[str, float]]:
    assignments, weights, fx = assign_points(hubs, cloud)
    g1 = compute_path_smoothness(hubs, weights, edges)
    g2 = lambda_value * sum(point.x * point.x + point.y * point.y for point in hubs)
    g3 = compute_graph_variation(hubs, edges, mu)
    g4 = len(hubs) * station_cost
    objective = fx + g1 + g2 + g3 + g4
    return objective, {"fx": fx, "g1": g1, "g2": g2, "g3": g3, "g4": g4, "weights": float(sum(weights)), "assigned_clusters": float(len(assignments))}


def format_number(value: float) -> str:
    return f"{value:.6f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare MPPC and app.js losses on the same seeded point cloud.")
    parser.add_argument("--seed", type=int, default=12, help="Point-cloud seed from app.js")
    parser.add_argument("--hub-count", type=int, default=9, help="Shared support-point count")
    parser.add_argument("--lambda1", type=float, default=0.02, help="MPPC length weight")
    parser.add_argument("--lambda2", type=float, default=0.5, help="MPPC component weight")
    parser.add_argument("--lambda-app", type=float, default=0.12, dest="lambda_app", help="app.js shrinkage weight")
    parser.add_argument("--mu", type=float, default=0.26, help="app.js graph-TV weight")
    parser.add_argument("--station-cost", type=float, default=0.0, help="app.js station-count weight")
    args = parser.parse_args()

    cloud = seeded_point_cloud(args.seed)
    shared_hubs, _ = kmeans(cloud, args.hub_count, iterations=12, seed=91)
    app_edges = build_knn_edges(shared_hubs, 2)
    ordered_chain = order_chain(shared_hubs)

    app_total, app_terms = app_objective(shared_hubs, cloud, app_edges, args.lambda_app, args.mu, args.station_cost)
    mppc_discrete, mppc_continuous, length_term = mppc_energies(ordered_chain, cloud, args.lambda1, args.lambda2)
    _, _, shared_sse = assign_points(shared_hubs, cloud)

    print(f"# Same-seed loss comparison (seed={args.seed}, hubs={args.hub_count})")
    print()
    print("This uses one shared set of support points from the seeded cloud, then evaluates the native MPPC and `app.js` loss formulas on that same geometry.")
    print("The raw loss values are not directly interchangeable because the objectives are different, so shared fidelity terms are included for context.")
    print()
    print("| Metric | Value |")
    print("| --- | ---: |")
    print(f"| Cloud points | {len(cloud)} |")
    print(f"| Shared support points | {len(shared_hubs)} |")
    print(f"| Shared nearest-center SSE term used by `app.js` (`fx`) | {format_number(shared_sse)} |")
    print(f"| Ordered-chain length used by MPPC | {format_number(length_term)} |")
    print(f"| `app.js` objective total | {format_number(app_total)} |")
    print(f"| `app.js` fidelity `fx` | {format_number(app_terms['fx'])} |")
    print(f"| `app.js` path term `g1` | {format_number(app_terms['g1'])} |")
    print(f"| `app.js` shrinkage term `g2` | {format_number(app_terms['g2'])} |")
    print(f"| `app.js` graph-TV term `g3` | {format_number(app_terms['g3'])} |")
    print(f"| `app.js` station term `g4` | {format_number(app_terms['g4'])} |")
    print(f"| MPPC discrete energy | {format_number(mppc_discrete)} |")
    print(f"| MPPC continuous energy | {format_number(mppc_continuous)} |")


if __name__ == "__main__":
    main()
