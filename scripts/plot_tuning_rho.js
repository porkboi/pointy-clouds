#!/usr/bin/env node
"use strict";

const fs = require("fs");
const path = require("path");

const TAU = Math.PI * 2;

function mulberry32(seed) {
  let t = seed >>> 0;
  return () => {
    t += 0x6d2b79f5;
    let value = Math.imul(t ^ (t >>> 15), t | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

function randomNormal(rng) {
  const u1 = Math.max(rng(), 1e-9);
  const u2 = rng();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(TAU * u2);
}

function vec(x = 0, y = 0) {
  return { x, y };
}

function add(a, b) {
  return { x: a.x + b.x, y: a.y + b.y };
}

function sub(a, b) {
  return { x: a.x - b.x, y: a.y - b.y };
}

function scale(a, s) {
  return { x: a.x * s, y: a.y * s };
}

function dist(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function squaredDist(a, b) {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return dx * dx + dy * dy;
}

function norm(a) {
  return Math.hypot(a.x, a.y);
}

function normalize(a) {
  const n = norm(a);
  return n < 1e-9 ? vec(0, 0) : scale(a, 1 / n);
}

function average(points) {
  if (points.length === 0) {
    return vec(0, 0);
  }
  let sum = vec(0, 0);
  for (const point of points) {
    sum = add(sum, point);
  }
  return scale(sum, 1 / points.length);
}

function clonePoints(points) {
  return points.map((point) => ({ x: point.x, y: point.y }));
}

function seededPointCloud(seed = 12) {
  const rng = mulberry32(seed);
  const points = [];

  function pushCluster(center, count, sx, sy) {
    for (let i = 0; i < count; i += 1) {
      points.push(vec(center.x + randomNormal(rng) * sx, center.y + randomNormal(rng) * sy));
    }
  }

  pushCluster(vec(0, 0), 320, 0.24, 0.19);
  pushCluster(vec(-1.45, 0.9), 135, 0.19, 0.13);
  pushCluster(vec(-1.2, -0.95), 128, 0.18, 0.14);
  pushCluster(vec(-0.2, 1.4), 114, 0.17, 0.12);
  pushCluster(vec(1.18, 1.08), 138, 0.19, 0.14);
  pushCluster(vec(1.46, 0.28), 132, 0.16, 0.11);
  pushCluster(vec(1.12, -0.85), 126, 0.17, 0.13);
  pushCluster(vec(-0.1, -1.52), 120, 0.17, 0.16);

  for (let i = 0; i < 240; i += 1) {
    const t = rng() * 4.2 - 2.1;
    points.push(vec(0.98 * t + randomNormal(rng) * 0.08, 0.5 * t + randomNormal(rng) * 0.09));
  }
  for (let i = 0; i < 210; i += 1) {
    const t = rng() * 3.8 - 1.9;
    points.push(vec(0.48 * t + randomNormal(rng) * 0.08, -1.02 * t + randomNormal(rng) * 0.08));
  }

  return points;
}

function farthestPointSampling(points, count, seed = 77) {
  const rng = mulberry32(seed + count * 17);
  const first = Math.floor(rng() * points.length);
  const chosen = [points[first]];
  const used = new Set([first]);

  while (chosen.length < count) {
    let bestIndex = -1;
    let bestScore = -Infinity;
    for (let i = 0; i < points.length; i += 1) {
      if (used.has(i)) {
        continue;
      }
      let minD2 = Infinity;
      for (const center of chosen) {
        minD2 = Math.min(minD2, squaredDist(points[i], center));
      }
      if (minD2 > bestScore) {
        bestScore = minD2;
        bestIndex = i;
      }
    }
    if (bestIndex === -1) {
      break;
    }
    used.add(bestIndex);
    chosen.push(points[bestIndex]);
  }

  return clonePoints(chosen.slice(0, count));
}

function nearestCenterIndex(point, centers) {
  let best = 0;
  let bestDist = Infinity;
  for (let i = 0; i < centers.length; i += 1) {
    const d2 = squaredDist(point, centers[i]);
    if (d2 < bestDist) {
      bestDist = d2;
      best = i;
    }
  }
  return best;
}

function kMeans(points, k, iterations = 12, seed = 91, initialCenters = null) {
  if (points.length === 0 || k <= 0) {
    return { centers: [], assignments: [] };
  }
  const centerCount = Math.min(k, points.length);
  let centers = initialCenters ? clonePoints(initialCenters.slice(0, centerCount)) : farthestPointSampling(points, centerCount, seed);
  const assignments = Array(points.length).fill(0);

  for (let iter = 0; iter < iterations; iter += 1) {
    for (let i = 0; i < points.length; i += 1) {
      assignments[i] = nearestCenterIndex(points[i], centers);
    }
    const buckets = Array.from({ length: centerCount }, () => []);
    for (let i = 0; i < points.length; i += 1) {
      buckets[assignments[i]].push(points[i]);
    }
    centers = centers.map((center, index) => (buckets[index].length > 0 ? average(buckets[index]) : center));
  }

  return { centers, assignments };
}

function edgeKey(a, b) {
  return `${Math.min(a, b)}-${Math.max(a, b)}`;
}

function canonicalizeEdges(edges) {
  const unique = new Set();
  const canonical = [];
  for (const [aRaw, bRaw] of edges) {
    const a = Math.min(aRaw, bRaw);
    const b = Math.max(aRaw, bRaw);
    if (a === b) {
      continue;
    }
    const key = `${a}-${b}`;
    if (!unique.has(key)) {
      unique.add(key);
      canonical.push([a, b]);
    }
  }
  return canonical;
}

function graphFromEdges(points, edges) {
  const adjacency = points.map(() => []);
  for (const [a, b] of canonicalizeEdges(edges)) {
    const w = dist(points[a], points[b]);
    adjacency[a].push({ to: b, w });
    adjacency[b].push({ to: a, w });
  }
  return { adjacency, edges: canonicalizeEdges(edges) };
}

function buildKnnGraph(points, k = 2) {
  const edgeSet = new Set();
  const edges = [];

  for (let i = 0; i < points.length; i += 1) {
    const neighbors = [];
    for (let j = 0; j < points.length; j += 1) {
      if (i !== j) {
        neighbors.push({ j, d: dist(points[i], points[j]) });
      }
    }
    neighbors.sort((a, b) => a.d - b.d);
    for (const { j } of neighbors.slice(0, Math.min(k, neighbors.length))) {
      const key = edgeKey(i, j);
      if (!edgeSet.has(key)) {
        edgeSet.add(key);
        edges.push(key.split("-").map(Number));
      }
    }
  }

  return graphFromEdges(points, edges);
}

function buildCompleteGraph(points) {
  const edges = [];
  for (let i = 0; i < points.length; i += 1) {
    for (let j = i + 1; j < points.length; j += 1) {
      edges.push([i, j]);
    }
  }
  return graphFromEdges(points, edges);
}

function buildMaxGraph(points) {
  return buildCompleteGraph(points);
}

function dijkstra(adjacency, start) {
  const n = adjacency.length;
  const distArr = Array(n).fill(Infinity);
  const prev = Array(n).fill(-1);
  const visited = Array(n).fill(false);
  distArr[start] = 0;

  for (let step = 0; step < n; step += 1) {
    let u = -1;
    let best = Infinity;
    for (let i = 0; i < n; i += 1) {
      if (!visited[i] && distArr[i] < best) {
        best = distArr[i];
        u = i;
      }
    }
    if (u === -1) {
      break;
    }
    visited[u] = true;
    for (const { to, w } of adjacency[u]) {
      const candidate = distArr[u] + w;
      if (candidate < distArr[to]) {
        distArr[to] = candidate;
        prev[to] = u;
      }
    }
  }

  return { dist: distArr, prev };
}

function reconstructPath(prev, start, end) {
  if (start === end) {
    return [start];
  }
  const path = [];
  let current = end;
  while (current !== -1) {
    path.push(current);
    if (current === start) {
      break;
    }
    current = prev[current];
  }
  path.reverse();
  return path[0] === start ? path : [];
}

function isGraphConnected(points, edges) {
  if (points.length <= 1) {
    return true;
  }
  const graph = graphFromEdges(points, edges);
  const queue = [0];
  const seen = new Set([0]);
  while (queue.length > 0) {
    const node = queue.shift();
    for (const { to } of graph.adjacency[node]) {
      if (!seen.has(to)) {
        seen.add(to);
        queue.push(to);
      }
    }
  }
  return seen.size === points.length;
}

function assignPoints(hubs, points) {
  const assignments = Array.from({ length: hubs.length }, () => []);
  const weights = Array(hubs.length).fill(0);
  for (const point of points) {
    const best = nearestCenterIndex(point, hubs);
    assignments[best].push(point);
    weights[best] += 1;
  }
  return { assignments, weights };
}

function computeFx(hubs, assignments) {
  let total = 0;
  for (let i = 0; i < hubs.length; i += 1) {
    for (const point of assignments[i]) {
      total += 0.5 * squaredDist(point, hubs[i]);
    }
  }
  return total;
}

function computeG2(z2, lambda) {
  let sum = 0;
  for (const point of z2) {
    sum += point.x * point.x + point.y * point.y;
  }
  return lambda * sum;
}

function computeGraphVariation(points, edges, mu) {
  let total = 0;
  for (const [a, b] of edges) {
    total += mu * dist(points[a], points[b]);
  }
  return total;
}

function computeStationBuildCost(hubs, stationCost) {
  return hubs.length * stationCost;
}

function computePathSmoothness(points, weights, graph) {
  const n = points.length;
  const totalMass = Math.max(weights.reduce((sum, value) => sum + value, 0), 1);
  let smoothness = 0;
  let highlighted = { pair: [0, 0], path: [0], score: 0 };

  for (let i = 0; i < n; i += 1) {
    const result = dijkstra(graph.adjacency, i);
    for (let j = i + 1; j < n; j += 1) {
      if (!Number.isFinite(result.dist[j])) {
        continue;
      }
      const path = reconstructPath(result.prev, i, j);
      if (path.length === 0) {
        continue;
      }
      let pathPenalty = 0;
      for (let p = 0; p < path.length - 1; p += 1) {
        pathPenalty += dist(points[path[p]], points[path[p + 1]]);
      }
      const score = 0.5 * (weights[i] * weights[j]) / (totalMass * totalMass) * pathPenalty;
      smoothness += score;
      if (score > highlighted.score) {
        highlighted = { pair: [i, j], path, score };
      }
    }
  }

  return { value: smoothness, highlighted };
}

function computeClusterEnergies(assignments, hubs) {
  return assignments.map((bucket, index) => {
    let total = 0;
    for (const point of bucket) {
      total += squaredDist(point, hubs[index]);
    }
    return total;
  });
}

function evaluateObjective(hubs, edges, cloud, params) {
  const assigned = assignPoints(hubs, cloud);
  const graph = graphFromEdges(hubs, edges);
  const fx = computeFx(hubs, assigned.assignments);
  const g1Info = computePathSmoothness(hubs, assigned.weights, graph);
  const g2 = computeG2(hubs, params.lambda);
  const g3 = computeGraphVariation(hubs, graph.edges, params.mu);
  const g4 = computeStationBuildCost(hubs, params.stationCost);
  return {
    objective: fx + g1Info.value + g2 + g3 + g4,
    fx,
    g1: g1Info.value,
    g2,
    g3,
    g4,
    assignments: assigned.assignments,
    weights: assigned.weights,
    highlightedPath: g1Info.highlighted.path,
    highlightedPair: g1Info.highlighted.pair,
    graph,
  };
}

function proxX(state) {
  const vPoints = state.x.map((_, i) =>
    scale(add(add(sub(state.z1[i], state.u1[i]), sub(state.z2[i], state.u2[i])), sub(state.z3[i], state.u3[i])), 1 / 3),
  );
  const { assignments } = assignPoints(vPoints, state.cloud);
  const nextX = [];
  for (let i = 0; i < vPoints.length; i += 1) {
    const bucket = assignments[i];
    let sum = vec(0, 0);
    for (const point of bucket) {
      sum = add(sum, point);
    }
    const consensus = add(add(sub(state.z1[i], state.u1[i]), sub(state.z2[i], state.u2[i])), sub(state.z3[i], state.u3[i]));
    const denom = bucket.length + 3 * state.rho;
    nextX.push(scale(add(sum, scale(consensus, 1e-6)), 1 / Math.max(denom, 1e-6)));
  }
  return nextX;
}

function proxZ2(state) {
  const factor = state.rho / (state.rho + 2 * state.lambda);
  return state.x.map((point, i) => scale(add(point, state.u2[i]), factor));
}

function proxZ1(state, graph, weights) {
  const z = clonePoints(state.z1);
  const v = state.x.map((point, i) => add(point, state.u1[i]));
  const totalMass = Math.max(weights.reduce((sum, value) => sum + value, 0), 1);

  for (let step = 0; step < 15; step += 1) {
    const grad = z.map(() => vec(0, 0));
    for (let i = 0; i < z.length; i += 1) {
      const result = dijkstra(graph.adjacency, i);
      for (let j = i + 1; j < z.length; j += 1) {
        if (!Number.isFinite(result.dist[j])) {
          continue;
        }
        const coeff = 0.5 * (weights[i] * weights[j]) / (totalMass * totalMass);
        const path = reconstructPath(result.prev, i, j);
        for (let p = 0; p < path.length - 1; p += 1) {
          const a = path[p];
          const b = path[p + 1];
          const unit = normalize(sub(z[a], z[b]));
          grad[a] = add(grad[a], scale(unit, 1e-6 * coeff));
          grad[b] = add(grad[b], scale(unit, -1e-6 * coeff));
        }
      }
    }
    for (let i = 0; i < z.length; i += 1) {
      const proxGrad = add(grad[i], scale(sub(z[i], v[i]), state.rho));
      z[i] = sub(z[i], scale(proxGrad, 0.12));
    }
  }

  return z;
}

function proxZ3(state, edges) {
  const z = clonePoints(state.z3);
  const v = state.x.map((point, i) => add(point, state.u3[i]));

  for (let step = 0; step < 15; step += 1) {
    const grad = z.map(() => vec(0, 0));
    for (const [a, b] of edges) {
      const unit = normalize(sub(z[a], z[b]));
      grad[a] = add(grad[a], scale(unit, 1e-6));
      grad[b] = add(grad[b], scale(unit, -1e-6));
    }
    for (let i = 0; i < z.length; i += 1) {
      const proxGrad = add(grad[i], scale(sub(z[i], v[i]), state.rho));
      z[i] = sub(z[i], scale(proxGrad, 0.11));
    }
  }

  return z;
}

function residualSum(a, b) {
  let total = 0;
  for (let i = 0; i < a.length; i += 1) {
    total += dist(a[i], b[i]);
  }
  return total;
}

function pointSetNormSum(points) {
  let total = 0;
  for (const point of points) {
    total += norm(point);
  }
  return total;
}

function computeResidualThresholds(state, absTolerance = 0.0005, relTolerance = 0.004) {
  const variableCount = state.x.length * 3;
  const xNorm = pointSetNormSum(state.x);
  const zNorm = pointSetNormSum(state.z1) + pointSetNormSum(state.z2) + pointSetNormSum(state.z3);
  const uNorm = pointSetNormSum(state.u1) + pointSetNormSum(state.u2) + pointSetNormSum(state.u3);
  return {
    primal: absTolerance * variableCount + relTolerance * Math.max(xNorm, zNorm),
    dual: absTolerance * variableCount + relTolerance * state.rho * uNorm,
  };
}

function computeMotionStats(current, previous) {
  let total = 0;
  let max = 0;
  for (let i = 0; i < current.length; i += 1) {
    const motion = dist(current[i], previous[i]);
    total += motion;
    max = Math.max(max, motion);
  }
  return { total, avg: current.length ? total / current.length : 0, max };
}

function resetAdmmVariables(state, x, edges) {
  state.x = clonePoints(x);
  state.z1 = clonePoints(x);
  state.z2 = clonePoints(x);
  state.z3 = clonePoints(x);
  state.u1 = x.map(() => vec(0, 0));
  state.u2 = x.map(() => vec(0, 0));
  state.u3 = x.map(() => vec(0, 0));
  state.weights = Array(x.length).fill(0);
  state.edges = canonicalizeEdges(edges);
  state.highlightedPath = [0];
  state.highlightedPair = [0, 0];
}

function createBaseState(mode, cloud, x, edges, params) {
  const state = {
    mode,
    cloud,
    iteration: 0,
    outerIteration: 0,
    optimizer: params.optimizer,
    lambda: params.lambda,
    mu: params.mu,
    rho: params.rho,
    stationCost: params.stationCost,
    sgdLearningRate: params.sgdLearningRate,
    sgdDecay: params.sgdDecay,
    sgdBatchSize: params.sgdBatchSize,
    sgdPairSamples: params.sgdPairSamples,
    sgdRng: mulberry32(params.sgdSeed + x.length * 31 + mode.length * 17),
    history: [],
    lastAction: "Initialised",
    forceStop: false,
  };
  resetAdmmVariables(state, x, edges);
  return state;
}

function createTuningState(cloud, params) {
  const clustering = kMeans(cloud, 3, 16, 111);
  const graph = buildMaxGraph(clustering.centers);
  const state = createBaseState("tuning", cloud, clustering.centers, graph.edges, params);
  state.lastAction = "Start with 3 hubs from KMeans";
  return state;
}

function recordHistory(state, metrics, primal, dual) {
  state.weights = metrics.weights;
  state.highlightedPath = metrics.highlightedPath;
  state.highlightedPair = metrics.highlightedPair;
  state.history.push({
    objective: metrics.objective,
    primal,
    dual,
    fx: metrics.fx,
    g1: metrics.g1,
    g2: metrics.g2,
    g3: metrics.g3,
    g4: metrics.g4,
    hubCount: state.x.length,
    edgeCount: state.edges.length,
  });
}

function runAdmmIteration(state, graphMode = "knn", pushHistory = true, fixedEdges = null) {
  const previousZ1 = clonePoints(state.z1);
  const previousZ2 = clonePoints(state.z2);
  const previousZ3 = clonePoints(state.z3);
  const assignmentBefore = assignPoints(state.x, state.cloud);
  state.weights = assignmentBefore.weights;
  state.x = proxX(state);
  const graph =
    fixedEdges !== null
      ? graphFromEdges(state.x, fixedEdges)
      : graphMode === "fixed"
        ? graphFromEdges(state.x, state.edges)
        : buildKnnGraph(state.x, 2);
  state.edges = graph.edges;
  state.z1 = proxZ1(state, graph, assignmentBefore.weights);
  state.z2 = proxZ2(state);
  state.z3 = proxZ3(state, graph.edges);

  for (let i = 0; i < state.x.length; i += 1) {
    state.u1[i] = add(state.u1[i], sub(state.x[i], state.z1[i]));
    state.u2[i] = add(state.u2[i], sub(state.x[i], state.z2[i]));
    state.u3[i] = add(state.u3[i], sub(state.x[i], state.z3[i]));
  }

  const metrics = evaluateObjective(state.x, state.edges, state.cloud, state);
  const primal = residualSum(state.x, state.z1) + residualSum(state.x, state.z2) + residualSum(state.x, state.z3);
  const dual =
    state.rho * residualSum(state.z1, previousZ1) +
    state.rho * residualSum(state.z2, previousZ2) +
    state.rho * residualSum(state.z3, previousZ3);
  state.iteration += 1;
  if (pushHistory) {
    recordHistory(state, metrics, primal, dual);
  }
  return { metrics, primal, dual };
}

function sampleWithoutReplacement(points, count, rng) {
  if (count >= points.length) {
    return points.slice();
  }
  const indices = new Set();
  while (indices.size < count) {
    indices.add(Math.floor(rng() * points.length));
  }
  return [...indices].map((index) => points[index]);
}

function computeSgdGradient(x, cloud, params, rng, edges = null) {
  const grad = x.map(() => vec(0, 0));
  const batch = sampleWithoutReplacement(cloud, Math.min(params.sgdBatchSize, cloud.length), rng);
  const batchScale = cloud.length / Math.max(batch.length, 1);

  for (const point of batch) {
    const best = nearestCenterIndex(point, x);
    grad[best] = add(grad[best], scale(sub(x[best], point), batchScale));
  }
  for (let i = 0; i < x.length; i += 1) {
    grad[i] = add(grad[i], scale(x[i], 2 * params.lambda));
  }

  const graph = edges ? graphFromEdges(x, edges) : buildKnnGraph(x, 2);
  for (const [a, b] of graph.edges) {
    const unit = normalize(sub(x[a], x[b]));
    grad[a] = add(grad[a], scale(unit, params.mu));
    grad[b] = add(grad[b], scale(unit, -params.mu));
  }

  const { weights } = assignPoints(x, cloud);
  const totalMass = Math.max(weights.reduce((sum, value) => sum + value, 0), 1);
  const sampledPairs = Math.min(params.sgdPairSamples, (x.length * (x.length - 1)) / 2);
  const pairNormaliser = ((x.length * (x.length - 1)) / 2) / Math.max(sampledPairs, 1);

  for (let sample = 0; sample < sampledPairs; sample += 1) {
    const i = Math.floor(rng() * x.length);
    let j = Math.floor(rng() * (x.length - 1));
    if (j >= i) {
      j += 1;
    }
    const lo = Math.min(i, j);
    const hi = Math.max(i, j);
    const result = dijkstra(graph.adjacency, lo);
    if (!Number.isFinite(result.dist[hi])) {
      continue;
    }
    const coeff = pairNormaliser * 0.5 * (weights[lo] * weights[hi]) / (totalMass * totalMass);
    const path = reconstructPath(result.prev, lo, hi);
    for (let p = 0; p < path.length - 1; p += 1) {
      const a = path[p];
      const b = path[p + 1];
      const unit = normalize(sub(x[a], x[b]));
      grad[a] = add(grad[a], scale(unit, coeff));
      grad[b] = add(grad[b], scale(unit, -coeff));
    }
  }

  return { grad, graph };
}

function runSgdIteration(state, graphMode = "knn", pushHistory = true, fixedEdges = null) {
  const previousX = clonePoints(state.x);
  const previousObjective = state.history[state.history.length - 1]?.objective ?? null;
  const graphEdges = fixedEdges !== null ? fixedEdges : graphMode === "fixed" ? state.edges : null;
  const { grad } = computeSgdGradient(state.x, state.cloud, state, state.sgdRng, graphEdges);
  const learningRate = state.sgdLearningRate / Math.sqrt(1 + state.iteration * state.sgdDecay);
  state.x = state.x.map((point, index) => sub(point, scale(grad[index], learningRate)));
  const graph =
    fixedEdges !== null
      ? graphFromEdges(state.x, fixedEdges)
      : graphMode === "fixed"
        ? graphFromEdges(state.x, state.edges)
        : buildKnnGraph(state.x, 2);
  state.edges = graph.edges;
  const metrics = evaluateObjective(state.x, state.edges, state.cloud, state);
  const motion = computeMotionStats(state.x, previousX);
  const objectiveDelta = previousObjective === null ? 0 : Math.abs(metrics.objective - previousObjective);
  state.iteration += 1;
  if (pushHistory) {
    recordHistory(state, metrics, motion.total, objectiveDelta);
  }
  return { metrics, primal: motion.total, dual: objectiveDelta, motion };
}

function runFixedKSgdConvergence(state, options = {}) {
  const {
    maxIterations = 96,
    minIterations = 12,
    pushHistory = true,
    graphMode = "knn",
    motionTolerance = 0.0005,
    objectiveTolerance = 0.0005,
  } = options;

  let latest = null;
  let motion = { total: Infinity, avg: Infinity, max: Infinity };
  const fixedEdges = graphMode === "knn" ? buildKnnGraph(state.x, 2).edges : canonicalizeEdges(state.edges);
  state.edges = fixedEdges;

  for (let iteration = 0; iteration < maxIterations; iteration += 1) {
    latest = runSgdIteration(state, graphMode, pushHistory, fixedEdges);
    motion = latest.motion;
    const settled = motion.avg <= motionTolerance && latest.dual <= objectiveTolerance;
    if (iteration + 1 >= minIterations && settled) {
      return { ...latest, motion, settled: true, iterations: iteration + 1 };
    }
  }

  return { ...latest, motion, settled: false, iterations: maxIterations };
}

function runFixedKConvergence(state, options = {}) {
  if (state.optimizer === "sgd") {
    return runFixedKSgdConvergence(state, options);
  }

  const {
    maxIterations = 48,
    minIterations = 6,
    primalAbsTolerance = 0.0005,
    primalRelTolerance = 0.004,
    pushHistory = true,
    graphMode = "knn",
  } = options;

  let latest = null;
  let motion = { total: Infinity, avg: Infinity, max: Infinity };
  const fixedEdges = graphMode === "knn" ? buildKnnGraph(state.x, 2).edges : canonicalizeEdges(state.edges);
  state.edges = fixedEdges;

  for (let iteration = 0; iteration < maxIterations; iteration += 1) {
    const previousX = clonePoints(state.x);
    latest = runAdmmIteration(state, graphMode, pushHistory, fixedEdges);
    motion = computeMotionStats(state.x, previousX);
    const thresholds = computeResidualThresholds(state, primalAbsTolerance, primalRelTolerance);
    const residualSettled = latest.primal <= thresholds.primal && latest.dual <= thresholds.dual;
    if (iteration + 1 >= minIterations && residualSettled) {
      return { ...latest, motion, settled: true, iterations: iteration + 1 };
    }
  }

  return { ...latest, motion, settled: false, iterations: maxIterations };
}

function optimizeEdgesForState(state, metrics = null) {
  if (state.x.length <= 1) {
    state.edges = [];
    return { edges: [], metrics: metrics ?? evaluateObjective(state.x, state.edges, state.cloud, state), removedEdges: 0, improved: false };
  }

  let currentEdges = canonicalizeEdges(state.edges);
  let canReuseMetrics = Boolean(metrics) && currentEdges.length === state.edges.length;
  if (!isGraphConnected(state.x, currentEdges)) {
    currentEdges = buildCompleteGraph(state.x).edges;
    canReuseMetrics = false;
  }

  let currentMetrics = canReuseMetrics ? metrics : evaluateObjective(state.x, currentEdges, state.cloud, state);
  let improved = false;
  let removedEdges = 0;

  while (currentEdges.length > state.x.length - 1) {
    let bestCandidate = null;

    for (let edgeIndex = 0; edgeIndex < currentEdges.length; edgeIndex += 1) {
      const candidateEdges = currentEdges.filter((_, index) => index !== edgeIndex);
      if (!isGraphConnected(state.x, candidateEdges)) {
        continue;
      }
      const candidateMetrics = evaluateObjective(state.x, candidateEdges, state.cloud, state);
      const delta = currentMetrics.objective - candidateMetrics.objective;
      if (delta > 1e-6 && (!bestCandidate || delta > bestCandidate.delta)) {
        bestCandidate = { edges: candidateEdges, metrics: candidateMetrics, delta };
      }
    }

    if (!bestCandidate) {
      break;
    }

    currentEdges = bestCandidate.edges;
    currentMetrics = bestCandidate.metrics;
    improved = true;
    removedEdges += 1;
  }

  state.edges = currentEdges;
  return { edges: currentEdges, metrics: currentMetrics, removedEdges, improved };
}

function runConvergedOptimizationWithEdgeOptimization(state, options = {}) {
  const convergenceOptions = {
    maxIterations: state.optimizer === "sgd" ? 96 : 48,
    minIterations: state.optimizer === "sgd" ? 12 : 6,
    primalAbsTolerance: 0.0005,
    primalRelTolerance: 0.004,
    pushHistory: true,
    graphMode: "knn",
    ...options,
  };

  const convergence = runFixedKConvergence(state, convergenceOptions);
  const edgeOptimization = optimizeEdgesForState(state, convergence.metrics);
  if (!edgeOptimization.improved) {
    return { ...convergence, edgeOptimization };
  }
  resetAdmmVariables(state, state.x, edgeOptimization.edges);
  const settled = runFixedKConvergence(state, { ...convergenceOptions, graphMode: "fixed" });
  return { ...settled, edgeOptimization };
}

function runFinalAdmmCompletion(state) {
  if (state.optimizer === "sgd") {
    return runFixedKConvergence(state, {
      maxIterations: 160,
      minIterations: 20,
      graphMode: "knn",
      motionTolerance: 0.00025,
      objectiveTolerance: 0.00025,
    });
  }
  return runFixedKConvergence(state, {
    maxIterations: 96,
    minIterations: 12,
    primalAbsTolerance: 0.00025,
    primalRelTolerance: 0.002,
    graphMode: "knn",
  });
}

function simulateFixedHubRefinement(baseState, x, edges) {
  const state = createBaseState("simulation", baseState.cloud, x, edges, baseState);
  const convergence = runConvergedOptimizationWithEdgeOptimization(state, { pushHistory: false, graphMode: "knn" });
  return { state, metrics: convergence.metrics, convergence };
}

function simulateFinalAdmmCompletion(baseState, x, edges) {
  const state = createBaseState("simulation", baseState.cloud, x, edges, baseState);
  const completion = runFinalAdmmCompletion(state);
  return { state, metrics: completion.metrics, completion };
}

function buildSplitProposal(state) {
  const currentEval = evaluateObjective(state.x, state.edges, state.cloud, state);
  const clusterEnergies = computeClusterEnergies(currentEval.assignments, state.x);
  let clusterIndex = -1;
  let bestEnergy = -Infinity;
  for (let i = 0; i < clusterEnergies.length; i += 1) {
    if (currentEval.assignments[i].length < 2) {
      continue;
    }
    if (clusterEnergies[i] > bestEnergy) {
      bestEnergy = clusterEnergies[i];
      clusterIndex = i;
    }
  }
  if (clusterIndex === -1) {
    return null;
  }

  const clusterPoints = currentEval.assignments[clusterIndex];
  const splitSeed = state.outerIteration + clusterIndex * 19 + state.x.length * 7;
  const split = kMeans(clusterPoints, 2, 12, splitSeed);
  if (split.centers.length < 2) {
    return null;
  }

  let best = null;
  for (const candidate of split.centers) {
    const proposalHubs = clonePoints(state.x);
    proposalHubs.push(candidate);
    const candidateIndex = proposalHubs.length - 1;
    for (let anchorIndex = 0; anchorIndex < state.x.length; anchorIndex += 1) {
      const proposalEdges = canonicalizeEdges(state.edges.concat([[anchorIndex, candidateIndex]]));
      const simulation = simulateFixedHubRefinement(state, proposalHubs, proposalEdges);
      const delta = currentEval.objective - simulation.metrics.objective;
      if (!best || delta > best.delta) {
        best = { delta, clusterIndex, candidate, anchorIndex, simulation };
      }
    }
  }

  return best;
}

function adoptSimulationState(target, simulation, lastAction) {
  resetAdmmVariables(target, simulation.state.x, simulation.state.edges);
  target.lambda = simulation.state.lambda;
  target.mu = simulation.state.mu;
  target.rho = simulation.state.rho;
  target.stationCost = simulation.state.stationCost;
  target.lastAction = lastAction;
}

function stepTuningState(state) {
  if (state.x.length === 0 || state.cloud.length === 0) {
    return;
  }

  let convergence = runConvergedOptimizationWithEdgeOptimization(state, { graphMode: "knn" });
  const baselineX = clonePoints(state.x);
  const baselineEdges = canonicalizeEdges(state.edges);
  const baselineIteration = state.iteration;
  const baselineWeights = state.weights.slice();
  const baselineHighlightedPath = [...state.highlightedPath];
  const baselineHighlightedPair = [...state.highlightedPair];
  const baselineHistoryLength = state.history.length;
  const baselineFinal = simulateFinalAdmmCompletion(state, baselineX, baselineEdges);
  const baselineFinalObjective = baselineFinal.metrics.objective;

  state.outerIteration += 1;
  const splitProposal = buildSplitProposal(state);

  if (splitProposal && splitProposal.delta > 0) {
    adoptSimulationState(
      state,
      splitProposal.simulation,
      `Added hub from Voronoi split ${splitProposal.clusterIndex} via ${splitProposal.anchorIndex} (delta=${splitProposal.delta.toFixed(3)})`,
    );
    convergence = runConvergedOptimizationWithEdgeOptimization(state, { graphMode: "knn" });
    const finalConvergence = runFinalAdmmCompletion(state);
    if (finalConvergence.metrics.objective + 1e-6 < baselineFinalObjective || state.forceStop) {
      state.lastAction = `Accepted split after ${convergence.iterations} settle steps`;
      return;
    }

    resetAdmmVariables(state, baselineX, baselineEdges);
    state.iteration = baselineIteration;
    state.weights = baselineWeights;
    state.highlightedPath = baselineHighlightedPath;
    state.highlightedPair = baselineHighlightedPair;
    state.history.length = baselineHistoryLength;
    state.lastAction = `Rejected split at outer step ${state.outerIteration}`;
    state.forceStop = true;
    return;
  }

  state.lastAction = "No improving hub split";
}

function runTuningSolver(cloud, params) {
  const state = createTuningState(cloud, params);
  for (let outer = 0; outer < params.maxOuterIterations; outer += 1) {
    stepTuningState(state);
    if (state.forceStop || state.lastAction === "No improving hub split") {
      break;
    }
  }
  const finalCompletion = simulateFinalAdmmCompletion(state, state.x, state.edges);
  return {
    optimizer: params.optimizer,
    rho: params.rho,
    objective: finalCompletion.metrics.objective,
    fx: finalCompletion.metrics.fx,
    g1: finalCompletion.metrics.g1,
    g2: finalCompletion.metrics.g2,
    g3: finalCompletion.metrics.g3,
    g4: finalCompletion.metrics.g4,
    hubCount: finalCompletion.state.x.length,
    edgeCount: finalCompletion.state.edges.length,
    outerIterations: state.outerIteration,
    settledIterations: finalCompletion.completion.iterations,
  };
}

function parseArgs(argv) {
  const options = {
    output: path.join(process.cwd(), "out", "tuning_rho_vs_objective.svg"),
    csv: path.join(process.cwd(), "out", "tuning_rho_vs_objective.csv"),
    rhoValues: [0.2, 0.4, 0.6, 0.8, 1.0, 1.15, 1.4, 1.8, 2.2, 2.8],
  };

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === "--rho-values" && i + 1 < argv.length) {
      options.rhoValues = argv[i + 1]
        .split(",")
        .map((value) => Number.parseFloat(value.trim()))
        .filter((value) => Number.isFinite(value) && value > 0);
      i += 1;
      continue;
    }
    if (arg === "--output" && i + 1 < argv.length) {
      options.output = path.resolve(argv[i + 1]);
      i += 1;
      continue;
    }
    if (arg === "--csv" && i + 1 < argv.length) {
      options.csv = path.resolve(argv[i + 1]);
      i += 1;
    }
  }

  options.rhoValues = [...new Set(options.rhoValues)].sort((a, b) => a - b);
  return options;
}

function formatNumber(value, digits = 6) {
  return Number.isFinite(value) ? value.toFixed(digits) : String(value);
}

function escapeXml(text) {
  return String(text)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function buildDualSeriesSvg(results) {
  const width = 1080;
  const height = 720;
  const margin = { top: 80, right: 70, bottom: 90, left: 110 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  const rhos = results.map((row) => row.rho);
  const values = results.flatMap((row) => [row.admm.objective, row.sgd.objective]);
  const minX = Math.min(...rhos);
  const maxX = Math.max(...rhos);
  const rawMinY = Math.min(...values);
  const rawMaxY = Math.max(...values);
  const padY = Math.max((rawMaxY - rawMinY) * 0.08, 1e-3);
  const minY = rawMinY - padY;
  const maxY = rawMaxY + padY;

  function xScale(value) {
    if (maxX === minX) {
      return margin.left + plotWidth / 2;
    }
    return margin.left + ((value - minX) / (maxX - minX)) * plotWidth;
  }

  function yScale(value) {
    return margin.top + ((maxY - value) / (maxY - minY)) * plotHeight;
  }

  function buildPath(key) {
    return results
      .map((row, index) => `${index === 0 ? "M" : "L"} ${xScale(row.rho).toFixed(2)} ${yScale(row[key].objective).toFixed(2)}`)
      .join(" ");
  }

  const xTicks = rhos;
  const yTicks = [];
  for (let i = 0; i <= 6; i += 1) {
    yTicks.push(minY + ((maxY - minY) * i) / 6);
  }

  const xGrid = xTicks.map((tick) => {
    const x = xScale(tick);
    return `
  <line x1="${x.toFixed(2)}" y1="${margin.top}" x2="${x.toFixed(2)}" y2="${(height - margin.bottom).toFixed(2)}" stroke="#e5e7eb" stroke-width="1" />
  <text x="${x.toFixed(2)}" y="${(height - margin.bottom + 28).toFixed(2)}" text-anchor="middle" font-size="15" fill="#374151">${escapeXml(formatNumber(tick, 2))}</text>`;
  }).join("");

  const yGrid = yTicks.map((tick) => {
    const y = yScale(tick);
    return `
  <line x1="${margin.left}" y1="${y.toFixed(2)}" x2="${(width - margin.right).toFixed(2)}" y2="${y.toFixed(2)}" stroke="#e5e7eb" stroke-width="1" />
  <text x="${(margin.left - 16).toFixed(2)}" y="${(y + 5).toFixed(2)}" text-anchor="end" font-size="15" fill="#374151">${escapeXml(formatNumber(tick, 3))}</text>`;
  }).join("");

  const points = results.map((row) => {
    const admmX = xScale(row.rho);
    const admmY = yScale(row.admm.objective);
    const sgdY = yScale(row.sgd.objective);
    return `
  <circle cx="${admmX.toFixed(2)}" cy="${admmY.toFixed(2)}" r="5.5" fill="#c2410c" stroke="#fff7ed" stroke-width="2" />
  <circle cx="${admmX.toFixed(2)}" cy="${sgdY.toFixed(2)}" r="5.5" fill="#0f766e" stroke="#f0fdfa" stroke-width="2" />`;
  }).join("");

  return `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" role="img" aria-labelledby="title desc">
  <title id="title">Hub tuning objective versus rho</title>
  <desc id="desc">Line plot of final tuning objective against rho for ADMM and SGD using the hub tuning flow from app.js.</desc>
  <rect width="${width}" height="${height}" fill="#fffbf5" />
  <rect x="${margin.left}" y="${margin.top}" width="${plotWidth}" height="${plotHeight}" rx="18" fill="#ffffff" stroke="#fed7aa" stroke-width="1.5" />
  ${xGrid}
  ${yGrid}
  <path d="${buildPath("admm")}" fill="none" stroke="#ea580c" stroke-width="4" stroke-linecap="round" stroke-linejoin="round" />
  <path d="${buildPath("sgd")}" fill="none" stroke="#0f766e" stroke-width="4" stroke-linecap="round" stroke-linejoin="round" />
  ${points}
  <text x="${(width / 2).toFixed(2)}" y="42" text-anchor="middle" font-size="28" font-weight="700" fill="#7c2d12">Hub Tuning Final Objective vs Rho</text>
  <text x="${(width / 2).toFixed(2)}" y="70" text-anchor="middle" font-size="16" fill="#7c2d12">Each curve runs the app.js tuning loop independently for ADMM and SGD.</text>
  <text x="${(width / 2).toFixed(2)}" y="${(height - 24).toFixed(2)}" text-anchor="middle" font-size="18" fill="#431407">rho</text>
  <text x="28" y="${(height / 2).toFixed(2)}" transform="rotate(-90 28 ${(height / 2).toFixed(2)})" text-anchor="middle" font-size="18" fill="#431407">final objective</text>
  <text x="${(width - margin.right - 165).toFixed(2)}" y="${(margin.top + 18).toFixed(2)}" font-size="16" fill="#c2410c">ADMM</text>
  <line x1="${(width - margin.right - 245).toFixed(2)}" y1="${(margin.top + 12).toFixed(2)}" x2="${(width - margin.right - 178).toFixed(2)}" y2="${(margin.top + 12).toFixed(2)}" stroke="#ea580c" stroke-width="4" />
  <text x="${(width - margin.right - 65).toFixed(2)}" y="${(margin.top + 18).toFixed(2)}" font-size="16" fill="#0f766e">SGD</text>
  <line x1="${(width - margin.right - 145).toFixed(2)}" y1="${(margin.top + 12).toFixed(2)}" x2="${(width - margin.right - 78).toFixed(2)}" y2="${(margin.top + 12).toFixed(2)}" stroke="#0f766e" stroke-width="4" />
</svg>`;
}

function writeOutputs(results, options) {
  fs.mkdirSync(path.dirname(options.output), { recursive: true });
  fs.mkdirSync(path.dirname(options.csv), { recursive: true });
  fs.writeFileSync(options.output, buildDualSeriesSvg(results));
  const csv = [
    "rho,admm_objective,sgd_objective,admm_hubs,sgd_hubs,admm_edges,sgd_edges",
    ...results.map((row) =>
      [
        formatNumber(row.rho),
        formatNumber(row.admm.objective),
        formatNumber(row.sgd.objective),
        row.admm.hubCount,
        row.sgd.hubCount,
        row.admm.edgeCount,
        row.sgd.edgeCount,
      ].join(","),
    ),
  ].join("\n");
  fs.writeFileSync(options.csv, `${csv}\n`);
}

function printSummary(results) {
  console.log("rho\tadmm_objective\tsgd_objective\tadmm_hubs\tsgd_hubs");
  for (const row of results) {
    console.log(
      [
        formatNumber(row.rho, 2),
        formatNumber(row.admm.objective, 3),
        formatNumber(row.sgd.objective, 3),
        row.admm.hubCount,
        row.sgd.hubCount,
      ].join("\t"),
    );
  }
}

function main() {
  const options = parseArgs(process.argv.slice(2));
  const baseParams = {
    lambda: 0.12,
    mu: 0.26,
    stationCost: 0,
    sgdLearningRate: 0.003,
    sgdDecay: 0.04,
    sgdBatchSize: 160,
    sgdPairSamples: 12,
    sgdSeed: 20260426,
    maxOuterIterations: 18,
  };

  const cloud = seededPointCloud(12);
  const results = options.rhoValues.map((rho) => ({
    rho,
    admm: runTuningSolver(cloud, { ...baseParams, rho, optimizer: "admm" }),
    sgd: runTuningSolver(cloud, { ...baseParams, rho, optimizer: "sgd" }),
  }));

  printSummary(results);
  writeOutputs(results, options);
  console.log("");
  console.log(`wrote svg to ${options.output}`);
  console.log(`wrote csv to ${options.csv}`);
}

main();
