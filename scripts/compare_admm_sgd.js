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
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return Math.hypot(dx, dy);
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
  if (n < 1e-9) {
    return vec(0, 0);
  }
  return scale(a, 1 / n);
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

  for (let i = 0; i < 170; i += 1) {
    const t = rng() * 4.2 - 2.1;
    points.push(vec(0.98 * t + randomNormal(rng) * 0.08, 0.5 * t + randomNormal(rng) * 0.09));
  }
  for (let i = 0; i < 170; i += 1) {
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

function sampleInitialHubs(points, count, seed = 77) {
  return farthestPointSampling(points, count, seed).map((point, index) =>
    add(point, vec(((index % 3) - 1) * 0.03, (((index + 1) % 3) - 1) * 0.03)),
  );
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

function edgeKey(a, b) {
  const lo = Math.min(a, b);
  const hi = Math.max(a, b);
  return `${lo}-${hi}`;
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
      if (i === j) {
        continue;
      }
      neighbors.push({ j, d: dist(points[i], points[j]) });
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
    const hub = hubs[i];
    for (const point of assignments[i]) {
      total += 0.5 * squaredDist(point, hub);
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
    }
  }

  return { value: smoothness };
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
    graph,
  };
}

function proxX(state) {
  const vPoints = state.x.map((_, i) =>
    scale(
      add(add(sub(state.z1[i], state.u1[i]), sub(state.z2[i], state.u2[i])), sub(state.z3[i], state.u3[i])),
      1 / 3,
    ),
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

  for (let step = 0; step < 7; step += 1) {
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
          const delta = sub(z[a], z[b]);
          const unit = normalize(delta);
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

  for (let step = 0; step < 7; step += 1) {
    const grad = z.map(() => vec(0, 0));
    for (const [a, b] of edges) {
      const delta = sub(z[a], z[b]);
      const unit = normalize(delta);
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

function createAdmmState(cloud, x, params) {
  const graph = buildKnnGraph(x, 2);
  return {
    cloud,
    x: clonePoints(x),
    z1: clonePoints(x),
    z2: clonePoints(x),
    z3: clonePoints(x),
    u1: x.map(() => vec(0, 0)),
    u2: x.map(() => vec(0, 0)),
    u3: x.map(() => vec(0, 0)),
    edges: graph.edges,
    lambda: params.lambda,
    mu: params.mu,
    rho: params.rho,
    stationCost: params.stationCost,
  };
}

function runAdmmIteration(state, fixedEdges) {
  const previousZ1 = clonePoints(state.z1);
  const previousZ2 = clonePoints(state.z2);
  const previousZ3 = clonePoints(state.z3);
  const assignmentBefore = assignPoints(state.x, state.cloud);

  state.x = proxX(state);
  const graph = graphFromEdges(state.x, fixedEdges);
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

  return { metrics, primal, dual };
}

function runAdmm(cloud, initialHubs, params) {
  const state = createAdmmState(cloud, initialHubs, params);
  const fixedEdges = buildKnnGraph(state.x, 2).edges;
  state.edges = fixedEdges;
  let latest = evaluateObjective(state.x, state.edges, state.cloud, state);
  let primal = Infinity;
  let dual = Infinity;
  let iterations = 0;

  for (let i = 0; i < params.admmMaxIterations; i += 1) {
    const step = runAdmmIteration(state, fixedEdges);
    latest = step.metrics;
    primal = step.primal;
    dual = step.dual;
    iterations = i + 1;
    const thresholds = computeResidualThresholds(state);
    if (iterations >= params.admmMinIterations && primal <= thresholds.primal && dual <= thresholds.dual) {
      break;
    }
  }

  return {
    solver: "ADMM",
    hubs: state.x,
    edges: state.edges,
    metrics: latest,
    iterations,
    primal,
    dual,
  };
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

function computeSgdGradient(x, cloud, params, rng) {
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

  const graph = buildKnnGraph(x, 2);
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

function runSgd(cloud, initialHubs, params) {
  const rng = mulberry32(params.sgdSeed);
  let x = clonePoints(initialHubs);
  let best = null;
  let latest = null;

  for (let iteration = 0; iteration < params.sgdIterations; iteration += 1) {
    const { grad } = computeSgdGradient(x, cloud, params, rng);
    const lr = params.sgdLearningRate / Math.sqrt(1 + iteration * params.sgdDecay);
    x = x.map((point, index) => sub(point, scale(grad[index], lr)));
    latest = evaluateObjective(x, buildKnnGraph(x, 2).edges, cloud, params);
    if (!best || latest.objective < best.metrics.objective) {
      best = {
        solver: "SGD",
        hubs: clonePoints(x),
        edges: latest.graph.edges,
        metrics: latest,
        iterations: iteration + 1,
      };
    }
  }

  return best ?? {
    solver: "SGD",
    hubs: clonePoints(x),
    edges: buildKnnGraph(x, 2).edges,
    metrics: evaluateObjective(x, buildKnnGraph(x, 2).edges, cloud, params),
    iterations: 0,
  };
}

function formatNumber(value) {
  if (!Number.isFinite(value)) {
    return String(value);
  }
  return value.toFixed(6);
}

function printTable(rows) {
  const headers = ["Metric", "ADMM", "SGD", "SGD-ADMM"];
  const stringRows = rows.map(([metric, admm, sgd, delta]) => [metric, admm, sgd, delta].map(String));
  const widths = headers.map((header, index) =>
    Math.max(header.length, ...stringRows.map((row) => row[index].length)),
  );

  const divider = `|-${widths.map((width) => "-".repeat(width)).join("-|-")}-|`;
  const renderRow = (cells) => `| ${cells.map((cell, index) => cell.padEnd(widths[index])).join(" | ")} |`;

  console.log(renderRow(headers));
  console.log(divider);
  for (const row of stringRows) {
    console.log(renderRow(row));
  }
}

function compareResults(admm, sgd) {
  const rows = [
    ["objective", formatNumber(admm.metrics.objective), formatNumber(sgd.metrics.objective), formatNumber(sgd.metrics.objective - admm.metrics.objective)],
    ["fx", formatNumber(admm.metrics.fx), formatNumber(sgd.metrics.fx), formatNumber(sgd.metrics.fx - admm.metrics.fx)],
    ["g1", formatNumber(admm.metrics.g1), formatNumber(sgd.metrics.g1), formatNumber(sgd.metrics.g1 - admm.metrics.g1)],
    ["g2", formatNumber(admm.metrics.g2), formatNumber(sgd.metrics.g2), formatNumber(sgd.metrics.g2 - admm.metrics.g2)],
    ["g3", formatNumber(admm.metrics.g3), formatNumber(sgd.metrics.g3), formatNumber(sgd.metrics.g3 - admm.metrics.g3)],
    ["g4", formatNumber(admm.metrics.g4), formatNumber(sgd.metrics.g4), formatNumber(sgd.metrics.g4 - admm.metrics.g4)],
    ["iterations", admm.iterations, sgd.iterations, sgd.iterations - admm.iterations],
    ["edgeCount", admm.edges.length, sgd.edges.length, sgd.edges.length - admm.edges.length],
    ["hubCount", admm.hubs.length, sgd.hubs.length, sgd.hubs.length - admm.hubs.length],
    ["primalResidual", formatNumber(admm.primal), "n/a", "n/a"],
    ["dualResidual", formatNumber(admm.dual), "n/a", "n/a"],
  ];

  printTable(rows);
}

function parseArgs(argv) {
  const options = {
    hubCounts: null,
    output: path.join(process.cwd(), "out", "compare_admm_sgd_objective_diff_vs_hubs.svg"),
  };

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === "--hub-counts" && i + 1 < argv.length) {
      options.hubCounts = argv[i + 1]
        .split(",")
        .map((value) => Number.parseInt(value.trim(), 10))
        .filter((value) => Number.isFinite(value) && value > 1);
      i += 1;
      continue;
    }
    if (arg === "--output" && i + 1 < argv.length) {
      options.output = path.resolve(argv[i + 1]);
      i += 1;
    }
  }

  if (!options.hubCounts || options.hubCounts.length === 0) {
    options.hubCounts = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60];
  }

  options.hubCounts = [...new Set(options.hubCounts)].sort((a, b) => a - b);
  return options;
}

function escapeXml(text) {
  return String(text)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function buildObjectiveDiffSvg(results, outputFile) {
  const width = 1080;
  const height = 720;
  const margin = { top: 80, right: 60, bottom: 90, left: 110 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  const hubs = results.map((result) => result.hubCount);
  const diffs = results.map((result) => result.objectiveDiff);
  const minHub = Math.min(...hubs);
  const maxHub = Math.max(...hubs);
  const rawMinDiff = Math.min(...diffs, 0);
  const rawMaxDiff = Math.max(...diffs, 0);
  const diffSpan = Math.max(rawMaxDiff - rawMinDiff, 1e-6);
  const pad = diffSpan * 0.1;
  const minDiff = rawMinDiff - pad;
  const maxDiff = rawMaxDiff + pad;

  function xScale(value) {
    if (maxHub === minHub) {
      return margin.left + plotWidth / 2;
    }
    return margin.left + ((value - minHub) / (maxHub - minHub)) * plotWidth;
  }

  function yScale(value) {
    return margin.top + ((maxDiff - value) / (maxDiff - minDiff)) * plotHeight;
  }

  const xTicks = hubs;
  const yTicks = [];
  const yTickCount = 6;
  for (let i = 0; i <= yTickCount; i += 1) {
    yTicks.push(minDiff + ((maxDiff - minDiff) * i) / yTickCount);
  }

  const linePath = results
    .map((result, index) => `${index === 0 ? "M" : "L"} ${xScale(result.hubCount).toFixed(2)} ${yScale(result.objectiveDiff).toFixed(2)}`)
    .join(" ");

  const zeroY = yScale(0);
  const pointLabels = results
    .map((result) => {
      const x = xScale(result.hubCount);
      const y = yScale(result.objectiveDiff);
      return `
  <circle cx="${x.toFixed(2)}" cy="${y.toFixed(2)}" r="5.5" fill="#d9480f" stroke="#fff7ed" stroke-width="2" />
  <text x="${x.toFixed(2)}" y="${(y - 12).toFixed(2)}" text-anchor="middle" font-size="14" fill="#7c2d12">${escapeXml(formatNumber(result.objectiveDiff))}</text>`;
    })
    .join("");

  const xGrid = xTicks
    .map((tick) => {
      const x = xScale(tick);
      return `
  <line x1="${x.toFixed(2)}" y1="${margin.top}" x2="${x.toFixed(2)}" y2="${(height - margin.bottom).toFixed(2)}" stroke="#e2e8f0" stroke-width="1" />
  <text x="${x.toFixed(2)}" y="${(height - margin.bottom + 30).toFixed(2)}" text-anchor="middle" font-size="15" fill="#334155">${tick}</text>`;
    })
    .join("");

  const yGrid = yTicks
    .map((tick) => {
      const y = yScale(tick);
      return `
  <line x1="${margin.left}" y1="${y.toFixed(2)}" x2="${(width - margin.right).toFixed(2)}" y2="${y.toFixed(2)}" stroke="#e2e8f0" stroke-width="1" />
  <text x="${(margin.left - 16).toFixed(2)}" y="${(y + 5).toFixed(2)}" text-anchor="end" font-size="15" fill="#334155">${escapeXml(formatNumber(tick))}</text>`;
    })
    .join("");

  return `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" role="img" aria-labelledby="title desc">
  <title id="title">Objective difference versus number of hubs</title>
  <desc id="desc">Line plot of SGD objective minus ADMM objective for different hub counts.</desc>
  <rect width="${width}" height="${height}" fill="#fffaf0" />
  <rect x="${margin.left}" y="${margin.top}" width="${plotWidth}" height="${plotHeight}" rx="18" fill="#ffffff" stroke="#fed7aa" stroke-width="1.5" />
  ${xGrid}
  ${yGrid}
  <line x1="${margin.left}" y1="${zeroY.toFixed(2)}" x2="${(width - margin.right).toFixed(2)}" y2="${zeroY.toFixed(2)}" stroke="#9a3412" stroke-width="1.5" stroke-dasharray="8 6" />
  <path d="${linePath}" fill="none" stroke="#ea580c" stroke-width="4" stroke-linecap="round" stroke-linejoin="round" />
  ${pointLabels}
  <text x="${(width / 2).toFixed(2)}" y="42" text-anchor="middle" font-size="28" font-weight="700" fill="#7c2d12">SGD Objective - ADMM Objective vs Number of Hubs</text>
  <text x="${(width / 2).toFixed(2)}" y="70" text-anchor="middle" font-size="16" fill="#9a3412">Positive values mean ADMM reached a lower objective.</text>
  <text x="${(width / 2).toFixed(2)}" y="${(height - 24).toFixed(2)}" text-anchor="middle" font-size="18" fill="#431407">Number of hubs</text>
  <text x="28" y="${(height / 2).toFixed(2)}" transform="rotate(-90 28 ${(height / 2).toFixed(2)})" text-anchor="middle" font-size="18" fill="#431407">Objective difference</text>
</svg>`;
}

function writeObjectiveDiffPlot(results, outputFile) {
  fs.mkdirSync(path.dirname(outputFile), { recursive: true });
  fs.writeFileSync(outputFile, buildObjectiveDiffSvg(results, outputFile));
}

function runComparisonForHubCount(cloud, hubCount, params) {
  const runParams = { ...params, hubCount };
  const initialHubs = sampleInitialHubs(cloud, hubCount, 77);
  const admm = runAdmm(cloud, initialHubs, runParams);
  const sgd = runSgd(cloud, initialHubs, runParams);
  return {
    hubCount,
    admm,
    sgd,
    objectiveDiff: sgd.metrics.objective - admm.metrics.objective,
  };
}

function main() {
  const options = parseArgs(process.argv.slice(2));
  const params = {
    hubCount: 5,
    rho: 1.15,
    lambda: 0.12,
    mu: 0.26,
    stationCost: 0,
    admmMaxIterations: 100,
    admmMinIterations: 6,
    sgdIterations: 400,
    sgdLearningRate: 0.003,
    sgdDecay: 0.04,
    sgdBatchSize: 160,
    sgdPairSamples: 12,
    sgdSeed: 20260426,
  };

  const cloud = seededPointCloud(12);
  const comparisons = options.hubCounts.map((hubCount) => runComparisonForHubCount(cloud, hubCount, params));
  const selected = comparisons.find((comparison) => comparison.hubCount === params.hubCount) ?? comparisons[0];

  console.log("app.js explorer comparison with an SGD minimization path");
  console.log(`cloudPoints=${cloud.length} hubCounts=${options.hubCounts.join(",")} seed=12`);
  console.log("");

  console.log(`detailed comparison for hubCount=${selected.hubCount}`);
  compareResults(selected.admm, selected.sgd);
  console.log("");

  const summaryRows = comparisons.map((comparison) => [
    comparison.hubCount,
    formatNumber(comparison.admm.metrics.objective),
    formatNumber(comparison.sgd.metrics.objective),
    formatNumber(comparison.objectiveDiff),
  ]);
  printTable(summaryRows.map(([hubCount, admmObj, sgdObj, diff]) => [`hubCount=${hubCount}`, admmObj, sgdObj, diff]));

  writeObjectiveDiffPlot(comparisons, options.output);
  console.log("");
  console.log(`wrote plot to ${options.output}`);
}

main();
