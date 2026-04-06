const TAU = Math.PI * 2;
const sceneCanvas = document.getElementById("scene");
const sceneCtx = sceneCanvas.getContext("2d");
const objectiveCanvas = document.getElementById("objectiveChart");
const objectiveCtx = objectiveCanvas.getContext("2d");
const residualCanvas = document.getElementById("residualChart");
const residualCtx = residualCanvas.getContext("2d");

const ui = {
  iterValue: document.getElementById("iterValue"),
  objectiveValue: document.getElementById("objectiveValue"),
  primalValue: document.getElementById("primalValue"),
  dualValue: document.getElementById("dualValue"),
  fxValue: document.getElementById("fxValue"),
  g1Value: document.getElementById("g1Value"),
  g2Value: document.getElementById("g2Value"),
  g3Value: document.getElementById("g3Value"),
  weightLabel: document.getElementById("weightLabel"),
  weightValue: document.getElementById("weightValue"),
  pathLabel: document.getElementById("pathLabel"),
  pathValue: document.getElementById("pathValue"),
  playPause: document.getElementById("playPause"),
  stepButton: document.getElementById("stepButton"),
  resetButton: document.getElementById("resetButton"),
  hubCountGroup: document.getElementById("hubCountGroup"),
  hubCount: document.getElementById("hubCount"),
  hubCountValue: document.getElementById("hubCountValue"),
  rho: document.getElementById("rho"),
  rhoValue: document.getElementById("rhoValue"),
  lambda: document.getElementById("lambda"),
  lambdaValue: document.getElementById("lambdaValue"),
  mu: document.getElementById("mu"),
  muValue: document.getElementById("muValue"),
  explorerTab: document.getElementById("explorerTab"),
  tuningTab: document.getElementById("tuningTab"),
  modeSummary: document.getElementById("modeSummary"),
  zoomIn: document.getElementById("zoomIn"),
  zoomOut: document.getElementById("zoomOut"),
  panUp: document.getElementById("panUp"),
  panDown: document.getElementById("panDown"),
  panLeft: document.getElementById("panLeft"),
  panRight: document.getElementById("panRight"),
};

const modeCopy = {
  explorer:
    'The browser runs a 2D, deterministic ADMM-style solver over hub locations <code>x, z₁, z₂, z₃, u₁, u₂, u₃</code>. Each frame updates point assignments, shortest-path regularisation, quadratic shrinkage, and graph total variation on a local kNN graph.',
  tuning:
    "This tab reuses the explorer's local kNN-graph ADMM solver at each fixed hub count. It starts from 3 KMeans hubs, settles the current geometry with the same ADMM updates as the explorer, then splits the highest-energy Voronoi cell and keeps the new hub only when that locally converged split lowers the loss.",
};

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

function kMeans(points, k, iterations = 12, seed = 91, initialCenters = null) {
  if (points.length === 0 || k <= 0) {
    return { centers: [], assignments: [] };
  }
  const centerCount = Math.min(k, points.length);
  let centers = initialCenters ? clonePoints(initialCenters.slice(0, centerCount)) : sampleInitialHubs(points, centerCount, seed);
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

function pathContainsEdge(path, a, b) {
  const key = edgeKey(a, b);
  for (let i = 0; i < path.length - 1; i += 1) {
    if (edgeKey(path[i], path[i + 1]) === key) {
      return true;
    }
  }
  return false;
}

function pathSquaredLength(points, path) {
  let total = 0;
  for (let i = 0; i < path.length - 1; i += 1) {
    total += squaredDist(points[path[i]], points[path[i + 1]]);
  }
  return total;
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
    nextX.push(scale(add(sum, scale(consensus, state.rho)), 1 / Math.max(denom, 1e-6)));
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
          grad[a] = add(grad[a], scale(unit, coeff));
          grad[b] = add(grad[b], scale(unit, -coeff));
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
      grad[a] = add(grad[a], scale(unit, state.mu));
      grad[b] = add(grad[b], scale(unit, -state.mu));
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

function computeMotionStats(current, previous) {
  let total = 0;
  let max = 0;
  for (let i = 0; i < current.length; i += 1) {
    const motion = dist(current[i], previous[i]);
    total += motion;
    if (motion > max) {
      max = motion;
    }
  }
  return {
    total,
    avg: current.length ? total / current.length : 0,
    max,
  };
}

function evaluateObjective(hubs, edges, cloud, params) {
  const assigned = assignPoints(hubs, cloud);
  const graph = graphFromEdges(hubs, edges);
  const fx = computeFx(hubs, assigned.assignments);
  const g1Info = computePathSmoothness(hubs, assigned.weights, graph);
  const g2 = computeG2(hubs, params.lambda);
  const g3 = computeGraphVariation(hubs, graph.edges, params.mu);
  return {
    objective: fx + g1Info.value + g2 + g3,
    fx,
    g1: g1Info.value,
    g2,
    g3,
    assignments: assigned.assignments,
    weights: assigned.weights,
    highlightedPath: g1Info.highlighted.path,
    highlightedPair: g1Info.highlighted.pair,
    graph,
  };
}

function syncStateParams(state) {
  state.rho = Number(ui.rho.value);
  state.lambda = Number(ui.lambda.value);
  state.mu = Number(ui.mu.value);
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

function createBaseState(mode, cloud, x, edges) {
  const state = {
    mode,
    cloud,
    playing: true,
    iteration: 0,
    lambda: Number(ui.lambda.value),
    mu: Number(ui.mu.value),
    rho: Number(ui.rho.value),
    history: [],
    lastAction: "Initialised",
  };
  resetAdmmVariables(state, x, edges);
  return state;
}

function createExplorerState() {
  const cloud = seededPointCloud();
  const hubCount = Number(ui.hubCount.value);
  const x = sampleInitialHubs(cloud, hubCount);
  return createBaseState("explorer", cloud, x, []);
}

function createTuningState() {
  const cloud = seededPointCloud();
  const startK = 3;
  const clustering = kMeans(cloud, startK, 16, 111);
  const graph = buildMaxGraph(clustering.centers);
  const state = createBaseState("tuning", cloud, clustering.centers, graph.edges);
  state.outerIteration = 0;
  state.lastAction = `Start with ${startK} hubs from KMeans`;
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
    maxWeight: Math.max(...metrics.weights, 0),
    hubCount: state.x.length,
    edgeCount: state.edges.length,
  });
  if (state.history.length > 180) {
    state.history.shift();
  }
}

function runAdmmIteration(state, graphMode = "knn", pushHistory = true) {
  syncStateParams(state);
  const previousZ1 = clonePoints(state.z1);
  const previousZ2 = clonePoints(state.z2);
  const previousZ3 = clonePoints(state.z3);
  const assignmentBefore = assignPoints(state.x, state.cloud);
  state.weights = assignmentBefore.weights;

  state.x = proxX(state);

  const graph = graphMode === "fixed" ? graphFromEdges(state.x, state.edges) : buildKnnGraph(state.x, 2);
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

function runFixedKConvergence(state, options = {}) {
  const {
    maxIterations = 48,
    minIterations = 6,
    avgTolerance = 0.003,
    maxTolerance = 0.0075,
    pushHistory = true,
    graphMode = "knn",
  } = options;

  let latest = null;
  let motion = { total: Infinity, avg: Infinity, max: Infinity };

  for (let iteration = 0; iteration < maxIterations; iteration += 1) {
    const previousX = clonePoints(state.x);
    latest = runAdmmIteration(state, graphMode, pushHistory);
    motion = computeMotionStats(state.x, previousX);
    if (iteration + 1 >= minIterations && motion.avg <= avgTolerance && motion.max <= maxTolerance) {
      return { ...latest, motion, settled: true, iterations: iteration + 1 };
    }
  }

  return { ...latest, motion, settled: false, iterations: maxIterations };
}

function simulateFixedHubRefinement(baseState, x, edges) {
  const state = createBaseState("simulation", baseState.cloud, x, edges);
  state.lambda = baseState.lambda;
  state.mu = baseState.mu;
  state.rho = baseState.rho;
  const convergence = runFixedKConvergence(state, { pushHistory: false, graphMode: "knn" });
  return { state, metrics: convergence.metrics, convergence };
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
        best = {
          delta,
          clusterIndex,
          clusterEnergy: bestEnergy,
          candidate,
          anchorIndex,
          simulation,
        };
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
  target.lastAction = lastAction;
}

function stepExplorer() {
  runAdmmIteration(explorerState, "knn", true);
}

function stepTuning() {
  syncStateParams(tuningState);
  let convergence = runFixedKConvergence(tuningState, { graphMode: "knn" });

  tuningState.outerIteration += 1;

  const splitProposal = buildSplitProposal(tuningState);
  if (splitProposal && splitProposal.delta > 0) {
    adoptSimulationState(
      tuningState,
      splitProposal.simulation,
      `Added hub from Voronoi split ${splitProposal.clusterIndex} via ${splitProposal.anchorIndex} (ΔF=${splitProposal.delta.toFixed(3)})`,
    );
    convergence = runFixedKConvergence(tuningState, { graphMode: "knn" });
    tuningState.lastAction = `${tuningState.lastAction}; settled in ${convergence.iterations} ADMM sweeps`;
  } else {
    const settleLabel = convergence.settled ? "settled" : "hit the ADMM sweep cap";
    tuningState.lastAction = `No improving hub split after ${settleLabel} at outer step ${tuningState.outerIteration}; converged`;
    tuningState.playing = false;
  }
}

const camera = {
  zoom: 1,
  offsetX: 0,
  offsetY: 0,
};

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function projectPoint(point, bounds, size, view = camera) {
  const pad = 46;
  const sx = (size.width - pad * 2) / (bounds.maxX - bounds.minX);
  const sy = (size.height - pad * 2) / (bounds.maxY - bounds.minY);
  const scaleValue = Math.min(sx, sy);
  const baseX = pad + (point.x - bounds.minX) * scaleValue;
  const baseY = size.height - pad - (point.y - bounds.minY) * scaleValue;
  const centerX = size.width * 0.5;
  const centerY = size.height * 0.5;
  return {
    x: centerX + (baseX - centerX) * view.zoom + view.offsetX,
    y: centerY + (baseY - centerY) * view.zoom + view.offsetY,
  };
}

function computeBounds(cloud, hubs) {
  const all = cloud.concat(hubs);
  const xs = all.map((point) => point.x);
  const ys = all.map((point) => point.y);
  return {
    minX: Math.min(...xs) - 0.3,
    maxX: Math.max(...xs) + 0.3,
    minY: Math.min(...ys) - 0.3,
    maxY: Math.max(...ys) + 0.3,
  };
}

function scaleVisualSize(base, zoom, min, max, exponent = 0.55) {
  return clamp(base * zoom ** exponent, min, max);
}

function drawScene(state) {
  const { width, height } = sceneCanvas;
  sceneCtx.clearRect(0, 0, width, height);

  const bounds = computeBounds(state.cloud, state.x);
  const { assignments } = assignPoints(state.x, state.cloud);

  sceneCtx.fillStyle = "#fff8ef";
  sceneCtx.fillRect(0, 0, width, height);

  sceneCtx.strokeStyle = "rgba(20, 16, 13, 0.08)";
  sceneCtx.lineWidth = scaleVisualSize(1, camera.zoom, 0.75, 2.2, 0.2);
  for (let i = 0; i < 9; i += 1) {
    const t = i / 8;
    const y = 40 + t * (height - 80);
    const x = 40 + t * (width - 80);
    sceneCtx.beginPath();
    sceneCtx.moveTo(34, y);
    sceneCtx.lineTo(width - 34, y);
    sceneCtx.stroke();
    sceneCtx.beginPath();
    sceneCtx.moveTo(x, 34);
    sceneCtx.lineTo(x, height - 34);
    sceneCtx.stroke();
  }

  for (let hubIndex = 0; hubIndex < assignments.length; hubIndex += 1) {
    const color = `hsla(${(hubIndex * 53) % 360} 70% 52% / 0.12)`;
    sceneCtx.fillStyle = color;
    for (const point of assignments[hubIndex]) {
      const projected = projectPoint(point, bounds, { width, height });
      sceneCtx.beginPath();
      sceneCtx.arc(projected.x, projected.y, scaleVisualSize(2.4, camera.zoom, 1.5, 4.4, 0.45), 0, TAU);
      sceneCtx.fill();
    }
  }

  sceneCtx.strokeStyle = "rgba(35, 117, 111, 0.42)";
  sceneCtx.lineWidth = scaleVisualSize(1.3, camera.zoom, 1, 3.4, 0.35);
  for (const [a, b] of state.edges) {
    const pa = projectPoint(state.x[a], bounds, { width, height });
    const pb = projectPoint(state.x[b], bounds, { width, height });
    sceneCtx.beginPath();
    sceneCtx.moveTo(pa.x, pa.y);
    sceneCtx.lineTo(pb.x, pb.y);
    sceneCtx.stroke();
  }

  if (state.highlightedPath.length > 1) {
    sceneCtx.strokeStyle = "#d77b2a";
    sceneCtx.lineWidth = scaleVisualSize(4, camera.zoom, 2.4, 7.2, 0.4);
    sceneCtx.lineJoin = "round";
    sceneCtx.beginPath();
    const start = projectPoint(state.x[state.highlightedPath[0]], bounds, { width, height });
    sceneCtx.moveTo(start.x, start.y);
    for (let i = 1; i < state.highlightedPath.length; i += 1) {
      const point = projectPoint(state.x[state.highlightedPath[i]], bounds, { width, height });
      sceneCtx.lineTo(point.x, point.y);
    }
    sceneCtx.stroke();
  }

  for (let i = 0; i < state.x.length; i += 1) {
    const projected = projectPoint(state.x[i], bounds, { width, height });
    const weight = state.weights[i] ?? 0;
    const hubRadius = scaleVisualSize(6.5 + Math.min(weight / 24, 8), camera.zoom, 4.5, 19, 0.55);
    sceneCtx.fillStyle = "#10100f";
    sceneCtx.beginPath();
    sceneCtx.arc(projected.x, projected.y, hubRadius, 0, TAU);
    sceneCtx.fill();
    sceneCtx.fillStyle = "#fff3e0";
    sceneCtx.font = `${Math.round(scaleVisualSize(12, camera.zoom, 9, 18, 0.35))}px IBM Plex Mono`;
    sceneCtx.textAlign = "center";
    sceneCtx.fillText(String(i), projected.x, projected.y + scaleVisualSize(4, camera.zoom, 3, 6, 0.3));
  }
}

function drawChart(ctx, canvas, seriesList, colors, labels) {
  const { width, height } = canvas;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#fff7ed";
  ctx.fillRect(0, 0, width, height);

  const pad = 26;
  const values = seriesList.flat();
  const min = Math.min(...values, 0);
  const max = Math.max(...values, 1);
  const span = Math.max(max - min, 1e-6);

  ctx.strokeStyle = "rgba(24, 22, 18, 0.08)";
  ctx.lineWidth = 1;
  for (let i = 0; i < 5; i += 1) {
    const y = pad + (i / 4) * (height - pad * 2);
    ctx.beginPath();
    ctx.moveTo(pad, y);
    ctx.lineTo(width - pad, y);
    ctx.stroke();
  }

  seriesList.forEach((series, index) => {
    ctx.strokeStyle = colors[index];
    ctx.lineWidth = 2.4;
    ctx.beginPath();
    series.forEach((value, i) => {
      const x = pad + (i / Math.max(series.length - 1, 1)) * (width - pad * 2);
      const y = height - pad - ((value - min) / span) * (height - pad * 2);
      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });
    ctx.stroke();
  });

  labels.forEach((label, index) => {
    ctx.fillStyle = colors[index];
    ctx.font = "12px IBM Plex Mono";
    ctx.fillText(label, 18 + index * 112, 18);
  });
}

let activeMode = "explorer";
let explorerState = createExplorerState();
let tuningState = createTuningState();

function getActiveState() {
  return activeMode === "explorer" ? explorerState : tuningState;
}

function updateUi() {
  const state = getActiveState();
  const current = state.history[state.history.length - 1] || {
    objective: 0,
    primal: 0,
    dual: 0,
    fx: 0,
    g1: 0,
    g2: 0,
    g3: 0,
    maxWeight: 0,
    hubCount: state.x.length,
    edgeCount: state.edges.length,
  };

  ui.iterValue.textContent = String(state.mode === "tuning" ? state.outerIteration : state.iteration);
  ui.objectiveValue.textContent = current.objective.toFixed(3);
  ui.primalValue.textContent = current.primal.toFixed(3);
  ui.dualValue.textContent = current.dual.toFixed(3);
  ui.fxValue.textContent = current.fx.toFixed(3);
  ui.g1Value.textContent = current.g1.toFixed(3);
  ui.g2Value.textContent = current.g2.toFixed(3);
  ui.g3Value.textContent = current.g3.toFixed(3);
  ui.playPause.textContent = state.playing ? "Pause" : "Play";
  ui.hubCountValue.textContent = ui.hubCount.value;
  ui.rhoValue.textContent = Number(ui.rho.value).toFixed(2);
  ui.lambdaValue.textContent = Number(ui.lambda.value).toFixed(2);
  ui.muValue.textContent = Number(ui.mu.value).toFixed(2);
  ui.modeSummary.innerHTML = modeCopy[activeMode];
  ui.explorerTab.classList.toggle("is-active", activeMode === "explorer");
  ui.tuningTab.classList.toggle("is-active", activeMode === "tuning");
  ui.hubCountGroup.classList.toggle("is-hidden", activeMode === "tuning");

  if (state.mode === "tuning") {
    ui.weightLabel.textContent = "Hub / edge count";
    ui.weightValue.textContent = `${current.hubCount} / ${current.edgeCount}`;
    ui.pathLabel.textContent = "Last tuning action";
    ui.pathValue.textContent = state.lastAction;
  } else {
    ui.weightLabel.textContent = "Largest hub weight";
    ui.weightValue.textContent = String(current.maxWeight);
    ui.pathLabel.textContent = "Highlighted path";
    ui.pathValue.textContent = `${state.highlightedPair[0]} → ${state.highlightedPair[1]}`;
  }

  drawScene(state);
  drawChart(
    objectiveCtx,
    objectiveCanvas,
    [state.history.map((row) => row.objective), state.history.map((row) => row.fx)],
    ["#cb5e38", "#23756f"],
    ["objective", "f(x)"],
  );
  drawChart(
    residualCtx,
    residualCanvas,
    [state.history.map((row) => row.primal), state.history.map((row) => row.dual)],
    ["#d77b2a", "#3f5d53"],
    ["primal", "dual"],
  );
}

function resetState(mode = activeMode) {
  if (mode === "explorer") {
    explorerState = createExplorerState();
    for (let i = 0; i < 3; i += 1) {
      stepExplorer();
    }
  } else {
    tuningState = createTuningState();
    for (let i = 0; i < 2; i += 1) {
      stepTuning();
    }
  }
  updateUi();
}

function switchMode(mode) {
  activeMode = mode;
  if (getActiveState().history.length === 0) {
    resetState(mode);
    return;
  }
  updateUi();
}

function nudgeCamera(dx = 0, dy = 0) {
  camera.offsetX += dx;
  camera.offsetY += dy;
  updateUi();
}

function changeZoom(factor) {
  camera.zoom = clamp(camera.zoom * factor, 0.65, 3.5);
  updateUi();
}

ui.playPause.addEventListener("click", () => {
  const state = getActiveState();
  state.playing = !state.playing;
  updateUi();
});

ui.stepButton.addEventListener("click", () => {
  const state = getActiveState();
  state.playing = false;
  if (activeMode === "explorer") {
    stepExplorer();
  } else {
    stepTuning();
  }
  updateUi();
});

ui.resetButton.addEventListener("click", () => resetState(activeMode));
ui.explorerTab.addEventListener("click", () => switchMode("explorer"));
ui.tuningTab.addEventListener("click", () => switchMode("tuning"));
ui.zoomIn.addEventListener("click", () => changeZoom(1.2));
ui.zoomOut.addEventListener("click", () => changeZoom(1 / 1.2));
ui.panUp.addEventListener("click", () => nudgeCamera(0, -34));
ui.panDown.addEventListener("click", () => nudgeCamera(0, 34));
ui.panLeft.addEventListener("click", () => nudgeCamera(-34, 0));
ui.panRight.addEventListener("click", () => nudgeCamera(34, 0));

for (const slider of [ui.hubCount, ui.rho, ui.lambda, ui.mu]) {
  slider.addEventListener("input", () => {
    ui.hubCountValue.textContent = ui.hubCount.value;
    ui.rhoValue.textContent = Number(ui.rho.value).toFixed(2);
    ui.lambdaValue.textContent = Number(ui.lambda.value).toFixed(2);
    ui.muValue.textContent = Number(ui.mu.value).toFixed(2);
  });
}

ui.hubCount.addEventListener("change", () => {
  explorerState = createExplorerState();
  tuningState = createTuningState();
  resetState(activeMode);
});

let lastTick = 0;
function animate(timestamp) {
  if (timestamp - lastTick > 150) {
    const state = getActiveState();
    if (state.playing) {
      if (activeMode === "explorer") {
        stepExplorer();
      } else {
        stepTuning();
      }
    }
    updateUi();
    lastTick = timestamp;
  }
  requestAnimationFrame(animate);
}

resetState("explorer");
requestAnimationFrame(animate);
