# Multiple Penalized Principal Curves vs `app.js` ADMM

Source paper: https://www.math.cmu.edu/users/slepcev/mppc.pdf

Compared implementation: `/home/porkboi/Documents/pointy-clouds/app.js`

## Overview

The MPPC paper studies a variational curve-fitting problem for one-dimensional structure discovery in point clouds. Its main objective combines squared distance-to-curve fidelity with a length penalty and, for the multi-curve version, an explicit penalty on the number of connected components. The paper then solves the fixed-assignment subproblem with an EM-style outer loop and an inner ADMM solve on a chain-structured fused-lasso problem.

`app.js` is not a direct implementation of the paper’s MPPC/PPC objective or its ADMM subproblem. It is a related ADMM-style hub-placement method that mixes k-means-like fidelity, a shortest-path penalty on a dynamically rebuilt kNN graph, quadratic shrinkage toward the origin, and graph total variation. It borrows the high-level split-variable idea and some k-means/splitting heuristics, but the mathematical problem and the numerical method are materially different.

## Problem Setup

- Goal: compare the MPPC paper’s objective and algorithm with the active browser solver in `app.js`.
- Inputs: a weighted point cloud in the paper; a synthetic 2D point cloud plus a hub count in `app.js`.
- Output: a piecewise linear curve or multiple curves in the paper; a set of hubs plus graph edges in `app.js`.

## Notation

| Symbol | Meaning |
| --- | --- |
| `mu` | Data measure in the paper |
| `gamma` | Curve or set of curves in the paper |
| `Gamma` | Image of the curve(s) |
| `lambda_1` | Length penalty in MPPC/PPC |
| `lambda_2` | Component-count penalty in MPPC |
| `I_j` | Data points assigned to node `y_j` in the paper’s discrete algorithm |
| `y_j` | Discrete curve nodes in the paper |
| `D` | First-difference operator along a chain in the paper |
| `x` | Hub positions in `app.js` |
| `z1, z2, z3` | Split variables in `app.js` |
| `u1, u2, u3` | Dual variables in `app.js` |
| `rho` | ADMM penalty parameter in both formulations |
| `lambda` | Origin-shrinkage weight in `app.js`, not MPPC `lambda_1`/`lambda_2` |
| `mu` | Graph-TV weight in `app.js`, not the data measure |

## Core Definitions

- **PPC functional**: `E^lambda_mu(gamma) = integral d(x, Gamma)^p dmu(x) + lambda L(gamma)`.
- **MPPC functional**: `E^(lambda_1,lambda_2)_mu(gamma) = integral d(x, Gamma)^p dmu(x) + lambda_1 (L(gamma) + lambda_2 (k(gamma)-1))`.
- **Paper’s discrete PPC energy**: assign each data point to its nearest node `y_j`, then minimize squared assignment error plus total variation `sum |y_(j+1)-y_j|`.
- **`app.js` energy**: `f(x) + g1(x) + g2(x) + g3(x)` where `f` is nearest-hub squared error, `g1` is all-pairs shortest-path cost on the hub graph, `g2` is quadratic norm penalty, and `g3` is graph total variation.

## Main Results From The Paper

- **Existence of minimizers**: both PPC and MPPC have minimizers under the stated assumptions.
- **Stationarity condition**: for smooth critical curves, curvature is balanced against projected data moments.
- **Linear stability / overfitting threshold**: a straight segment becomes unstable when `lambda_1 < 2 alpha H^2`, where `alpha` is projected linear density and `H^2` is mean squared projection distance.
- **Smoothing length scale**: the paper highlights `sqrt(lambda_1 / (2 alpha))` as the relevant local smoothing scale.
- **Critical density for splitting into components**: for the uniform-line example, `alpha* = (4/3)^2 lambda_1 / lambda_2^2`. Below this density, a continuous curve is no longer preferred over disconnected approximations.
- **Optimization insight**: enlarging the configuration space from one curve to multiple curves can reduce bad local minima by allowing cutting, reconnecting, and singleton management.

## Key Equations

1. `E_PPC(gamma) = integral d(x, Gamma)^p dmu(x) + lambda L(gamma)`: single-curve objective.
2. `E_MPPC(gamma) = integral d(x, Gamma)^p dmu(x) + lambda_1 (L(gamma) + lambda_2 (k(gamma)-1))`: multiple-curve objective.
3. `sum_j sum_{i in I_j} w_i |x_i - y_j|^2 + lambda_1 sum_j |y_(j+1)-y_j|`: discrete PPC objective with fixed assignments.
4. `min_{y,z: z=Dy} ||y||^2_{\bar w} - 2 (y,\bar x)_{\bar w} + lambda ||z||_{1,2}`: constrained form solved by ADMM in the paper.
5. `objective_app = fx + g1 + g2 + g3`: the implemented browser objective, where `g1` and `g3` are both graph-based path/TV penalties and `g2` is origin shrinkage.

## Derivation Flow

1. The paper starts from distance-to-curve fidelity plus curve complexity penalties.
2. It discretizes a curve by ordered nodes and nearest-node assignments.
3. With assignments fixed, the problem becomes convex and is written with a chain difference operator `D`.
4. ADMM splits the nonsmooth TV term `||Dy||_{1,2}` from the quadratic fidelity term.
5. MPPC then adds separate topology-changing routines to disconnect, reconnect, add, and remove components according to explicit energy changes.

## Method or Algorithm

1. Recompute nearest assignments from data to current discrete curve nodes.
2. Solve the fixed-assignment PPC subproblem with ADMM on a chain graph.
3. For MPPC, periodically run topology moves based on explicit energy change tests.
4. Re-parameterize and refine the node discretization where needed.

## Comparison With `app.js`

### What matches

- Both methods alternate between assignment-like updates and geometry updates.
- Both use split variables and dual variables with an ADMM-style structure.
- Both use k-means ideas for initialization and local splitting.
- Both try to escape poor local minima by allowing model-structure changes.

### What is theoretically different

- **Different object being fit**: the paper fits one or more piecewise linear curves with ordered nodes and explicit components. `app.js` fits an unordered set of hubs connected by a rebuilt kNN graph.
- **Different regularizer**: the paper’s core penalty is chain total variation `sum |y_(j+1)-y_j|`. `app.js` uses two graph penalties:
  - all-pairs weighted shortest-path cost on the current graph;
  - graph total variation over edges.
- **Missing MPPC component penalty**: the paper’s `lambda_1 lambda_2 (k-1)` term is central to topology control. `app.js` has no explicit component-count penalty and no singleton state.
- **Different ADMM subproblem**: the paper’s fixed-assignment problem is convex and has an exact cheap `y`-update via a tridiagonal linear solve plus exact block soft-thresholding for `z`. `app.js` uses heuristic inner gradient steps for `z1` and `z3`, and its graph itself changes during optimization.
- **Different projection model**: the paper assigns data to discrete curve nodes on an ordered curve. `app.js` assigns each point to its nearest hub, which is closer to vector quantization / k-means than to nearest-point-on-curve fitting.
- **Extra origin bias**: `app.js` adds `lambda * sum ||x_i||^2`, which pulls hubs toward the origin. The paper does not include this term.
- **Dynamic graph topology inside the local step**: the paper freezes a curve topology during the ADMM solve and changes topology with explicit energy routines. `app.js` rebuilds a 2-NN graph every local iteration, which changes the objective surface during optimization.

### Where `app.js` resembles the paper most closely

- The tuning mode’s split proposal is qualitatively similar to the paper’s use of k-means/singleton-based topology improvement: start from cluster centers, run local relaxation, then test whether a split lowers the objective.

### Where `app.js` departs most strongly

- The paper’s theory relies on an ordered chain or multiple chains and explicit component penalties. `app.js` has neither, so the paper’s stability, density-threshold, and resolution results do not transfer directly.

## Performance Implications

### Likely advantages of the paper’s method

- **Stronger theoretical grounding**: fixed-assignment subproblems are convex, and the ADMM updates are exact for that discretization.
- **Cheaper local linear algebra**: the paper’s `y`-update is tridiagonal on a chain, so one inner solve is `O(md)` after assignments are fixed.
- **More interpretable topology control**: splitting/connecting/singletons are tied to explicit energy tests involving `lambda_2`.
- **Better behavior on sparse or gappy data**: the component penalty and critical density logic help distinguish continuous curves from disconnected structure or noise.

### Likely advantages of `app.js`

- **Simpler interactive implementation**: no explicit curve ordering, reparameterization, or singleton bookkeeping.
- **Flexible graph geometry**: kNN rebuilding can adapt quickly when the intended structure is more graph-like than chain-like.
- **Reasonable browser responsiveness at small hub counts**: with a small number of hubs, the heuristic updates are cheap enough for animation.

### Likely performance disadvantages of `app.js`

- **Weaker convergence behavior**: because the graph changes during optimization and `z1`/`z3` are solved only approximately by a few gradient steps, the method is not solving the paper’s convex subproblem exactly and loses the ADMM guarantees associated with that setting.
- **Higher combinatorial overhead per iteration**: `computePathSmoothness` and `proxZ1` run repeated all-pairs shortest-path logic over the hub graph, which grows superlinearly in the number of hubs.
- **Possible double-counting of length-like penalties**: both `g1` and `g3` reward short edges/paths, which can over-compress the geometry compared with the paper’s single length term.
- **No explicit low-density component model**: the paper can represent sparse regions by disconnected curves or singletons; `app.js` may instead distort the hub graph or add hubs in an ad hoc way.
- **Origin attraction artifact**: the quadratic `g2` term can bias solutions toward the coordinate origin, which has no analogue in MPPC and can degrade geometric fidelity unless the cloud is centered appropriately.

### Likely scaling comparison

- For a chain with `m` curve nodes and `n` data points, the paper’s expensive step is usually reassignment, roughly `O(nm d)`, while the inner ADMM solve is structured and cheap.
- In `app.js`, assignment is also present, but repeated Dijkstra-based path computations in both objective evaluation and `z1` updates add substantial overhead. Since the graph is only kNN with small `k`, this is manageable for small `m`, but it will likely scale worse than the paper’s chain-based local solve as hub count grows.
- The paper explicitly reports viability for large point clouds and higher dimensions because the core local solve exploits chain structure. `app.js` is better viewed as a small-scale interactive approximation than as the same algorithmic class.

## Assumptions and Limits

- This comparison uses the active browser implementation in `app.js`, not the separate Python demo solver.
- I did not run a numerical benchmark against the paper’s reference implementation, so the performance discussion is theoretical and code-structure-based rather than measured.
- The paper’s formulas are summarized from the PDF text extraction and visible OCR; notation around some displayed equations is slightly compressed by PDF extraction.

## Uncertainties

- The paper’s exact implementation details around reparameterization and some topology heuristics are summarized rather than reproduced line-by-line.
- `app.js` does not document a formal optimization derivation for `g1` and `g3`, so part of the comparison is an inference from the implemented objective and update rules.

## Bottom Line

`app.js` is inspired by the same general problem family as MPPC, but it is not an implementation of the paper’s PPC/MPPC algorithm. The main differences are:

1. chain-structured curve fitting in the paper vs unordered hubs on a dynamic kNN graph in `app.js`;
2. exact convex ADMM subproblem in the paper vs approximate heuristic inner proximal-gradient steps in `app.js`;
3. explicit component penalty and singleton machinery in the paper vs no explicit component model in `app.js`.

As a result, the paper should have better theoretical interpretability and cleaner large-scale optimization behavior for principal-curve recovery, while `app.js` is better understood as an interactive graph-regularized clustering/hub-relaxation method.
