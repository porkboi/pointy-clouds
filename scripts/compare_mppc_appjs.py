#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
APP_JS = ROOT / "app.js"
MPPC_MAIN = ROOT / "MPPC-GitHub" / "MPPC-GitHub" / "mppc.m"
MPPC_UPDATE = ROOT / "MPPC-GitHub" / "MPPC-GitHub" / "updateyzb.m"
MPPC_ENERGY = ROOT / "MPPC-GitHub" / "MPPC-GitHub" / "calculateEnergy.m"
MPPC_INIT = ROOT / "MPPC-GitHub" / "MPPC-GitHub" / "initialize.m"


@dataclass(frozen=True)
class SourceFile:
    label: str
    path: Path
    lines: list[str]

    @classmethod
    def load(cls, label: str, path: Path) -> "SourceFile":
        return cls(label=label, path=path, lines=path.read_text().splitlines())

    def line_no(self, needle: str) -> int:
        for index, line in enumerate(self.lines, start=1):
            if needle in line:
                return index
        raise ValueError(f"Could not find {needle!r} in {self.path}")

    def ref(self, needle: str) -> str:
        return f"{self.label}:{self.line_no(needle)}"


def escape_cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", "<br>")


def join_refs(refs: Iterable[str]) -> str:
    return ", ".join(refs)


def build_rows(app: SourceFile, mppc: SourceFile, update: SourceFile, energy: SourceFile, init: SourceFile) -> list[tuple[str, str, str, str]]:
    return [
        (
            "Representation",
            "Ordered curve nodes `y` with optional disconnected components encoded by `cut_indices`.",
            "Unordered hub set `x` plus explicit graph edges rebuilt from geometry.",
            join_refs(
                [
                    mppc.ref("cut_indices - empty vector if y0 is connected."),
                    app.ref("function buildKnnGraph(points, k = 2) {"),
                    app.ref("state.edges = graph.edges;"),
                ]
            ),
        ),
        (
            "Objective terms",
            "Squared data-fit + length penalty + component-count penalty.",
            "Squared nearest-hub fit + path smoothness + origin shrinkage + graph TV + per-station cost.",
            join_refs(
                [
                    energy.ref("disc_energy = lambda1*(y_leng+lambda2*length(cut_indices)) + mass*sum((x-y(I,:)).^2,2);"),
                    app.ref("objective: fx + g1Info.value + g2 + g3 + g4,"),
                ]
            ),
        ),
        (
            "Assignment model",
            "Nearest-node assignment with optional continuous reprojection energy.",
            "Nearest-hub assignment only.",
            join_refs(
                [
                    mppc.ref("I = dsearchn(y_new,x)';"),
                    energy.ref("I_partial = computeProjs(x,y,I,cut_indices);"),
                    app.ref("function assignPoints(hubs, points) {"),
                ]
            ),
        ),
        (
            "Topology control",
            "Topology is explicit and edited by cut/connect/singleton passes tied to energy checks.",
            "Topology is implicit in graph edges; hubs can be split and edges greedily removed when objective drops.",
            join_refs(
                [
                    mppc.ref("[y_new,z_new,b,I,cut_indices_new] = energyCut(x,mass,y_new,z_new,b,[],cut_indices,lambda1,lambda2);"),
                    mppc.ref("[y_new,cut_indices_new] = connectCompsE(x,y_new,mass,I,cut_indices,lambda1,lambda2);"),
                    mppc.ref("[y_new,z_new,b,cut_indices_new] = checkSingletons(x,mass,I,y_new,z_new,b,cut_indices,lambda1,lambda2);"),
                    app.ref("function optimizeEdgesForState(state, metrics = null) {"),
                    app.ref("function buildSplitProposal(state) {"),
                ]
            ),
        ),
        (
            "Inner solver",
            "ADMM / split-Bregman with exact block shrinkage for `z` and tridiagonal solve for `y`.",
            "ADMM-style splitting, but `z1` and `z3` are updated by a fixed number of gradient-like steps.",
            join_refs(
                [
                    update.ref("z = bsxfun(@times,normalprez, max(normsprez-lambda/rho,0));"),
                    update.ref("y_new = tridiag((-rho)*ones(m-1,1),D,(-rho)*ones(m-1,1),RHS);"),
                    app.ref("for (let step = 0; step < 7; step += 1) {"),
                    app.ref("const proxGrad = add(grad[i], scale(sub(z[i], v[i]), state.rho));"),
                ]
            ),
        ),
        (
            "Graph / chain during local solve",
            "Chain structure is fixed inside the ADMM subproblem; components are handled separately.",
            "Graph may be rebuilt from current hubs during iteration, changing the local geometry.",
            join_refs(
                [
                    update.ref("for j=2:length(cut_indices)+2   %perform updates on each component. This is parallelizable"),
                    app.ref(': buildKnnGraph(state.x, 2);'),
                    app.ref("const fixedEdges = graphMode === \"knn\" ? buildKnnGraph(state.x, 2).edges : canonicalizeEdges(state.edges);"),
                ]
            ),
        ),
        (
            "Initialization",
            "Optional automatic initialization searches over singleton-heavy KMeans seeds.",
            "Explorer seeds by farthest-point sampling; tuning mode starts from 3-cluster KMeans.",
            join_refs(
                [
                    mppc.ref("[y0,cut_indices,~] = initialize(x,mass,lambda1,lambda2);"),
                    init.ref("[I_new,y_new] = kmeans(x,k); cut_indices = (1:k-1)';"),
                    app.ref("function sampleInitialHubs(points, count, seed = 77) {"),
                    app.ref("const clustering = kMeans(cloud, startK, 16, 111);"),
                ]
            ),
        ),
        (
            "Stopping rule",
            "Primary outer test is energy decrease plus topology/respacing checks; rho can adapt from residuals.",
            "Convergence is based on primal/dual residual thresholds with iteration and runtime caps.",
            join_refs(
                [
                    mppc.ref("while (~isequal(size(y),size(y_new)) || energy_prev-energy>tol*energy || energy_prev<energy ..."),
                    mppc.ref("if r_pri>5*r_dual, rho = rho*1.3;"),
                    app.ref("function computeResidualThresholds(state, absTolerance = 0.0005, relTolerance = 0.004) {"),
                    app.ref("if (iteration + 1 >= minIterations && residualSettled) {"),
                ]
            ),
        ),
        (
            "Dominant expensive step",
            "Nearest assignments plus structured linear solve on each chain component.",
            "Nearest assignments plus repeated all-pairs shortest-path traversals inside objective and `z1` updates.",
            join_refs(
                [
                    update.ref("[I,~] = dsearchn(y,x);"),
                    update.ref("y_new = tridiag((-rho)*ones(m-1,1),D,(-rho)*ones(m-1,1),RHS);"),
                    app.ref("const result = dijkstra(graph.adjacency, i);"),
                    app.ref("const g1Info = computePathSmoothness(hubs, assigned.weights, graph);"),
                ]
            ),
        ),
    ]


def main() -> None:
    app = SourceFile.load("app.js", APP_JS)
    mppc = SourceFile.load("mppc.m", MPPC_MAIN)
    update = SourceFile.load("updateyzb.m", MPPC_UPDATE)
    energy = SourceFile.load("calculateEnergy.m", MPPC_ENERGY)
    init = SourceFile.load("initialize.m", MPPC_INIT)

    rows = build_rows(app, mppc, update, energy, init)

    print("# MPPC-GitHub vs app.js")
    print()
    print("This table compares the current working-tree versions of the MATLAB MPPC reference and `app.js`.")
    print("Each row is source-backed rather than paper-backed, so it stays fair if either implementation changes.")
    print()
    print("| Aspect | MPPC-GitHub | app.js | Evidence |")
    print("| --- | --- | --- | --- |")
    for aspect, left, right, evidence in rows:
        print(
            f"| {escape_cell(aspect)} | {escape_cell(left)} | {escape_cell(right)} | {escape_cell(evidence)} |"
        )


if __name__ == "__main__":
    main()
