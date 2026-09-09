"""Self-check: the engine runs its own identities on any graph handed to it and reports.

The probes hold the engine's guarantees on graphs built to stress
them; a self-check runs a chosen subset of the same identities on a
graph a reader brings, so a strange input is examined by the facts
that must hold for any graph rather than by one algorithm's word.
The identities are the cheap universal ones: the degree sum is twice
the edge count; the number of zero Laplacian eigenvalues equals the
number of pieces; the eigenvalue squares sum to twice the edges; the
quad census sums to n choose 4 and its implied triangle count
matches a direct count; the Wiener index of a tree equals its
edge-cut sum; and a summary line's counts agree with the graph's
own. Each check returns a name, a verdict, and a reading, and the
self-check refuses nothing: on a directed graph the undirected
identities are marked skipped with the reason, on a graph too large
for the exact counts the census is skipped, and an empty graph runs
the checks that make sense on it. The report is a list of lines so
the command line can print it, and a single line says how many held,
how many were skipped, and how many failed, which is the number a
reader looks for. The tests hold that every identity holds on a
handful of shapes, that a directed graph produces skips rather than
failures, and that the summary line counts add up.
"""

from __future__ import annotations

from math import comb

from mesh.graph import Graph
from mesh.graphenergy import GraphEnergy
from mesh.graphsummary import GraphSummary
from mesh.laplacianspectrum import LaplacianSpectrum
from mesh.quadcensus import QuadCensus
from mesh.wienerindex import WienerIndex

Result = tuple[str, str, str]


class SelfCheck:
    def __init__(self, graph: Graph) -> None:
        self.graph = graph
        self.results: list[Result] = []
        self._run()

    def _add(self, name: str, verdict: str, reading: str) -> None:
        self.results.append((name, verdict, reading))

    def _run(self) -> None:
        g = self.graph
        degree_sum = sum(g.degree(n) for n in g.nodes())
        expected = 2 * g.edge_count() if not g.directed else g.edge_count()
        self._add(
            "handshake",
            "held" if degree_sum == expected else "failed",
            f"degree sum {degree_sum} against {expected}",
        )
        if g.directed:
            for name in ("laplacian zeros", "spectrum squares", "quad census", "tree wiener"):
                self._add(name, "skipped", "undirected identity")
        else:
            self._undirected()
        summary = GraphSummary(g)
        line = summary.lines()[0]
        # the empty graph's line names no counts, and was first read as a failure
        counts_ok = f"{g.node_count()} node(s) and {g.edge_count()} edge(s)" in line or (
            g.node_count() == 0 and line == "empty graph: no nodes"
        )
        self._add("summary counts", "held" if counts_ok else "failed", line)

    def _undirected(self) -> None:
        g = self.graph
        pieces = GraphSummary(g).pieces()
        if g.node_count():
            zeros = LaplacianSpectrum(g).components()
            self._add(
                "laplacian zeros",
                "held" if zeros == pieces else "failed",
                f"{zeros} zero(s) against {pieces} piece(s)",
            )
            squares = GraphEnergy(g).closed_walks(2)
            # Jacobi rounds to about a millionth on a ten-node graph, and a tolerance of
            # 1e-6 failed a probe that printed 40.000 against 40; the bar is now relative
            close = abs(squares - 2 * g.edge_count()) < 1e-4 * max(1, 2 * g.edge_count())
            self._add(
                "spectrum squares",
                "held" if close else "failed",
                f"{squares:.3f} against {2 * g.edge_count()}",
            )
        else:
            self._add("laplacian zeros", "skipped", "no nodes")
            self._add("spectrum squares", "skipped", "no nodes")
        if g.node_count() <= 40:
            qc = QuadCensus(g)
            ok = qc.sums_to_choose_four() and qc.triangle_identity_holds()
            self._add(
                "quad census",
                "held" if ok else "failed",
                f"{qc.total()} quad(s) against {comb(g.node_count(), 4)}",
            )
        else:
            self._add("quad census", "skipped", "more than forty nodes")
        if g.node_count() and g.edge_count() == g.node_count() - 1 and pieces == 1:
            wi = WienerIndex(g)
            self._add(
                "tree wiener",
                "held" if wi.tree_edge_identity_holds() else "failed",
                f"index {wi.wiener():g}",
            )
        else:
            self._add("tree wiener", "skipped", "not a tree")

    def counts(self) -> dict[str, int]:
        out = {"held": 0, "skipped": 0, "failed": 0}
        for _name, verdict, _reading in self.results:
            out[verdict] += 1
        return out

    def all_held(self) -> bool:
        return self.counts()["failed"] == 0

    def lines(self) -> list[str]:
        out = [f"{name}: {verdict} ({reading})" for name, verdict, reading in self.results]
        c = self.counts()
        out.append(f"{c['held']} held, {c['skipped']} skipped, {c['failed']} failed")
        return out

    def note(self) -> str:
        return self.lines()[-1]
