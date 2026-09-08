"""Stoer-Wagner: the cheapest way to cut a graph in two, with no source or sink.

The s-t minimum cut separates two chosen nodes. The global minimum cut
asks for the cheapest set of edges whose removal splits the graph into
two non-empty pieces, with no say over which nodes end up on which side:
the weakest seam in the whole network. It could be found by running an
s-t cut between every pair, but Stoer-Wagner does it in a single sweep of
phases without any flow computation at all. Each phase grows a set A
from an arbitrary start by repeatedly adding the node most tightly
connected to A, the one whose total edge weight into A is largest, until
every node is in. The last two nodes added, s then t, define a cut of the
phase: t alone against everything else, with weight equal to t's total
connection into A at the moment it was added. The theorem is that this
cut is a minimum s-t cut for that particular s and t. So either the
global minimum cut separates s from t, in which case this phase found
it, or it does not, in which case s and t are on the same side and
merging them into one node loses nothing. The phase ends by merging s
and t, adding their edge weights together, and the next phase runs on
the smaller graph. After nodes-minus-one phases the smallest cut seen is
the global minimum. The tightness selection dominates the cost, nodes
squared per phase with a plain scan, nodes cubed overall, which is the
version here. The engine records which original nodes each merged node
stands for so the cut can be reported as a partition, refuses a
directed graph and one with fewer than two nodes, and reports the cut
weight against the minimum degree, because the minimum weighted degree
is a cut too, a single node against the rest, and the global cut being
strictly cheaper means the weakest seam runs between groups rather than
around one poorly attached node.
"""

from __future__ import annotations

import math

from mesh.errors import Invalid
from mesh.graph import Graph


class StoerWagner:
    def __init__(self, graph: Graph) -> None:
        if graph.directed:
            raise Invalid("the global minimum cut is defined on an undirected graph")
        if graph.node_count() < 2:
            raise Invalid("a cut needs at least two nodes to separate")
        self.graph = graph
        self.weight = math.inf
        self.partition: set[str] = set()
        self._run()

    def _run(self) -> None:
        # weights between super-nodes, and which original nodes each stands for
        nodes = self.graph.nodes()
        w: dict[str, dict[str, float]] = {u: {} for u in nodes}
        for u, v, wt in self.graph.edges():
            w[u][v] = w[u].get(v, 0.0) + wt
            w[v][u] = w[v].get(u, 0.0) + wt
        stands_for: dict[str, set[str]] = {u: {u} for u in nodes}
        while len(w) > 1:
            s, t, cut = self._phase(w)
            if cut < self.weight:
                self.weight = cut
                self.partition = set(stands_for[t])
            self._merge(w, stands_for, s, t)

    @staticmethod
    def _phase(w: dict[str, dict[str, float]]) -> tuple[str, str, float]:
        # grow A by the most tightly connected node; the last two define the cut
        remaining = set(w)
        tight: dict[str, float] = dict.fromkeys(remaining, 0.0)
        order: list[str] = []
        while remaining:
            node = max(remaining, key=lambda n: (tight[n], n))
            remaining.discard(node)
            order.append(node)
            for nbr, wt in w[node].items():
                if nbr in remaining:
                    tight[nbr] += wt
        s, t = order[-2], order[-1]
        return s, t, tight[t]

    @staticmethod
    def _merge(
        w: dict[str, dict[str, float]], stands_for: dict[str, set[str]], s: str, t: str
    ) -> None:
        # fold t into s, adding parallel weights, and drop t
        for nbr, wt in w[t].items():
            if nbr == s:
                continue
            w[s][nbr] = w[s].get(nbr, 0.0) + wt
            w[nbr][s] = w[nbr].get(s, 0.0) + wt
            del w[nbr][t]
        w[s].pop(t, None)
        del w[t]
        stands_for[s] |= stands_for.pop(t)

    def sides(self) -> tuple[set[str], set[str]]:
        rest = set(self.graph.nodes()) - self.partition
        return self.partition, rest

    def min_degree_cut(self) -> float:
        return min(sum(self.graph.neighbors(n).values()) for n in self.graph.nodes())

    def note(self) -> str:
        seam = "between groups" if self.weight < self.min_degree_cut() else \
            "around one poorly attached node"
        return (
            f"global minimum cut {self.weight} against a minimum degree of "
            f"{self.min_degree_cut()}; the weakest seam runs {seam}"
        )
