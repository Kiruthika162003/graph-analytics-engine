"""Directed Eulerian circuits: walk every arrow once, the way it points.

The undirected Eulerian question is decided by even degrees. The
directed version, walking every edge exactly once in its own direction
and returning to the start, is decided by a balance instead: every node
must have in-degree equal to out-degree, since each visit arrives on one
edge and leaves on another, and the nodes carrying edges must all lie in
one strongly connected piece, since a walk that follows arrows must be
able to get from any edge to any other and back. An open directed trail
relaxes the balance at exactly two nodes, the start with one more edge
out than in and the end with one more in than out. These conditions
are what de Bruijn sequences, DNA assembly by k-mer overlap, and
one-way street sweeping all rest on: assembling a genome from reads is,
in one formulation, finding an Eulerian path in a directed graph of
overlaps. The walk is Hierholzer's again, now consuming each edge only
from its tail, so no edge is walked backward: follow unused out-edges
until stuck, which happens only at the start by balance, and splice in
sub-circuits from any node on the walk with out-edges remaining. The
finder checks balance and strong connectivity of the edge-carrying
nodes before walking, refuses with the offending node named when
either fails, returns the circuit or open trail, and reports the
imbalance count, zero for a circuit and two for a trail, because that
count is the whole verdict and it is known before a single edge is
walked.
"""

from __future__ import annotations

from mesh.errors import Invalid
from mesh.graph import Graph
from mesh.tarjanscc import TarjanSCC


class DirectedEulerian:
    def __init__(self, graph: Graph) -> None:
        if not graph.directed:
            raise Invalid("this finder walks directed edges; use the undirected one otherwise")
        if graph.edge_count() == 0:
            raise Invalid("a graph with no edges has no trail to walk")
        self.graph = graph
        self.start_node, self.end_node = self._balance()
        self._check_strong()
        self.trail = self._hierholzer()

    def _balance(self) -> tuple[str | None, str | None]:
        out_minus_in = dict.fromkeys(self.graph.nodes(), 0)
        for u, v, _w in self.graph.edges():
            out_minus_in[u] += 1
            out_minus_in[v] -= 1
        start = [n for n, d in out_minus_in.items() if d == 1]
        end = [n for n, d in out_minus_in.items() if d == -1]
        others = [n for n, d in out_minus_in.items() if d not in (-1, 0, 1)]
        if others or len(start) != len(end) or len(start) > 1:
            bad = others[0] if others else (start + end)[0]
            raise Invalid(
                f"'{bad}' breaks the balance; a directed trail needs in-degree equal to "
                "out-degree everywhere, or one surplus out and one surplus in"
            )
        return (start[0], end[0]) if start else (None, None)

    def _check_strong(self) -> None:
        # the edge-carrying nodes must be mutually reachable along the arrows,
        # with the open trail's endpoints joined by an imaginary closing edge
        g = Graph(directed=True)
        for n in self.graph.nodes():
            g.add_node(n)
        for u, v, w in self.graph.edges():
            g.add_edge(u, v, w)
        if self.start_node is not None and not g.has_edge(self.end_node, self.start_node):
            g.add_edge(self.end_node, self.start_node)
        carriers = {n for n in g.nodes() if g.degree(n) > 0 or g.in_degree(n) > 0}
        comps = TarjanSCC(g).components()
        biggest = max(comps, key=len)
        if not carriers <= biggest:
            raise Invalid("the edges are not strongly connected; no single walk covers them")

    def _hierholzer(self) -> list[str]:
        unused = {n: sorted(self.graph.neighbors(n), reverse=True) for n in self.graph.nodes()}
        start = self.start_node if self.start_node is not None else next(
            n for n in self.graph.nodes() if unused[n]
        )
        stack = [start]
        trail: list[str] = []
        while stack:
            node = stack[-1]
            if unused[node]:
                stack.append(unused[node].pop())  # consume from the tail only
            else:
                trail.append(stack.pop())
        trail.reverse()
        return trail

    def is_circuit(self) -> bool:
        return self.start_node is None

    def imbalance_count(self) -> int:
        return 0 if self.is_circuit() else 2

    def note(self) -> str:
        kind = "circuit" if self.is_circuit() else "open trail"
        return (
            f"directed Eulerian {kind} over {len(self.trail) - 1} edge(s) with "
            f"{self.imbalance_count()} unbalanced node(s); the verdict was known before "
            "a single edge was walked"
        )
