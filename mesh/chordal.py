"""Chordal graphs: every long cycle has a shortcut, and hard problems turn easy.

A graph is chordal when every cycle of four or more nodes has a chord,
an edge joining two nodes of the cycle that are not neighbors on it.
Trees, complete graphs, and interval graphs are chordal; a bare square
is the smallest graph that is not. The class matters because on chordal
graphs problems that are NP-hard in general become polynomial: the
chromatic number equals the clique number, the treewidth is the clique
number minus one, and both are read off a perfect elimination ordering,
an ordering of the nodes in which each node's later neighbors form a
clique. Chordal graphs are exactly the graphs that have such an
ordering, and maximum cardinality search finds one when it exists: pick
nodes one at a time, always the unpicked node with the most already-
picked neighbors, and the reverse of that order is a perfect elimination
ordering if and only if the graph is chordal. Checking the ordering is
then a matter of verifying, for each node, that its later neighbors are
pairwise adjacent, and a failure of that check is a witness that a
chordless cycle exists. The engine runs the search, checks the ordering,
and when the graph is chordal computes the clique number from the
ordering as the largest later-neighborhood plus one, which it confirms
against Bron-Kerbosch and uses to report the chromatic number and
treewidth exactly. It refuses a directed graph, and reports the verdict
with the offending node when the check fails, because a non-chordal
verdict without the node that broke it leaves the reader hunting for the
square.
"""

from __future__ import annotations

from mesh.bronkerbosch import BronKerbosch
from mesh.errors import Invalid
from mesh.graph import Graph


class Chordal:
    def __init__(self, graph: Graph) -> None:
        if graph.directed:
            raise Invalid("chordality is defined on undirected graphs")
        self.graph = graph
        self._nbrs = {n: set(graph.neighbors(n)) for n in graph.nodes()}
        self.order = self._maximum_cardinality_search()
        self.witness: str | None = None
        self.is_chordal = self._perfect_elimination(self.order)

    def _maximum_cardinality_search(self) -> list[str]:
        # always take the unpicked node with the most picked neighbors
        picked: list[str] = []
        weight = dict.fromkeys(self.graph.nodes(), 0)
        remaining = set(self.graph.nodes())
        while remaining:
            node = max(remaining, key=lambda n: (weight[n], n))
            remaining.discard(node)
            picked.append(node)
            for m in self._nbrs[node]:
                if m in remaining:
                    weight[m] += 1
        picked.reverse()  # the reverse of the search order is the candidate ordering
        return picked

    def _perfect_elimination(self, order: list[str]) -> bool:
        position = {n: i for i, n in enumerate(order)}
        for node in order:
            later = [m for m in self._nbrs[node] if position[m] > position[node]]
            for i, a in enumerate(later):
                for b in later[i + 1 :]:
                    if b not in self._nbrs[a]:
                        self.witness = node  # its later neighbors are not a clique
                        return False
        return True

    def clique_number(self) -> int:
        if not self.is_chordal:
            raise Invalid("the clique number from the ordering is exact only when chordal")
        position = {n: i for i, n in enumerate(self.order)}
        best = 0
        for node in self.order:
            later = sum(1 for m in self._nbrs[node] if position[m] > position[node])
            best = max(best, later + 1)
        return best

    def chromatic_number(self) -> int:
        return self.clique_number()  # perfect graphs: colors equal the largest clique

    def treewidth(self) -> int:
        return self.clique_number() - 1

    def matches_bron_kerbosch(self) -> bool:
        return self.clique_number() == BronKerbosch(self.graph).clique_number()

    def note(self) -> str:
        if not self.is_chordal:
            return f"not chordal: '{self.witness}' has later neighbors that are not a clique"
        return (
            f"chordal: clique number {self.clique_number()}, so chromatic number "
            f"{self.chromatic_number()} and treewidth {self.treewidth()} exactly, read off "
            "the perfect elimination ordering"
        )
