"""Bron-Kerbosch: every maximal clique, found once, with pivoting to prune the search.

A clique is a set of nodes every pair of which is adjacent, and a maximal
clique is one that no further node can join. Maximal cliques are the
tightest groups a graph holds, the committees where everyone knows
everyone, and enumerating them is the basis of clique-based community
finding and of the maximum clique problem, which is the largest of them.
The number of maximal cliques can be exponential in the node count, so no
algorithm is fast on every graph, but Bron-Kerbosch with pivoting is the
standard that runs well on the sparse graphs of practice. It maintains
three sets: R, the clique being built; P, the candidates that are adjacent
to everything in R and could extend it; and X, the nodes already
processed that are adjacent to everything in R but must be excluded to
avoid reporting the same clique twice. When both P and X are empty, R is
maximal and is reported. Otherwise the search picks a candidate v from P,
recurses with v added to R and P and X restricted to v's neighbors, then
moves v from P to X. The pivot is the pruning: choose a pivot node from P
or X with the most neighbors in P, and only branch on candidates that are
not neighbors of the pivot, because any maximal clique containing a
pivot-neighbor also contains the pivot or was reached through it, so those
branches are redundant. Pivoting cuts the branching factor sharply on
dense regions. The enumerator returns every maximal clique, the largest,
and the clique number, and it refuses a directed graph. It reports the
count of maximal cliques against the node count, because a count far
above the node count is a graph of heavily overlapping dense groups, the
shape where the exponential worst case starts to show.
"""

from __future__ import annotations

from mesh.errors import Invalid
from mesh.graph import Graph


class BronKerbosch:
    def __init__(self, graph: Graph) -> None:
        if graph.directed:
            raise Invalid("cliques are defined on undirected graphs")
        self.graph = graph
        self._nbrs = {n: set(graph.neighbors(n)) for n in graph.nodes()}
        self.cliques: list[frozenset[str]] = []
        self.branches = 0
        # an empty graph would otherwise report the empty set as a clique
        if graph.node_count():
            self._search(set(), set(graph.nodes()), set())
        self.cliques.sort(key=lambda c: (-len(c), sorted(c)))

    def _search(self, r: set[str], p: set[str], x: set[str]) -> None:
        if not p and not x:
            self.cliques.append(frozenset(r))
            return
        # pivot on the node with the most candidates among its neighbors
        pivot = max(p | x, key=lambda u: (len(self._nbrs[u] & p), u))
        for v in sorted(p - self._nbrs[pivot]):
            self.branches += 1
            self._search(r | {v}, p & self._nbrs[v], x & self._nbrs[v])
            p = p - {v}
            x = x | {v}

    def largest(self) -> frozenset[str]:
        if not self.cliques:
            raise Invalid("an empty graph has no clique")
        return self.cliques[0]

    def clique_number(self) -> int:
        return len(self.cliques[0]) if self.cliques else 0

    def note(self) -> str:
        return (
            f"{len(self.cliques)} maximal clique(s) over {self.graph.node_count()} "
            f"node(s), clique number {self.clique_number()}, {self.branches} "
            "branch(es); a count far above the node count is heavily overlapping "
            "dense groups"
        )
