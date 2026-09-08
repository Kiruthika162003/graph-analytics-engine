"""Split graphs: nodes that divide into one clique and one independent set.

A split graph is one whose nodes can be partitioned into a clique and
an independent set, with any edges at all between the two parts. It is
the shape of a core of mutual acquaintances surrounded by strangers who
know only some of the core, and it is a class where the hard problems
collapse: the clique number is the core's size plus at most one, the
independence number is the fringe plus at most one, and both are found
by a degree count. Hammer and Simeone showed the class is decided by the
degree sequence alone. Sort degrees descending, and let m be the largest
index such that the m-th degree is at least m minus one; the graph is
split exactly when the sum of the first m degrees equals m times m minus
one plus the sum of the remaining degrees. The identity says the top m
nodes have exactly the degree a clique of m gives them plus one per
edge to the fringe, and the fringe's degrees are all accounted for by
those edges. When the identity holds, the top m nodes by degree are the
clique and the rest the independent set, which the engine verifies
directly rather than trusting the count. Split graphs are chordal, and
their complements are split too, both of which are cheap invariants to
check against the modules already here. The recognizer returns the
verdict and the partition, verifies the clique is a clique and the
fringe is independent, refuses a directed graph, and reports the two
part sizes, because a split graph is a core-periphery structure with
the fuzz removed, and the sizes say how much core there was.
"""

from __future__ import annotations

from itertools import combinations

from mesh.chordal import Chordal
from mesh.errors import Invalid
from mesh.graph import Graph


class SplitGraph:
    def __init__(self, graph: Graph) -> None:
        if graph.directed:
            raise Invalid("split graphs are undirected")
        self.graph = graph
        self.by_degree = sorted(graph.nodes(), key=lambda n: (-graph.degree(n), n))
        self.is_split, self.core_size = self._hammer_simeone()
        self.clique = set(self.by_degree[: self.core_size]) if self.is_split else set()
        self.fringe = set(self.by_degree[self.core_size :]) if self.is_split else set()

    def _hammer_simeone(self) -> tuple[bool, int]:
        degrees = [self.graph.degree(n) for n in self.by_degree]
        n = len(degrees)
        if n == 0:
            return True, 0
        # m is the largest index whose degree is at least m - 1
        m = 0
        for i in range(1, n + 1):
            if degrees[i - 1] >= i - 1:
                m = i
        left = sum(degrees[:m])
        right = m * (m - 1) + sum(degrees[m:])
        return left == right, m

    def partition_holds(self) -> bool:
        if not self.is_split:
            return False
        core = sorted(self.clique)
        clique_ok = all(self.graph.has_edge(a, b) for a, b in combinations(core, 2))
        fringe_ok = not any(
            self.graph.has_edge(a, b) for a, b in combinations(sorted(self.fringe), 2)
        )
        return clique_ok and fringe_ok

    def is_chordal_too(self) -> bool:
        return Chordal(self.graph).is_chordal

    def note(self) -> str:
        if not self.is_split:
            return "not a split graph: the degree identity of Hammer and Simeone fails"
        return (
            f"split graph: a clique of {len(self.clique)} and an independent fringe of "
            f"{len(self.fringe)}, verified directly; a core-periphery with the fuzz removed"
        )
