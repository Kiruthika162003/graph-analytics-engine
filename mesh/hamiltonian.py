"""Hamiltonian paths: visit every node exactly once, found by search or ruled out by degree.

An Eulerian trail uses every edge once and is decided by a parity check.
A Hamiltonian path visits every node once, and that innocent change of
word makes the problem NP-complete: no parity trick exists, and the
honest tool is a backtracking search with pruning, capped at a node
count where it stays tractable. The search extends a path one node at
a time to an unvisited neighbor and backs up when stuck, and two prunes
carry most of the work. If any unvisited node has no unvisited neighbor
and is not the only node left, the path can never reach it, so the
branch dies at once. And for a cycle rather than a path, the search
starts at a fixed node, since a cycle through every node passes through
any node, which removes a factor of the node count. Sufficient
conditions can sometimes settle the question without a search. Dirac's
theorem says that if every node has degree at least half the node
count, a Hamiltonian cycle exists, and the engine checks it first and
reports when it applied, though it still constructs the cycle by search
since the theorem promises existence without exhibiting one. Necessary
conditions rule cases out: a cycle needs every node to have degree at
least two and the graph to be connected, a path needs at most two nodes
of degree one in a connected graph, and the engine refuses to search
when those fail, naming the reason. The finder returns whether a path
or cycle exists and one witness when it does, and reports the branches
explored against the factorial the unpruned search would face, because
that ratio is the pruning doing its work and a ratio near one is a
graph that gave the prunes nothing to cut.
"""

from __future__ import annotations

import math

from mesh.connectedcomponents import ConnectedComponents
from mesh.errors import Invalid
from mesh.graph import Graph

_NODE_CAP = 14


class Hamiltonian:
    def __init__(self, graph: Graph, cycle: bool = False) -> None:
        if graph.directed:
            raise Invalid("this Hamiltonian search is for undirected graphs")
        if graph.node_count() == 0:
            raise Invalid("an empty graph has no path to find")
        if graph.node_count() > _NODE_CAP:
            raise Invalid(f"more than {_NODE_CAP} nodes; the search is capped")
        self.graph = graph
        self.cycle = cycle
        self.nodes = graph.nodes()
        self._nbrs = {n: set(graph.neighbors(n)) for n in self.nodes}
        self.branches = 0
        self.reason = ""
        self.dirac = self._dirac()
        self.witness: list[str] | None = None
        self.exists = self._decide()

    def _dirac(self) -> bool:
        n = len(self.nodes)
        return n >= 3 and all(self.graph.degree(v) * 2 >= n for v in self.nodes)

    def _ruled_out(self) -> bool:
        n = len(self.nodes)
        if n > 1 and not ConnectedComponents(self.graph).is_connected():
            self.reason = "disconnected"
            return True
        if self.cycle and n >= 3 and any(self.graph.degree(v) < 2 for v in self.nodes):
            self.reason = "a node of degree below two"
            return True
        if not self.cycle and sum(1 for v in self.nodes if self.graph.degree(v) == 1) > 2:
            self.reason = "more than two nodes of degree one"
            return True
        return False

    def _decide(self) -> bool:
        if self._ruled_out():
            return False
        starts = self.nodes[:1] if self.cycle else self.nodes
        for start in starts:
            path = [start]
            if self._extend(path, {start}):
                self.witness = path
                return True
        self.reason = "search exhausted"
        return False

    def _extend(self, path: list[str], seen: set[str]) -> bool:
        self.branches += 1
        if len(path) == len(self.nodes):
            return not self.cycle or path[0] in self._nbrs[path[-1]]
        # a stranded unvisited node with no unvisited neighbor kills the branch
        unvisited = [v for v in self.nodes if v not in seen]
        if len(unvisited) > 1:
            for v in unvisited:
                if not (self._nbrs[v] - seen):
                    return False
        for nxt in sorted(self._nbrs[path[-1]] - seen):
            path.append(nxt)
            seen.add(nxt)
            if self._extend(path, seen):
                return True
            path.pop()
            seen.discard(nxt)
        return False

    def unpruned_bound(self) -> int:
        n = len(self.nodes)
        return math.factorial(n - 1) if self.cycle else math.factorial(n)

    def note(self) -> str:
        kind = "cycle" if self.cycle else "path"
        if not self.exists:
            return f"no Hamiltonian {kind}: {self.reason} after {self.branches} branch(es)"
        via = "Dirac guaranteed it" if self.dirac and self.cycle else "found by search"
        return (
            f"Hamiltonian {kind} exists, {via}, {self.branches} branch(es) against an "
            f"unpruned bound of {self.unpruned_bound()}"
        )
