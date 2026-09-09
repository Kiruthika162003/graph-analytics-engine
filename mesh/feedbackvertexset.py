"""Feedback vertex set: the fewest nodes whose removal leaves no cycle at all.

A feedback vertex set breaks every cycle; what remains is a forest.
It is the set of processes to kill to resolve every deadlock, the
variables to condition on to make a factor graph a tree, and one of
Karp's original NP-complete problems. Two methods live here. The
exact one tries subsets in order of size, smallest first, and returns
the first whose removal leaves a forest, which is checked by the edge
count of each remaining component being one less than its node count;
it is fine to fourteen nodes. The greedy one prunes first, deleting
any node of degree at most one since no cycle uses it, then removes
the remaining node of highest degree, and repeats until nothing is
left, which is the natural heuristic and is never below the exact
size. Known values pin both: a tree needs nothing, a cycle needs one
node, two cycles sharing a node need that one node, a wheel needs the
hub and one rim node, a complete graph on n nodes needs n minus 2,
and a three by three grid needs two. The engine reports both answers,
checks that each is a feedback set by removal, and states how far the
greedy answer sits above the exact one. A directed graph is refused
since a directed feedback set is a different problem with its own
arcs to break.
"""

from __future__ import annotations

from itertools import combinations

from mesh.errors import Invalid
from mesh.graph import Graph


class FeedbackVertexSet:
    def __init__(self, graph: Graph) -> None:
        if graph.directed:
            raise Invalid("this feedback set breaks undirected cycles; pass an undirected one")
        self.graph = graph
        self.nodes = graph.nodes()

    def _is_forest(self, removed: set[str]) -> bool:
        remaining = [n for n in self.nodes if n not in removed]
        seen: set[str] = set()
        for start in remaining:
            if start in seen:
                continue
            component = {start}
            stack = [start]
            edges = 0
            while stack:
                node = stack.pop()
                for m in self.graph.neighbors(node):
                    if m in removed:
                        continue
                    edges += 1
                    if m not in component:
                        component.add(m)
                        stack.append(m)
            seen |= component
            if edges // 2 != len(component) - 1:
                return False
        return True

    def exact(self) -> set[str]:
        if len(self.nodes) > 14:
            raise Invalid("the exact search tries every subset; keep it to fourteen nodes")
        for size in range(len(self.nodes) + 1):
            for chosen in combinations(self.nodes, size):
                if self._is_forest(set(chosen)):
                    return set(chosen)
        return set(self.nodes)

    def greedy(self) -> set[str]:
        removed: set[str] = set()
        alive = set(self.nodes)
        degree = {n: self.graph.degree(n) for n in self.nodes}
        while alive:
            # prune every node that no cycle can use
            pruned = True
            while pruned:
                pruned = False
                for node in sorted(alive):
                    if degree[node] <= 1:
                        alive.discard(node)
                        for m in self.graph.neighbors(node):
                            if m in alive:
                                degree[m] -= 1
                        pruned = True
            if not alive:
                break
            victim = max(sorted(alive), key=lambda n: degree[n])
            removed.add(victim)
            alive.discard(victim)
            for m in self.graph.neighbors(victim):
                if m in alive:
                    degree[m] -= 1
        return removed

    def breaks_every_cycle(self, chosen: set[str]) -> bool:
        return self._is_forest(chosen)

    def note(self) -> str:
        exact = self.exact()
        greedy = self.greedy()
        gap = len(greedy) - len(exact)
        return (
            f"exact feedback set {sorted(exact)} of size {len(exact)}; greedy took "
            f"{len(greedy)}, {gap} above the exact size"
        )
