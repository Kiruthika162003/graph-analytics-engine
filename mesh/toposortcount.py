"""Counting topological orders: how many valid schedules a dependency graph allows.

A topological sort gives one order in which the tasks of a DAG can run.
The number of such orders says something the single order cannot: how
much freedom the dependencies leave. A chain has exactly one order, a
set of independent tasks has every permutation, and a real dependency
graph sits somewhere between, its count a direct measure of scheduling
slack. Counting them exactly is hard in general, sharp-P-complete, so
the honest tool is a dynamic program over subsets that is exact and
exponential, capped at a node count where it stays fast. The state is
the set of tasks already placed, and the count for a set is the sum,
over every task whose predecessors all lie in the set, of the count for
the set with that task removed, since any valid order of the set ends
with such a task. The empty set has one order, and the full set's count
is the answer, reached over two to the n states with a bitmask for each.
Two facts anchor the result. A cyclic graph has zero orders, which the
program discovers when the full set is unreachable, and the count of a
DAG is at least one and at most the factorial of the node count, with
the factorial attained only when there are no edges. The counter
returns the number of orders, refuses a graph past the cap, and reports
the count against the factorial as a fraction, because that fraction is
the freedom the dependencies left: near one is a schedule with almost
no constraints and near zero is a graph that is nearly a chain, where
the single topological order was the only choice all along.
"""

from __future__ import annotations

import math

from mesh.errors import Invalid
from mesh.graph import Graph

_NODE_CAP = 20


class TopologicalCount:
    def __init__(self, graph: Graph) -> None:
        if not graph.directed:
            raise Invalid("topological orders are defined on a directed graph")
        if graph.node_count() > _NODE_CAP:
            raise Invalid(f"more than {_NODE_CAP} nodes; the subset program is capped")
        self.graph = graph
        self.nodes = graph.nodes()
        self._index = {n: i for i, n in enumerate(self.nodes)}
        self.count = self._count()

    def _count(self) -> int:
        n = len(self.nodes)
        # predecessor mask per node: a node may be placed once all are placed
        preds = [0] * n
        for u, v, _w in self.graph.edges():
            preds[self._index[v]] |= 1 << self._index[u]
        ways = [0] * (1 << n)
        ways[0] = 1
        for mask in range(1 << n):
            if ways[mask] == 0:
                continue
            for i in range(n):
                bit = 1 << i
                if mask & bit or preds[i] & ~mask:
                    continue  # already placed, or a predecessor is missing
                ways[mask | bit] += ways[mask]
        return ways[(1 << n) - 1]

    def is_acyclic(self) -> bool:
        return self.count > 0

    def freedom(self) -> float:
        n = len(self.nodes)
        return self.count / math.factorial(n) if n else 1.0

    def note(self) -> str:
        if not self.is_acyclic():
            return "zero orders: the graph has a cycle and no schedule exists"
        return (
            f"{self.count} topological order(s), {self.freedom() * 100:.1f}% of the "
            f"{math.factorial(len(self.nodes))} permutations; near 100 is nearly no "
            "constraint, near 0 is nearly a chain"
        )
