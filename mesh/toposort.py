"""Topological sort: order a DAG so every edge points forward, or prove it cannot.

A topological order of a directed graph lists its nodes so that for every
edge, the node it comes from appears before the node it goes to. It is the
order you can do tasks in when some tasks depend on others: a dependency
must come before the thing that needs it. Such an order exists exactly when
the graph has no directed cycle, because a cycle is a set of tasks each
waiting on the next around the loop, with no place to start. Kahn's
algorithm builds the order and detects the impossibility in one pass. It
counts each node's in-degree, the number of edges pointing at it, and seeds
a queue with every node of in-degree zero, the tasks that depend on
nothing. It repeatedly removes a zero-in-degree node, appends it to the
order, and decrements the in-degree of each node it points to, which may
drop those to zero and enqueue them in turn. When the queue empties, if the
order contains every node the graph was acyclic and the order is valid; if
some nodes never reached in-degree zero, they are exactly the nodes caught
in or downstream of a cycle, and no topological order exists, so the
algorithm refuses rather than returning a partial order that silently drops
them. The choice of which zero-in-degree node to take next is free, so a
DAG can have many valid orders; taking them in a fixed sorted order makes
this one deterministic, which matters only for reproducibility, not
correctness. The sorter returns the order for a DAG, refuses a cyclic
graph by naming the nodes left stuck, and reports the count of source nodes,
the in-degree-zero starts, because a DAG with a single source has one
natural beginning while many sources mean independent chains.
"""

from __future__ import annotations

from collections import deque

from mesh.errors import Cyclic, Invalid
from mesh.graph import Graph


class TopologicalSort:
    def __init__(self, graph: Graph) -> None:
        if not graph.directed:
            raise Invalid("topological sort is defined only for a directed graph")
        self.graph = graph
        self.in_degree: dict[str, int] = dict.fromkeys(graph.nodes(), 0)
        for node in graph.nodes():
            for neighbor in graph.neighbors(node):
                self.in_degree[neighbor] += 1
        self.sources = sorted(n for n, d in self.in_degree.items() if d == 0)

    def order(self) -> list[str]:
        remaining = dict(self.in_degree)
        # a sorted seed makes the order deterministic among the valid ones
        queue: deque[str] = deque(sorted(n for n, d in remaining.items() if d == 0))
        result: list[str] = []
        while queue:
            node = queue.popleft()
            result.append(node)
            for neighbor in sorted(self.graph.neighbors(node)):
                remaining[neighbor] -= 1
                if remaining[neighbor] == 0:
                    queue.append(neighbor)
        if len(result) != self.graph.node_count():
            stuck = sorted(n for n in remaining if remaining[n] > 0)
            raise Cyclic(
                f"no topological order exists; {stuck} are caught in or below a "
                "cycle and never reached in-degree zero"
            )
        return result

    def is_acyclic(self) -> bool:
        try:
            self.order()
        except Cyclic:
            return False
        return True

    def note(self) -> str:
        return (
            f"{len(self.sources)} source node(s) of in-degree zero: "
            f"{self.sources}; one source is a single natural start, many mean "
            "independent chains"
        )
