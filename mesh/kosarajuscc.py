"""Kosaraju's SCC: two passes, on the graph and its reverse, meet at the components.

Kosaraju finds the same strongly connected components as Tarjan but by a
different and arguably more transparent route: two depth-first passes with
the edges reversed between them. The first pass runs DFS over the graph and
records the order in which nodes finish, pushing each node onto a stack as
its exploration completes, so the stack ends with the last-finished node on
top. The second pass runs DFS over the reversed graph, taking start nodes
off that stack from the top, and every tree it grows in this second pass is
exactly one strongly connected component. The reason the two passes meet at
the components is the property that reversing every edge leaves the strongly
connected components unchanged, because mutual reachability is symmetric in
direction, while it flips the reachability between different components. So
starting the second pass from the node that finished last, which sits in a
component that is a source of the condensation, and exploring in the
reversed graph, confines each tree to a single component: the reversed
edges that would have led onward to other components now lead backward into
already-assigned ones. Kosaraju is easier to reason about than Tarjan's
single-pass low-link bookkeeping, at the cost of building the reverse graph
and traversing twice. This engine keeps both so each can check the other:
they must always agree on the partition, and a test asserts exactly that.
The finder returns the components, tells a node's component, counts them,
and reports the count and largest size, the same readings Tarjan gives, so
the two are interchangeable at the interface and differ only inside.
"""

from __future__ import annotations

from mesh.errors import Invalid, Missing
from mesh.graph import Graph


class KosarajuSCC:
    def __init__(self, graph: Graph) -> None:
        if not graph.directed:
            raise Invalid("strongly connected components are defined for digraphs")
        self.graph = graph
        self._components: list[set[str]] = []
        self._assignment: dict[str, int] = {}
        self._run()

    def _finish_order(self) -> list[str]:
        visited: set[str] = set()
        order: list[str] = []
        for start in self.graph.nodes():
            if start in visited:
                continue
            stack: list[tuple[str, list[str]]] = [
                (start, list(self.graph.neighbors(start)))
            ]
            visited.add(start)
            while stack:
                node, pending = stack[-1]
                if pending:
                    nbr = pending.pop(0)
                    if nbr not in visited:
                        visited.add(nbr)
                        stack.append((nbr, list(self.graph.neighbors(nbr))))
                else:
                    order.append(node)
                    stack.pop()
        return order

    def _run(self) -> None:
        order = self._finish_order()
        reverse = self.graph.reverse()
        assigned: set[str] = set()
        for start in reversed(order):
            if start in assigned:
                continue
            component: set[str] = set()
            stack = [start]
            assigned.add(start)
            while stack:
                node = stack.pop()
                component.add(node)
                for nbr in reverse.neighbors(node):
                    if nbr not in assigned:
                        assigned.add(nbr)
                        stack.append(nbr)
            index = len(self._components)
            self._components.append(component)
            for node in component:
                self._assignment[node] = index

    def components(self) -> list[set[str]]:
        return sorted(self._components, key=len, reverse=True)

    def component_of(self, node: str) -> set[str]:
        if not self.graph.has_node(node):
            raise Missing(f"node '{node}' is not in the graph")
        return set(self._components[self._assignment[node]])

    def count(self) -> int:
        return len(self._components)

    def is_strongly_connected(self) -> bool:
        return self.graph.node_count() > 0 and self.count() == 1

    def note(self) -> str:
        largest = max((len(c) for c in self._components), default=0)
        return (
            f"{self.count()} strongly connected component(s) by two passes, "
            f"largest holds {largest}; must agree with Tarjan on the partition"
        )
