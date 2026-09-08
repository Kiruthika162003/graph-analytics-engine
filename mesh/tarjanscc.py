"""Tarjan's SCC: strongly connected components in one DFS, by low-link values.

A strongly connected component of a directed graph is a maximal set of
nodes where every node can reach every other by following edge directions.
Reachability alone is one-way in a directed graph, so an SCC is the set of
nodes that are mutually reachable, and contracting each SCC to a point turns
any directed graph into a directed acyclic graph of components, the
condensation. Tarjan's algorithm finds all SCCs in a single depth-first
traversal, which is what makes it elegant where the naive approach would
run a reachability search from every node. It stamps each node with a
discovery index in visit order and tracks a low-link value: the smallest
index reachable from the node through the DFS subtree and at most one back
edge. As the DFS unwinds, a node whose low-link equals its own index is the
root of an SCC, because nothing in its subtree reached any earlier node, so
it and everything pushed onto the working stack after it form one component.
The stack is the key device: nodes are pushed as they are discovered and
popped only when their SCC root is finalized, so a node stays on the stack
exactly while it might still turn out to share a component with something
still being explored. Cross edges to nodes already assigned to a finished
SCC are ignored in the low-link update, which is what keeps components from
bleeding into one another. The finder returns the components, tells which
component a node belongs to, counts them, and reports whether the graph is
itself strongly connected, one component covering everything. It reports the
component count and the largest size, because a single giant SCC is a
graph full of cycles while many singletons is a graph that is nearly a DAG.
"""

from __future__ import annotations

from mesh.errors import Invalid, Missing
from mesh.graph import Graph


class TarjanSCC:
    def __init__(self, graph: Graph) -> None:
        if not graph.directed:
            raise Invalid("strongly connected components are defined for digraphs")
        self.graph = graph
        self._index = 0
        self._indices: dict[str, int] = {}
        self._low: dict[str, int] = {}
        self._on_stack: set[str] = set()
        self._stack: list[str] = []
        self._components: list[set[str]] = []
        self._run()

    def _run(self) -> None:
        for node in self.graph.nodes():
            if node not in self._indices:
                self._strongconnect(node)

    def _strongconnect(self, start: str) -> None:
        # an explicit work stack simulates the recursion for deep graphs
        work: list[tuple[str, list[str]]] = [(start, list(self.graph.neighbors(start)))]
        self._indices[start] = self._low[start] = self._index
        self._index += 1
        self._stack.append(start)
        self._on_stack.add(start)
        while work:
            node, pending = work[-1]
            if pending:
                nbr = pending.pop(0)
                if nbr not in self._indices:
                    self._indices[nbr] = self._low[nbr] = self._index
                    self._index += 1
                    self._stack.append(nbr)
                    self._on_stack.add(nbr)
                    work.append((nbr, list(self.graph.neighbors(nbr))))
                elif nbr in self._on_stack:
                    # a back or cross edge to a node still on the stack
                    self._low[node] = min(self._low[node], self._indices[nbr])
            else:
                if self._low[node] == self._indices[node]:
                    self._pop_component(node)
                work.pop()
                if work:
                    parent = work[-1][0]
                    self._low[parent] = min(self._low[parent], self._low[node])

    def _pop_component(self, root: str) -> None:
        component: set[str] = set()
        while True:
            member = self._stack.pop()
            self._on_stack.discard(member)
            component.add(member)
            if member == root:
                break
        self._components.append(component)

    def components(self) -> list[set[str]]:
        return sorted(self._components, key=len, reverse=True)

    def component_of(self, node: str) -> set[str]:
        if not self.graph.has_node(node):
            raise Missing(f"node '{node}' is not in the graph")
        for comp in self._components:
            if node in comp:
                return set(comp)
        raise Missing(f"node '{node}' was not assigned a component")

    def count(self) -> int:
        return len(self._components)

    def is_strongly_connected(self) -> bool:
        return self.graph.node_count() > 0 and self.count() == 1

    def note(self) -> str:
        largest = max((len(c) for c in self._components), default=0)
        return (
            f"{self.count()} strongly connected component(s), largest holds "
            f"{largest}; one giant SCC is a graph full of cycles, many "
            "singletons is nearly a DAG"
        )
