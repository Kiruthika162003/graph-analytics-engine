"""Weakly connected components: a directed graph's islands when direction is ignored.

A directed graph has two notions of being in one piece, and they answer
different questions. Strong connectivity asks whether every node can reach
every other by following the arrows, which is the demanding version. Weak
connectivity asks only whether the nodes hang together at all, whether
the graph is one piece when every arrow is treated as a plain line. A
chain of one-way streets is weakly connected but not strongly, since you
can drive along it in one direction only; a set of islands with no roads
between them is neither. Weak components are what a person sees looking
at the drawing, and they are the right unit for questions like which
nodes could possibly influence one another in any direction, or how many
separate systems a dependency graph really describes. They are computed
by forgetting direction and taking ordinary connected components, which
the engine does by building the undirected shadow of the graph and
running union-find over its edges. The comparison against the strong
components is the interesting reading: a graph with few weak components
but many strong ones is a graph that is joined up on paper but full of
one-way passages, the shape where reachability is a real constraint,
while a graph whose weak and strong counts match is one where every
connection runs both ways in effect. The finder returns the weak
components, the component of a node, the count, and whether the graph is
weakly connected, refusing an undirected graph because there the two
notions coincide and the plain components module is the honest tool. It
reports the weak count beside the strong count, because the gap between
them is the number of one-way barriers the direction imposes.
"""

from __future__ import annotations

from mesh.connectedcomponents import ConnectedComponents
from mesh.errors import Invalid, Missing
from mesh.graph import Graph
from mesh.tarjanscc import TarjanSCC


class WeaklyConnected:
    def __init__(self, graph: Graph) -> None:
        if not graph.directed:
            raise Invalid("weak connectivity is a directed-graph notion; use components")
        self.graph = graph
        shadow = Graph(directed=False)
        for n in graph.nodes():
            shadow.add_node(n)
        for u, v, w in graph.edges():
            shadow.add_edge(u, v, w)  # direction forgotten
        self._components = ConnectedComponents(shadow)

    def components(self) -> list[set[str]]:
        return self._components.components()

    def component_of(self, node: str) -> set[str]:
        if not self.graph.has_node(node):
            raise Missing(f"node '{node}' is not in the graph")
        return self._components.component_of(node)

    def count(self) -> int:
        return self._components.count()

    def is_weakly_connected(self) -> bool:
        return self._components.is_connected()

    def strong_count(self) -> int:
        return TarjanSCC(self.graph).count()

    def one_way_barriers(self) -> int:
        return self.strong_count() - self.count()

    def note(self) -> str:
        return (
            f"{self.count()} weak component(s) against {self.strong_count()} strong; "
            f"the gap of {self.one_way_barriers()} is the one-way barriers direction "
            "imposes on a graph that is joined up on paper"
        )
