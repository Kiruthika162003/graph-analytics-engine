"""Condensation: collapse every strongly connected component to a point and get a DAG.

Any directed graph, however tangled with cycles, has a clean skeleton
underneath. Group its nodes into strongly connected components, the
maximal sets where every node reaches every other, and contract each
component to a single super-node, keeping an edge between two super-nodes
whenever any edge ran between their members. The result is the
condensation, and it is always a directed acyclic graph, because a cycle
among super-nodes would mean their members all reach one another and they
would have been one component to begin with. That single fact is what
makes the condensation useful: everything that is easy on a DAG becomes
available on a cyclic graph by way of it. Topological order of the
condensation gives an order in which to process components so that every
edge points forward. The source components, those with no incoming edge,
are the places a walk can start from and reach everything downstream, and
the sink components, with no outgoing edge, are where all walks end. The
number of source components is exactly the minimum number of nodes from
which to start walks to cover the whole graph. The condensation also
answers reachability between two nodes in the original graph as
reachability between their components, which the DAG makes cheap. The
builder runs Tarjan's algorithm for the components, maps each node to its
component index, builds the contracted graph without duplicate or self
edges, and exposes the component of a node, the component DAG, the sources
and sinks, and a topological order of components. It refuses an
undirected graph, and reports the compression ratio, components over
nodes, because a ratio near one is a graph that was already nearly a DAG
and a small ratio is a graph whose cycles swallowed most of its nodes.
"""

from __future__ import annotations

from mesh.errors import Invalid, Missing
from mesh.graph import Graph
from mesh.tarjanscc import TarjanSCC
from mesh.toposort import TopologicalSort


class Condensation:
    def __init__(self, graph: Graph) -> None:
        if not graph.directed:
            raise Invalid("condensation is defined for a directed graph")
        self.graph = graph
        scc = TarjanSCC(graph)
        self.components: list[set[str]] = scc.components()
        self._index: dict[str, int] = {}
        for i, comp in enumerate(self.components):
            for node in comp:
                self._index[node] = i
        self.dag = Graph(directed=True)
        for i in range(len(self.components)):
            self.dag.add_node(str(i))
        for u, v, _w in graph.edges():
            cu, cv = self._index[u], self._index[v]
            if cu != cv:
                # one contracted edge per component pair, no self loops
                self.dag.add_edge(str(cu), str(cv))

    def component_of(self, node: str) -> int:
        if node not in self._index:
            raise Missing(f"node '{node}' is not in the graph")
        return self._index[node]

    def sources(self) -> list[int]:
        return [i for i in range(len(self.components)) if self.dag.in_degree(str(i)) == 0]

    def sinks(self) -> list[int]:
        return [i for i in range(len(self.components)) if self.dag.out_degree(str(i)) == 0]

    def order(self) -> list[int]:
        return [int(c) for c in TopologicalSort(self.dag).order()]

    def reaches(self, u: str, v: str) -> bool:
        cu, cv = self.component_of(u), self.component_of(v)
        if cu == cv:
            return True
        seen = {cu}
        stack = [cu]
        while stack:
            c = stack.pop()
            for nbr in self.dag.neighbors(str(c)):
                n = int(nbr)
                if n == cv:
                    return True
                if n not in seen:
                    seen.add(n)
                    stack.append(n)
        return False

    def compression(self) -> float:
        n = self.graph.node_count()
        return len(self.components) / n if n else 1.0

    def note(self) -> str:
        return (
            f"{len(self.components)} component(s) from {self.graph.node_count()} "
            f"node(s), compression {self.compression():.2f}, {len(self.sources())} "
            "source(s); a ratio near one was already nearly a DAG"
        )
