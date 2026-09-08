"""Line graph: every edge becomes a node, and edge questions become node questions.

The line graph of a graph has one node for each original edge, with two
of them adjacent whenever the edges they stand for share an endpoint.
It is a change of viewpoint that converts a whole family of edge
problems into node problems the engine already solves. An edge coloring
of the original, no two edges at a node alike, is exactly a vertex
coloring of the line graph, so the chromatic index of a graph is the
chromatic number of its line graph. A matching, a set of edges sharing
no endpoint, is exactly an independent set of the line graph. A trail
that walks edges without repeating one is a path in the line graph. The
structure has a fixed shape worth checking: each edge-node's degree is
the sum of its two endpoints' degrees minus two, since it meets every
other edge at either end and neither of those counts itself, and the
line graph's edge count is the sum over original nodes of degree choose
two, one line-graph edge for every pair of edges at a node. A star
becomes a complete graph, because all its edges meet at the hub; a path
becomes a shorter path; a triangle becomes a triangle, and the triangle
is the one graph that is its own line graph among connected graphs
besides the single edge, which is why line graphs cannot be inverted
in general without extra information. The builder names each new node
by its edge's endpoints in sorted order, constructs the adjacency,
verifies the degree and edge-count identities, and reports the line
graph's size against the original's, because a line graph much larger
than its source is a graph with high-degree nodes, each contributing a
clique of edge-nodes, the case where the change of viewpoint costs
more than it saves.
"""

from __future__ import annotations

from itertools import combinations

from mesh.errors import Invalid, Missing
from mesh.graph import Graph


class LineGraph:
    def __init__(self, graph: Graph) -> None:
        if graph.directed:
            raise Invalid("this line graph is built from an undirected graph")
        self.graph = graph
        self.line = Graph()
        self._name: dict[frozenset[str], str] = {}
        for u, v, _w in graph.edges():
            name = "-".join(sorted((u, v)))
            self._name[frozenset((u, v))] = name
            self.line.add_node(name)
        # two edge-nodes are adjacent when their edges share an endpoint
        for node in graph.nodes():
            incident = [self._name[frozenset((node, m))] for m in graph.neighbors(node)]
            for a, b in combinations(sorted(incident), 2):
                if not self.line.has_edge(a, b):
                    self.line.add_edge(a, b)

    def node_for(self, u: str, v: str) -> str:
        key = frozenset((u, v))
        if key not in self._name:
            raise Missing(f"no edge between '{u}' and '{v}' in the original graph")
        return self._name[key]

    def degree_identity_holds(self) -> bool:
        for u, v, _w in self.graph.edges():
            expected = self.graph.degree(u) + self.graph.degree(v) - 2
            if self.line.degree(self.node_for(u, v)) != expected:
                return False
        return True

    def expected_edge_count(self) -> int:
        return sum(
            self.graph.degree(n) * (self.graph.degree(n) - 1) // 2 for n in self.graph.nodes()
        )

    def edge_count_identity_holds(self) -> bool:
        return self.line.edge_count() == self.expected_edge_count()

    def note(self) -> str:
        return (
            f"line graph of {self.line.node_count()} node(s) and {self.line.edge_count()} "
            f"edge(s) from {self.graph.node_count()} and {self.graph.edge_count()}; much "
            "larger than its source means hubs each contributing a clique of edge-nodes"
        )
