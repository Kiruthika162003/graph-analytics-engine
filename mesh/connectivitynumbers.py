"""Connectivity numbers: how many edges, and how many nodes, must fail to split the graph.

Two numbers summarize a network's resilience to failure. The edge
connectivity is the fewest edges whose removal disconnects the graph,
and the vertex connectivity is the fewest nodes. A ring has both equal
to two, a tree has both equal to one, a complete graph on n nodes has
edge connectivity n minus one and, by convention, vertex connectivity
n minus one as well since no removal of fewer than n minus one nodes
can disconnect what remains. The two are computed by machinery this
engine already has. Edge connectivity is the global minimum cut with
every edge weighted one, which Stoer-Wagner finds without choosing a
pair. Vertex connectivity is the minimum over pairs of non-adjacent
nodes of the node-disjoint path count, by Menger, and the engine takes
that minimum over every non-adjacent pair, which is quadratic in the
node count times a flow each and is stated as the cost. Whitney's
inequality ties the numbers together: vertex connectivity is at most
edge connectivity, which is at most the minimum degree, and the engine
asserts that chain on every graph it measures, because a violation
would mean one of the two underlying algorithms is wrong. A graph is
k-connected when its vertex connectivity is at least k, and the two
numbers say which failures it survives: edge connectivity three means
any two cable cuts leave it whole, vertex connectivity one means a
single router is enough to split it. The measure returns both numbers
and the minimum degree, checks Whitney's chain, and reports the gap
between vertex and edge connectivity, because a gap is a graph whose
edges are redundant but whose nodes are not, cables that all run
through one box.
"""

from __future__ import annotations

from itertools import combinations

from mesh.connectedcomponents import ConnectedComponents
from mesh.errors import Invalid
from mesh.graph import Graph
from mesh.menger import Menger
from mesh.stoerwagner import StoerWagner


class ConnectivityNumbers:
    def __init__(self, graph: Graph) -> None:
        if graph.directed:
            raise Invalid("these connectivity numbers are for undirected graphs")
        if graph.node_count() < 2:
            raise Invalid("connectivity needs at least two nodes to separate")
        self.graph = graph
        self.min_degree = min(graph.degree(n) for n in graph.nodes())
        self.edge_connectivity = self._edge()
        self.vertex_connectivity = self._vertex()

    def _edge(self) -> int:
        if not ConnectedComponents(self.graph).is_connected():
            return 0
        unit = Graph()
        for n in self.graph.nodes():
            unit.add_node(n)
        for u, v, _w in self.graph.edges():
            unit.add_edge(u, v, 1.0)
        return int(StoerWagner(unit).weight)

    def _vertex(self) -> int:
        if not ConnectedComponents(self.graph).is_connected():
            return 0
        nodes = self.graph.nodes()
        pairs = [(a, b) for a, b in combinations(nodes, 2) if not self.graph.has_edge(a, b)]
        if not pairs:
            return len(nodes) - 1  # complete: nothing short of n-1 removals splits it
        # the fewest node-disjoint routes between any non-adjacent pair
        return min(Menger(self.graph, a, b).node_disjoint for a, b in pairs)

    def whitney_holds(self) -> bool:
        return self.vertex_connectivity <= self.edge_connectivity <= self.min_degree

    def is_k_connected(self, k: int) -> bool:
        return self.vertex_connectivity >= k

    def note(self) -> str:
        gap = self.edge_connectivity - self.vertex_connectivity
        return (
            f"vertex {self.vertex_connectivity} <= edge {self.edge_connectivity} <= "
            f"min degree {self.min_degree}, Whitney holds: {self.whitney_holds()}; a gap of "
            f"{gap} is redundant cables all running through one box"
        )
