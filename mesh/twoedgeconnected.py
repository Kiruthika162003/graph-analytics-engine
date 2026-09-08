"""Two-edge-connected components: the pieces that survive any single cable cut.

A graph is two-edge-connected if removing any one edge leaves it
connected, which is the property a network needs to survive a single
link failure. Most graphs are not, but every graph decomposes into
maximal pieces that are: remove all the bridges, the edges whose loss
would disconnect something, and the components that remain are the
two-edge-connected components. Inside one of them any two nodes are
joined by two edge-disjoint paths, so no single edge cut can separate
them; between two of them runs exactly one bridge, or none. Contracting
each component to a point and keeping the bridges gives the bridge tree,
a tree whose edges are the graph's single points of failure laid bare,
and the number of its leaves is the minimum number of edges to add to
make the whole graph two-edge-connected, since pairing up the leaves
closes every bridge into a cycle. The decomposition runs in one pass:
find the bridges with the low-link search, then flood-fill the graph
while refusing to cross a bridge, labelling everything each flood
reaches as one component. The engine reuses the articulation module for
the bridges, so the two agree by construction rather than by luck. The
decomposer returns the components, the component of a node, the bridge
tree as a graph on component indices, and the leaf count of that tree,
which is the repair budget. It reports the component count against the
node count and the leaf count, because a graph that is one component is
already robust to any single cut, a graph of many singletons is a tree
with no redundancy anywhere, and the leaf count in between is how many
edges it would take to fix.
"""

from __future__ import annotations

from mesh.articulation import Articulation
from mesh.errors import Invalid, Missing
from mesh.graph import Graph


class TwoEdgeConnected:
    def __init__(self, graph: Graph) -> None:
        if graph.directed:
            raise Invalid("two-edge-connectivity is defined on undirected graphs")
        self.graph = graph
        self.bridges = Articulation(graph).bridges
        self._index: dict[str, int] = {}
        self.components: list[set[str]] = []
        self._flood()
        self.bridge_tree = self._bridge_tree()

    def _flood(self) -> None:
        # flood-fill without crossing a bridge; each flood is one component
        for start in self.graph.nodes():
            if start in self._index:
                continue
            idx = len(self.components)
            comp = {start}
            self._index[start] = idx
            stack = [start]
            while stack:
                u = stack.pop()
                for v in self.graph.neighbors(u):
                    if v in self._index or frozenset((u, v)) in self.bridges:
                        continue
                    self._index[v] = idx
                    comp.add(v)
                    stack.append(v)
            self.components.append(comp)

    def _bridge_tree(self) -> Graph:
        tree = Graph(directed=False)
        for i in range(len(self.components)):
            tree.add_node(str(i))
        for bridge in self.bridges:
            u, v = tuple(bridge)
            tree.add_edge(str(self._index[u]), str(self._index[v]))
        return tree

    def component_of(self, node: str) -> set[str]:
        if node not in self._index:
            raise Missing(f"node '{node}' is not in the graph")
        return set(self.components[self._index[node]])

    def is_two_edge_connected(self) -> bool:
        return self.graph.node_count() > 0 and len(self.components) == 1 and \
            len(self.graph.nodes()) == len(self.components[0])

    def leaf_count(self) -> int:
        return sum(1 for c in self.bridge_tree.nodes() if self.bridge_tree.degree(c) == 1)

    def edges_to_add(self) -> int:
        # pairing up the bridge tree's leaves closes every bridge into a cycle
        return (self.leaf_count() + 1) // 2

    def note(self) -> str:
        return (
            f"{len(self.components)} two-edge-connected component(s) over "
            f"{self.graph.node_count()} node(s), {len(self.bridges)} bridge(s), "
            f"{self.edges_to_add()} edge(s) to add for full redundancy"
        )
