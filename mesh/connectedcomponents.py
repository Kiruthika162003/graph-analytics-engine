"""Connected components: the islands of an undirected graph, and which is giant.

A connected component of an undirected graph is a maximal set of nodes
every pair of which is joined by some path. The components partition the
graph into islands: within an island you can walk between any two nodes,
and between islands you cannot walk at all. Finding them answers the first
question anyone asks of a network, is it one piece or many, and how big is
the biggest piece. The computation is a sweep: union every edge's two
endpoints, then read off the groups, which is exactly what union-find was
built for and runs in almost linear time. An equivalent sweep does a
breadth-first flood from each not-yet-labelled node, labelling everything
the flood reaches as one component before moving to the next unlabelled
node; the two give the same partition, and this engine uses union-find so
the components stay correct even as edges are added incrementally rather
than known all at once. Real networks almost always show a giant component,
one island holding most of the nodes, with a scatter of small ones around
it, so the size of the largest component and the count of components
together say whether the graph is essentially whole or badly fragmented.
The finder returns the component of a node as the set it belongs to, lists
all components largest first, counts them, and gives the largest size and
the fraction of nodes it holds. It refuses to look up a node not in the
graph. It reports the component count and the giant component's share,
because a share near one is a connected network with a few stragglers while
a share far below one is a graph that has broken into pieces.
"""

from __future__ import annotations

from mesh.errors import Invalid, Missing
from mesh.graph import Graph
from mesh.unionfind import UnionFind


class ConnectedComponents:
    def __init__(self, graph: Graph) -> None:
        if graph.directed:
            raise Invalid(
                "connected components are for undirected graphs; a directed "
                "graph has weakly and strongly connected components instead"
            )
        self.graph = graph
        self._uf = UnionFind()
        for node in graph.nodes():
            self._uf.add(node)
        for u, v, _w in graph.edges():
            self._uf.union(u, v)

    def component_of(self, node: str) -> set[str]:
        if not self.graph.has_node(node):
            raise Missing(f"node '{node}' is not in the graph")
        root = self._uf.find(node)
        return {n for n in self.graph.nodes() if self._uf.find(n) == root}

    def components(self) -> list[set[str]]:
        groups: dict[str, set[str]] = {}
        for node in self.graph.nodes():
            groups.setdefault(self._uf.find(node), set()).add(node)
        return sorted(groups.values(), key=len, reverse=True)

    def count(self) -> int:
        return self._uf.group_count()

    def is_connected(self) -> bool:
        return self.graph.node_count() > 0 and self.count() == 1

    def largest_size(self) -> int:
        comps = self.components()
        return len(comps[0]) if comps else 0

    def giant_share(self) -> float:
        n = self.graph.node_count()
        return self.largest_size() / n if n else 0.0

    def note(self) -> str:
        return (
            f"{self.count()} component(s), giant holds "
            f"{self.giant_share() * 100:.0f}% of nodes; a share near 100 is a "
            "whole network with stragglers, far below is a fragmented graph"
        )
