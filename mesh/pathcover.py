"""Path cover: the fewest chains that together touch every node of a DAG.

Given tasks with precedence, how few workers does it take if each
worker handles a chain of tasks one after another? Given train trips
with departure and arrival times, how few trains? Both ask for a
minimum path cover of a DAG: a set of node-disjoint directed paths, some
possibly a single node, that together include every node, as few paths
as possible. The answer has a clean reduction. Split every node into a
left copy and a right copy, draw an edge from the left copy of u to the
right copy of v for each DAG edge u to v, and find a maximum matching in
that bipartite graph. Each matched edge joins two consecutive nodes on
one path, so every matched edge reduces the path count by one, and the
minimum cover has the node count minus the matching size paths. The
reduction is exact because the paths are node-disjoint, which is what
the bipartite copies enforce: a left copy matched once means the node
has one successor on its path, a right copy matched once means one
predecessor. The engine runs Hopcroft-Karp on the split graph,
assembles the paths by following matched edges from nodes whose right
copy is unmatched, the path starts, and returns them. Dilworth's theorem
is the cousin: the minimum number of chains covering a partial order
equals its largest antichain, but that is over the transitive closure,
where a chain may skip intermediate nodes, while the path cover here
uses only the given edges; the engine computes both so the reader sees
the difference between covering with direct steps and covering with
any comparable pair. It refuses a cyclic graph, and reports the path
count beside the node count, because a cover of nearly n paths is a
graph with almost no precedence and a cover of one path is a total
order.
"""

from __future__ import annotations

from mesh.errors import Invalid
from mesh.graph import Graph
from mesh.hopcroftkarp import HopcroftKarp
from mesh.toposort import TopologicalSort
from mesh.transitiveclosure import TransitiveClosure


class PathCover:
    def __init__(self, graph: Graph) -> None:
        if not graph.directed:
            raise Invalid("a path cover is defined on a directed acyclic graph")
        TopologicalSort(graph).order()  # refuses a cycle
        self.graph = graph
        self.nodes = graph.nodes()
        self.successor = self._match(graph)
        self.paths = self._assemble()

    def _match(self, graph: Graph) -> dict[str, str]:
        left = [f"{n}\0L" for n in self.nodes]
        right = [f"{n}\0R" for n in self.nodes]
        hk = HopcroftKarp(left, right)
        for u, v, _w in graph.edges():
            hk.add_edge(f"{u}\0L", f"{v}\0R")
        hk.solve()
        return {u.split("\0")[0]: v.split("\0")[0] for u, v in hk.matching().items()}

    def _assemble(self) -> list[list[str]]:
        has_predecessor = set(self.successor.values())
        paths: list[list[str]] = []
        for start in sorted(self.nodes):
            if start in has_predecessor:
                continue  # a right copy that is matched is not a path start
            chain = [start]
            while chain[-1] in self.successor:
                chain.append(self.successor[chain[-1]])
            paths.append(chain)
        return paths

    def count(self) -> int:
        return len(self.paths)

    def covers_every_node_once(self) -> bool:
        seen = [n for p in self.paths for n in p]
        return sorted(seen) == sorted(self.nodes)

    def dilworth_count(self) -> int:
        # chains over the transitive closure: any comparable pair may be a step
        closure = TransitiveClosure(self.graph)
        skipping = Graph(directed=True)
        for n in self.nodes:
            skipping.add_node(n)
        for u in self.nodes:
            for v in closure.reach[u]:
                if u != v:
                    skipping.add_edge(u, v)
        return PathCover(skipping).count() if skipping.edge_count() else len(self.nodes)

    def note(self) -> str:
        return (
            f"{self.count()} path(s) cover {len(self.nodes)} node(s) with direct steps, "
            f"{self.dilworth_count()} chain(s) if steps may skip; one path is a total "
            "order, nearly n is almost no precedence"
        )
