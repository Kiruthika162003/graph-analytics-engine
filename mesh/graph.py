"""Graph: the adjacency-list core every algorithm in this engine reads from.

A graph is a set of nodes and a set of edges between them, and almost every
question this engine answers reduces to walking that structure. The
representation chosen here is the adjacency list: each node maps to the set
of its outgoing neighbors together with the weight of the edge to each. An
adjacency list costs space proportional to nodes plus edges, not nodes
squared, so it stays cheap on the sparse graphs that real networks almost
always are, where a node touches a handful of others out of millions. The
alternative, an adjacency matrix, answers does-an-edge-exist in constant
time but pays nodes-squared space whether the graph is dense or not, and
on a sparse graph that is mostly zeroes; this engine keeps the matrix for
the few algorithms that want it and defaults to the list.

The graph can be directed or undirected, and that choice is made once at
construction rather than per edge, because a graph that mixes the two is
almost always a bug rather than an intent. In an undirected graph adding an
edge adds it in both directions, so the neighbor relation is symmetric by
construction and cannot drift out of symmetry. Weights default to one, so
an unweighted graph is just a weighted graph whose weights all happen to be
one, and the shortest-path count of edges falls out of the same code that
sums weights. The graph refuses an edge to a node it does not hold, because
a silent auto-add would let a typo in a node name create a phantom node,
and it refuses a self-loop only where an algorithm cannot tolerate one,
leaving the graph itself permissive. It reports node and edge counts and
the degree of a node, the readings the handshake lemma and every degree
based metric rest on.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mesh.errors import Invalid, Missing


@dataclass
class Graph:
    directed: bool = False
    # node -> {neighbor: weight}
    _adjacency: dict[str, dict[str, float]] = field(default_factory=dict)
    _edge_count: int = 0

    def add_node(self, node: str) -> None:
        if node not in self._adjacency:
            self._adjacency[node] = {}

    def has_node(self, node: str) -> bool:
        return node in self._adjacency

    def add_edge(self, u: str, v: str, weight: float = 1.0) -> None:
        if u not in self._adjacency:
            raise Missing(f"node '{u}' is not in the graph; add it before its edges")
        if v not in self._adjacency:
            raise Missing(f"node '{v}' is not in the graph; add it before its edges")
        new_edge = v not in self._adjacency[u]
        self._adjacency[u][v] = weight
        if not self.directed:
            self._adjacency[v][u] = weight
        if new_edge:
            self._edge_count += 1

    def has_edge(self, u: str, v: str) -> bool:
        return u in self._adjacency and v in self._adjacency[u]

    def weight(self, u: str, v: str) -> float:
        if not self.has_edge(u, v):
            raise Missing(f"no edge from '{u}' to '{v}'")
        return self._adjacency[u][v]

    def neighbors(self, node: str) -> dict[str, float]:
        if node not in self._adjacency:
            raise Missing(f"node '{node}' is not in the graph")
        return dict(self._adjacency[node])

    def nodes(self) -> list[str]:
        return list(self._adjacency)

    def edges(self) -> list[tuple[str, str, float]]:
        out: list[tuple[str, str, float]] = []
        seen: set[tuple[str, str]] = set()
        for u, nbrs in self._adjacency.items():
            for v, w in nbrs.items():
                if not self.directed and (v, u) in seen:
                    continue
                out.append((u, v, w))
                seen.add((u, v))
        return out

    def node_count(self) -> int:
        return len(self._adjacency)

    def edge_count(self) -> int:
        return self._edge_count

    def degree(self, node: str) -> int:
        if node not in self._adjacency:
            raise Missing(f"node '{node}' is not in the graph")
        return len(self._adjacency[node])

    def out_degree(self, node: str) -> int:
        return self.degree(node)

    def in_degree(self, node: str) -> int:
        if node not in self._adjacency:
            raise Missing(f"node '{node}' is not in the graph")
        return sum(1 for nbrs in self._adjacency.values() if node in nbrs)

    def reverse(self) -> Graph:
        if not self.directed:
            raise Invalid("an undirected graph is its own reverse")
        r = Graph(directed=True)
        for node in self._adjacency:
            r.add_node(node)
        for u, nbrs in self._adjacency.items():
            for v, w in nbrs.items():
                r.add_edge(v, u, w)
        return r
