"""Ego networks and structural holes: the world as one node sees it, and how much it brokers.

An ego network is a node, everyone within some hops of it, and the
edges among them; it is the view a person has of their own circle
and the unit most social surveys collect. The engine extracts the
ego network at any radius, with or without the ego itself, and reads
its density, which is the clustering coefficient at radius one when
the ego is dropped. Burt's constraint measures the opposite of
brokerage: for each neighbor j of ego i, the share of i's ties that
go to j, directly or through mutual contacts, squared and summed. A
node whose contacts all know each other is heavily constrained and
brokers nothing; a node whose contacts are strangers to one another
sits in structural holes and scores low. The formula uses p_ij, the
fraction of i's tie weight that goes to j, and adds for each mutual
contact q the product p_iq times p_qj, so a redundant tie counts
twice. Closed forms pin it: the hub of a star has constraint 1 over
the leaf count, since its contacts are all strangers, a node of a
complete graph on n has constraint that rises toward one as n
shrinks, and an isolated node has no ties and is refused. Effective
size, the number of non-redundant contacts, is reported beside it
as the degree minus the mean number of ties each contact has to
other contacts, which is Burt's other reading of the same idea.
"""

from __future__ import annotations

from collections import deque

from mesh.errors import Invalid
from mesh.graph import Graph


class Neighborhood:
    def __init__(self, graph: Graph) -> None:
        if graph.directed:
            raise Invalid("ego networks and constraint are read on an undirected graph")
        self.graph = graph

    def ego(self, node: str, radius: int = 1, keep_center: bool = True) -> Graph:
        if node not in self.graph.nodes():
            raise Invalid(f"'{node}' is not a node of the graph")
        if radius < 0:
            raise Invalid("the radius cannot be negative")
        dist = {node: 0}
        queue = deque([node])
        while queue:
            here = queue.popleft()
            if dist[here] == radius:
                continue
            for other in self.graph.neighbors(here):
                if other not in dist:
                    dist[other] = dist[here] + 1
                    queue.append(other)
        inside = set(dist) - ({node} if not keep_center else set())
        g = Graph()
        for n in self.graph.nodes():
            if n in inside:
                g.add_node(n)
        for u, v, w in self.graph.edges():
            if u in inside and v in inside:
                g.add_edge(u, v, w)
        return g

    @staticmethod
    def density(g: Graph) -> float:
        n = g.node_count()
        return g.edge_count() / (n * (n - 1) / 2) if n > 1 else 0.0

    def _share(self, i: str, j: str) -> float:
        total = sum(self.graph.neighbors(i).values())
        return self.graph.weight(i, j) / total if total and self.graph.has_edge(i, j) else 0.0

    def constraint(self, node: str) -> float:
        if node not in self.graph.nodes():
            raise Invalid(f"'{node}' is not a node of the graph")
        contacts = list(self.graph.neighbors(node))
        if not contacts:
            raise Invalid(f"'{node}' has no ties, so constraint is undefined")
        total = 0.0
        for j in contacts:
            through = sum(
                self._share(node, q) * self._share(q, j) for q in contacts if q != j
            )
            total += (self._share(node, j) + through) ** 2
        return total

    def effective_size(self, node: str) -> float:
        contacts = list(self.graph.neighbors(node))
        if not contacts:
            return 0.0
        redundant = sum(
            sum(1 for q in contacts if q != j and self.graph.has_edge(j, q)) for j in contacts
        ) / len(contacts)
        return len(contacts) - redundant

    def note(self, node: str) -> str:
        circle = self.ego(node, 1, keep_center=False)
        density = self.density(circle)
        return (
            f"{node}: {circle.node_count()} contact(s) with density {density:.2f}, "
            f"constraint {self.constraint(node):.3f}, effective size "
            f"{self.effective_size(node):.2f}"
        )
