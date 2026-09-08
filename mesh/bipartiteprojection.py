"""Bipartite projection: collapse a two-mode network onto one of its sides.

Authors and the papers they wrote, people and the events they attended,
actors and the films they appeared in: two-mode networks whose edges
run only between the two kinds of node. The question people ask of them
is usually about one kind alone, which authors are collaborators, which
people move in the same circles, and the projection answers it. Two
nodes on the chosen side are joined in the projection whenever they
share at least one neighbor on the other side, and the edge carries the
count of shared neighbors as its weight, so coauthors of five papers
are joined more strongly than coauthors of one. Projection is lossy and
the engine says how. A paper with ten authors projects to forty-five
coauthor edges, a clique, so a single large event inflates the
projection's density far beyond what pairwise collaboration would
suggest; a projected edge cannot say whether two nodes shared one big
event or many small ones without the weight, and it cannot recover the
original two-mode structure at all. The Newman weighting divides each
shared neighbor's contribution by its degree minus one, so a paper with
two authors counts fully toward their tie and a paper with ten counts a
ninth, which corrects the clique inflation and is offered as an
alternative weight. The projector takes the side to keep, verifies the
graph is bipartite with that side as one part, builds the weighted
projection under either weighting, and reports the projection's edge
count against the edges a clique per shared neighbor would give,
because that ratio is how much the large events dominate.
"""

from __future__ import annotations

from itertools import combinations

from mesh.bipartite import Bipartite
from mesh.errors import Invalid, Missing
from mesh.graph import Graph


class BipartiteProjection:
    def __init__(self, graph: Graph, keep: list[str], newman: bool = False) -> None:
        if graph.directed:
            raise Invalid("projection is defined on an undirected two-mode graph")
        keep_set = set(keep)
        for n in keep_set:
            if not graph.has_node(n):
                raise Missing(f"node '{n}' is not in the graph")
        check = Bipartite(graph)
        if not check.is_bipartite:
            raise Invalid("the graph is not bipartite; there are no two modes to project")
        for u, v, _w in graph.edges():
            if (u in keep_set) == (v in keep_set):
                raise Invalid(f"edge {u}-{v} lies within one side; the kept side is not a part")
        self.graph = graph
        self.keep = sorted(keep_set)
        self.newman = newman
        self.other = [n for n in graph.nodes() if n not in keep_set]
        self.projection = self._project()

    def _project(self) -> Graph:
        p = Graph()
        for n in self.keep:
            p.add_node(n)
        weight: dict[frozenset[str], float] = {}
        for hub in self.other:
            members = sorted(self.graph.neighbors(hub))
            share = 1.0 / (len(members) - 1) if self.newman and len(members) > 1 else 1.0
            for a, b in combinations(members, 2):
                weight[frozenset((a, b))] = weight.get(frozenset((a, b)), 0.0) + share
        for pair, w in weight.items():
            a, b = sorted(pair)
            p.add_edge(a, b, w)
        return p

    def clique_edges(self) -> int:
        # what one clique per shared neighbor would produce if none overlapped
        total = 0
        for hub in self.other:
            k = self.graph.degree(hub)
            total += k * (k - 1) // 2
        return total

    def strongest_tie(self) -> tuple[str, str, float]:
        edges = self.projection.edges()
        if not edges:
            raise Invalid("the projection has no edges; nothing was shared")
        return max(edges, key=lambda e: (e[2], e[0], e[1]))

    def note(self) -> str:
        cliques = self.clique_edges() or 1
        ties = self.projection.edge_count()
        return (
            f"projected {ties} tie(s) among {len(self.keep)} node(s) from {cliques} "
            f"clique edge(s) ({ties / cliques:.2f} after overlap); large events inflate "
            "the density, the Newman weight corrects it"
        )
