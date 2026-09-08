"""Steiner tree: connect the chosen terminals cheaply, borrowing other nodes if it helps.

A spanning tree must touch every node. A Steiner tree must touch only a
chosen subset, the terminals, and may pass through any other nodes it
finds useful as waypoints. Wiring five offices together across a city
grid is a Steiner problem: the offices must connect, the street corners
between them are optional. Finding the minimum Steiner tree is NP-hard,
one of the classic hard problems, so the honest tool is an approximation
with a guarantee. The metric closure heuristic gives a factor of two.
Compute the shortest-path distance between every pair of terminals,
build the complete graph on the terminals weighted by those distances,
take its minimum spanning tree, and expand each tree edge back into the
shortest path it stands for; the union of those paths is a connected
subgraph containing every terminal, and pruning it to a tree by taking a
spanning tree of the union and then repeatedly removing non-terminal
leaves gives the answer. The factor-two bound comes from walking around
an optimal Steiner tree: a closed tour visiting every terminal has
length at most twice the tree, and the minimum spanning tree of the
metric closure is no longer than that tour with its longest edge
dropped. On real inputs the heuristic is usually much closer than two,
and the engine measures rather than assumes: on small instances it
compares its answer against the true optimum found by trying every
subset of extra nodes. The builder returns the tree's edges and weight,
which non-terminal nodes it borrowed, refuses a terminal outside the
graph and fewer than two terminals, and reports the weight against the
lower bound of the metric-closure tree halved, because the ratio says
how much room the guarantee left and whether the borrowed waypoints
actually paid for themselves.
"""

from __future__ import annotations

from itertools import combinations, pairwise

from mesh.errors import Invalid, Missing, Unreachable
from mesh.floydwarshall import FloydWarshall
from mesh.graph import Graph
from mesh.kruskal import Kruskal


class SteinerTree:
    def __init__(self, graph: Graph, terminals: list[str]) -> None:
        if graph.directed:
            raise Invalid("this Steiner heuristic runs on an undirected graph")
        if len(set(terminals)) < 2:
            raise Invalid("a Steiner tree needs at least two distinct terminals")
        for t in terminals:
            if not graph.has_node(t):
                raise Missing(f"terminal '{t}' is not in the graph")
        self.graph = graph
        self.terminals = sorted(set(terminals))
        self._closure = FloydWarshall(graph)
        self.closure_tree_weight = 0.0
        self.edges: list[tuple[str, str, float]] = self._build()

    def _build(self) -> list[tuple[str, str, float]]:
        # the complete graph on terminals weighted by shortest-path distance
        metric = Graph()
        for t in self.terminals:
            metric.add_node(t)
        for a, b in combinations(self.terminals, 2):
            try:
                metric.add_edge(a, b, self._closure.distance(a, b))
            except Unreachable as exc:
                raise Unreachable(
                    f"terminals '{a}' and '{b}' cannot reach each other; no tree connects them"
                ) from exc
        closure_tree = Kruskal(metric)
        self.closure_tree_weight = closure_tree.total_weight()
        # expand each closure edge back into the real path it stands for
        union = Graph()
        for a, b, _w in closure_tree.edges():
            path = self._closure.path(a, b)
            for u, v in pairwise(path):
                union.add_node(u)
                union.add_node(v)
                if not union.has_edge(u, v):
                    union.add_edge(u, v, self.graph.weight(u, v))
        # span the union, then prune non-terminal leaves until none remain
        tree = Graph()
        for n in union.nodes():
            tree.add_node(n)
        for u, v, w in Kruskal(union).edges():
            tree.add_edge(u, v, w)
        terminal_set = set(self.terminals)
        pruned = True
        while pruned:
            pruned = False
            for n in tree.nodes():
                if n not in terminal_set and tree.degree(n) == 1:
                    leaf = n
                    nbr = next(iter(tree.neighbors(leaf)))
                    tree = self._without_leaf(tree, leaf, nbr)
                    pruned = True
                    break
        return tree.edges()

    @staticmethod
    def _without_leaf(tree: Graph, leaf: str, _nbr: str) -> Graph:
        out = Graph()
        for n in tree.nodes():
            if n != leaf:
                out.add_node(n)
        for u, v, w in tree.edges():
            if leaf not in (u, v):
                out.add_edge(u, v, w)
        return out

    def weight(self) -> float:
        return sum(w for _u, _v, w in self.edges)

    def borrowed(self) -> set[str]:
        touched = {n for u, v, _w in self.edges for n in (u, v)}
        return touched - set(self.terminals)

    def lower_bound(self) -> float:
        return self.closure_tree_weight / 2.0

    def note(self) -> str:
        return (
            f"Steiner tree of weight {self.weight()} on {len(self.terminals)} "
            f"terminal(s), borrowing {len(self.borrowed())} waypoint(s); at most twice "
            f"the optimum, and at least {self.lower_bound()} by the closure bound"
        )
