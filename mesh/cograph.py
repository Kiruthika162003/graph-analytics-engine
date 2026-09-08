"""Cographs: graphs built by disjoint union and join alone, and read by a cotree.

A cograph is what you get by starting from single nodes and repeatedly
taking either the disjoint union of two graphs or their join, which is
the union plus every edge between the two sides. Equivalently, a
cograph has no induced path on four nodes. The recursive definition is
also the recognizer: a graph on two or more nodes is a cograph exactly
when it is disconnected and every component is a cograph, or its
complement is disconnected and every co-component is a cograph. A graph
that is connected with a connected complement is not a cograph, and a
smallest such graph is an induced P4, which the module finds and names
as the witness. The recursion tree is the cotree, and the hard numbers
fall out of it by two rules: at a union node the clique number is the
largest child's and the independence number is the children's sum, and
at a join node the two swap roles. The chromatic number follows the
clique number because cographs are perfect. The engine runs the
recursion on induced subgraphs it builds explicitly, keeps the cotree
depth and the operation count, and checks its three numbers against the
Bron-Kerbosch and chromatic modules, which is the point of having those
modules around. A directed graph is refused.
"""

from __future__ import annotations

from itertools import combinations, permutations

from mesh.errors import Invalid
from mesh.graph import Graph


class Cograph:
    def __init__(self, graph: Graph) -> None:
        if graph.directed:
            raise Invalid("cographs are undirected")
        self.graph = graph
        self.witness: tuple[str, ...] | None = None
        self.unions = 0
        self.joins = 0
        self.depth = 0
        self.clique_number = 0
        self.independence_number = 0
        self.is_cograph = self._recognize(set(graph.nodes()), 0)

    def _components(self, nodes: set[str], complement: bool) -> list[set[str]]:
        # connected pieces of the induced subgraph, or of its complement when asked
        remaining = set(nodes)
        parts: list[set[str]] = []
        while remaining:
            start = min(remaining)
            seen = {start}
            stack = [start]
            while stack:
                node = stack.pop()
                for other in nodes:
                    if other in seen:
                        continue
                    adjacent = self.graph.has_edge(node, other)
                    if adjacent != complement:
                        seen.add(other)
                        stack.append(other)
            parts.append(seen)
            remaining -= seen
        return parts

    def _recognize(self, nodes: set[str], level: int) -> bool:
        self.depth = max(self.depth, level)
        if len(nodes) <= 1:
            self.clique_number = len(nodes)
            self.independence_number = len(nodes)
            return True
        parts = self._components(nodes, complement=False)
        if len(parts) > 1:
            self.unions += 1
            return self._combine(parts, level, union=True)
        parts = self._components(nodes, complement=True)
        if len(parts) > 1:
            self.joins += 1
            return self._combine(parts, level, union=False)
        self.witness = self._find_p4(nodes)
        return False

    def _combine(self, parts: list[set[str]], level: int, union: bool) -> bool:
        cliques: list[int] = []
        independents: list[int] = []
        for part in parts:
            if not self._recognize(part, level + 1):
                return False
            cliques.append(self.clique_number)
            independents.append(self.independence_number)
        if union:
            self.clique_number = max(cliques)
            self.independence_number = sum(independents)
        else:
            self.clique_number = sum(cliques)
            self.independence_number = max(independents)
        return True

    def _find_p4(self, nodes: set[str]) -> tuple[str, ...] | None:
        for four in combinations(sorted(nodes), 4):
            for a, b, c, d in permutations(four):
                if a > d:
                    continue
                path_edges = (
                    self.graph.has_edge(a, b)
                    and self.graph.has_edge(b, c)
                    and self.graph.has_edge(c, d)
                )
                chords = (
                    self.graph.has_edge(a, c)
                    or self.graph.has_edge(a, d)
                    or self.graph.has_edge(b, d)
                )
                if path_edges and not chords:
                    return (a, b, c, d)
        return None

    def chromatic_number(self) -> int:
        if not self.is_cograph:
            raise Invalid("the cotree numbers are only exact on a cograph")
        return self.clique_number

    def note(self) -> str:
        if not self.is_cograph:
            return f"not a cograph: induced path {'-'.join(self.witness or ())} has no chord"
        return (
            f"cograph built by {self.unions} union(s) and {self.joins} join(s) to depth "
            f"{self.depth}: clique number {self.clique_number}, independence number "
            f"{self.independence_number}, chromatic number {self.chromatic_number()}"
        )
