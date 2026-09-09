"""Relabeling: rename the nodes by a mapping, a prefix, a canonical order, or a function.

Names are the one thing a graph carries that its structure does not
need, and they are also where two graphs fail to line up: one source
calls a node "Ada" and another "ada", one numbers from zero and one
from one. Relabeling rewrites the names while keeping every edge and
weight, and it must refuse the two ways that can go wrong: a mapping
that sends two nodes to one name, which would merge them, and a
mapping that misses a node, which would lose it. Four ways of
choosing new names live here. An explicit mapping is checked for
completeness and injectivity. A prefix or suffix keeps names apart
when two graphs are merged. A canonical relabeling numbers the nodes
in the order of a fixed traversal, degree descending then name, so
that two graphs with the same shape and names in a different case or
order get the same numbering wherever their structure lets it. And a
function applied to every name handles case folding or trimming. The
result is a new graph; the input is untouched; and the inverse
mapping is returned beside the graph so the new names can be traced
back. The tests check that edges and weights survive every method,
that merging and dropping are refused by name, that relabeling twice
by a mapping and its inverse returns an equal graph, and that a
canonical relabeling of two equal graphs in different insertion
orders gives equal graphs.
"""

from __future__ import annotations

from collections.abc import Callable

from mesh.errors import Invalid
from mesh.graph import Graph


class Relabel:
    def __init__(self, graph: Graph) -> None:
        self.graph = graph

    def _apply(self, mapping: dict[str, str]) -> tuple[Graph, dict[str, str]]:
        nodes = self.graph.nodes()
        missing = [n for n in nodes if n not in mapping]
        if missing:
            raise Invalid(f"the mapping drops '{missing[0]}'")
        targets = [mapping[n] for n in nodes]
        if len(set(targets)) != len(targets):
            clash = next(t for t in targets if targets.count(t) > 1)
            merged = [n for n in nodes if mapping[n] == clash]
            raise Invalid(f"the mapping merges {merged[0]} and {merged[1]} into '{clash}'")
        out = Graph(directed=self.graph.directed)
        for n in nodes:
            out.add_node(mapping[n])
        for u, v, w in self.graph.edges():
            out.add_edge(mapping[u], mapping[v], w)
        inverse = {new: old for old, new in mapping.items() if old in set(nodes)}
        return out, inverse

    def by_mapping(self, mapping: dict[str, str]) -> tuple[Graph, dict[str, str]]:
        return self._apply(mapping)

    def by_function(self, fn: Callable[[str], str]) -> tuple[Graph, dict[str, str]]:
        return self._apply({n: fn(n) for n in self.graph.nodes()})

    def with_prefix(self, prefix: str) -> tuple[Graph, dict[str, str]]:
        return self._apply({n: prefix + n for n in self.graph.nodes()})

    def with_suffix(self, suffix: str) -> tuple[Graph, dict[str, str]]:
        return self._apply({n: n + suffix for n in self.graph.nodes()})

    def canonical(self) -> tuple[Graph, dict[str, str]]:
        # degree descending then name, so equal graphs in different orders number alike
        order = sorted(self.graph.nodes(), key=lambda n: (-self.graph.degree(n), n))
        return self._apply({n: str(i) for i, n in enumerate(order)})

    def note(self, inverse: dict[str, str]) -> str:
        changed = sum(1 for new, old in inverse.items() if new != old)
        return f"{len(inverse)} node(s) relabeled, {changed} name(s) actually changed"
