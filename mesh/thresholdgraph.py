"""Threshold graphs: built one node at a time, each isolated or dominating.

A threshold graph grows from a single node by adding nodes one at a
time, each either isolated, with no edge to anything so far, or
dominating, with an edge to everything so far. The creation sequence is
the whole graph, and the recognizer runs it backwards: while nodes
remain, find one that is isolated or dominating among the rest, record
which and remove it. If at some point neither kind exists, the graph is
not threshold, and the remaining nodes hold one of the three forbidden
shapes, a square, a path on four, or two edges with nothing between
them. Threshold graphs are exactly the graphs that are both split and
cographs, and the module checks that identity on demand against those
two recognizers, which is a cheap way to catch a wrong verdict in any of
the three. The hard numbers are read straight off the sequence: the
dominating nodes form a clique that any earlier node can join, so the
clique number is the dominating count plus one, and the isolated nodes,
with the first node counted among them, are pairwise non-adjacent and
form a largest independent set. Both are checked against Bron-Kerbosch
and the complement's independence number. A directed graph is refused,
and the note states the creation sequence in the order it grew.
"""

from __future__ import annotations

from mesh.bronkerbosch import BronKerbosch
from mesh.cograph import Cograph
from mesh.complement import Complement
from mesh.errors import Invalid
from mesh.graph import Graph
from mesh.splitgraph import SplitGraph

ISOLATED = "isolated"
DOMINATING = "dominating"


class ThresholdGraph:
    def __init__(self, graph: Graph) -> None:
        if graph.directed:
            raise Invalid("threshold graphs are undirected")
        self.graph = graph
        self.sequence: list[tuple[str, str]] = []
        self.stuck: set[str] = set()
        self.is_threshold = self._peel()

    def _peel(self) -> bool:
        remaining = set(self.graph.nodes())
        removed: list[tuple[str, str]] = []
        while remaining:
            found = None
            for node in sorted(remaining):
                inside = sum(1 for m in self.graph.neighbors(node) if m in remaining)
                if inside == 0:
                    found = (node, ISOLATED)
                    break
                if inside == len(remaining) - 1:
                    found = (node, DOMINATING)
                    break
            if found is None:
                self.stuck = set(remaining)
                return False
            removed.append(found)
            remaining.discard(found[0])
        self.sequence = list(reversed(removed))
        return True

    def clique_number(self) -> int:
        self._require()
        if not self.sequence:
            return 0
        return 1 + sum(1 for _n, kind in self.sequence[1:] if kind == DOMINATING)

    def independence_number(self) -> int:
        self._require()
        if not self.sequence:
            return 0
        return 1 + sum(1 for _n, kind in self.sequence[1:] if kind == ISOLATED)

    def _require(self) -> None:
        if not self.is_threshold:
            raise Invalid("the sequence numbers are only exact on a threshold graph")

    def numbers_agree(self) -> bool:
        return (
            self.clique_number() == BronKerbosch(self.graph).clique_number()
            and self.independence_number() == Complement(self.graph).independence_number()
        )

    def matches_split_and_cograph(self) -> bool:
        both = SplitGraph(self.graph).is_split and Cograph(self.graph).is_cograph
        return both == self.is_threshold

    def note(self) -> str:
        if not self.is_threshold:
            return (
                f"not a threshold graph: {len(self.stuck)} node(s) remain with none isolated "
                "or dominating among them"
            )
        grown = ", ".join(f"{n}:{kind[0]}" for n, kind in self.sequence)
        return (
            f"threshold graph grown as [{grown}]: clique number {self.clique_number()}, "
            f"independence number {self.independence_number()}"
        )
