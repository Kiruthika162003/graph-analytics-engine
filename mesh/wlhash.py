"""Weisfeiler-Lehman hash: a fingerprint of a graph's shape, blind to its names.

Comparing two graphs by isomorphism search is expensive; comparing a
million graphs pairwise is impossible. A graph hash sidesteps the pairs
by giving each graph a fixed-length fingerprint computed from its
structure alone, such that isomorphic graphs always get the same
fingerprint, so equal hashes gather candidates and only those need the
full check. The Weisfeiler-Lehman hash builds the fingerprint from color
refinement. Every node starts with a color derived from its degree, then
for a fixed number of rounds each node's color becomes a hash of its own
color together with the sorted multiset of its neighbors' colors, the
same refinement the isomorphism decider uses. After the rounds the
multiset of all node colors, sorted, is hashed once more to produce the
graph's fingerprint. Because every step depends only on structure and
never on a node's name or the order nodes were inserted, relabelling a
graph cannot change its hash, which is the guarantee. The converse does
not hold and the engine says so: two non-isomorphic graphs can hash
alike whenever refinement cannot separate them, regular graphs of the
same degree being the classic case, so an equal hash is a candidate
match to be confirmed, not a verdict. More rounds distinguish more
graphs up to a point, after which the refinement has stabilized and
further rounds change nothing. The hasher takes the round count,
returns the fingerprint as a hex digest, exposes the per-node colors so
the same machinery can fingerprint neighborhoods, and reports how many
distinct node colors the refinement reached, because a graph whose
nodes all share one color is one the hash cannot tell from any other
regular graph of that degree and size.
"""

from __future__ import annotations

import hashlib
from collections import Counter

from mesh.errors import Invalid
from mesh.graph import Graph


class WLHash:
    def __init__(self, graph: Graph, rounds: int = 3) -> None:
        if rounds < 0:
            raise Invalid("the round count cannot be negative")
        self.graph = graph
        self.rounds = rounds
        self.color: dict[str, str] = self._refine()
        self.digest = self._digest()

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    def _refine(self) -> dict[str, str]:
        # the starting color is the degree, so names never enter
        color = {n: self._hash(f"deg:{self.graph.degree(n)}") for n in self.graph.nodes()}
        for _ in range(self.rounds):
            new = {}
            for n in self.graph.nodes():
                nbrs = sorted(color[m] for m in self.graph.neighbors(n))
                new[n] = self._hash(color[n] + "|" + ",".join(nbrs))
            color = new
        return color

    def _digest(self) -> str:
        bag = ",".join(sorted(self.color.values()))
        return self._hash(f"{self.graph.node_count()}:{self.graph.edge_count()}:{bag}")

    def distinct_colors(self) -> int:
        return len(set(self.color.values()))

    def color_histogram(self) -> dict[str, int]:
        return dict(Counter(self.color.values()))

    def note(self) -> str:
        return (
            f"digest {self.digest} with {self.distinct_colors()} distinct color(s) over "
            f"{self.graph.node_count()} node(s) after {self.rounds} round(s); one color "
            "for every node is a graph the hash cannot tell from any regular graph"
        )
