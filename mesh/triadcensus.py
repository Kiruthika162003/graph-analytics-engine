"""Triad census: every three nodes of a digraph sorted into one of sixteen shapes.

Three nodes and the arcs among them form one of sixteen shapes up to
relabeling, and counting how many triples fall in each is the triad
census of Holland and Leinhardt, the standard first reading of a
social network. The names encode the dyads: three digits for the
number of mutual, asymmetric, and null pairs, and a letter where the
digits leave more than one shape. 003 is empty, 300 is complete and
mutual, 030T is the transitive triple and 030C the cycle, 021D is one
node pointing at two, 021U two nodes pointing at one, 021C a chain,
111D is a mutual pair pointed at from outside, 111U a mutual pair
pointing out, 120D, 120U, and 120C are the 021 shapes with the free
pair made mutual, and 012, 102, 201, and 210 are the shapes the digits
alone determine. The engine enumerates every triple, reads the three
dyads, and settles the letter from where the asymmetric arcs meet. The
counts always sum to n choose 3, a transitive tournament is all 030T,
a directed triangle is 030C, and an undirected graph, whose every pair
is mutual or null, lands entirely in 003, 102, 201, and 300, which is
its own four-class census. The note lists the non-empty classes in
order of size.
"""

from __future__ import annotations

from itertools import combinations
from math import comb

from mesh.graph import Graph

TRIAD_NAMES = [
    "003", "012", "102", "021D", "021U", "021C", "111D", "111U",
    "030T", "030C", "201", "120D", "120U", "120C", "210", "300",
]


class TriadCensus:
    def __init__(self, graph: Graph) -> None:
        self.graph = graph
        self.counts = dict.fromkeys(TRIAD_NAMES, 0)
        for triple in combinations(graph.nodes(), 3):
            self.counts[self.classify(*triple)] += 1

    def _dyad(self, a: str, b: str) -> str:
        ab, ba = self.graph.has_edge(a, b), self.graph.has_edge(b, a)
        if ab and ba:
            return "M"
        if ab or ba:
            return "A"
        return "N"

    def classify(self, a: str, b: str, c: str) -> str:
        nodes = (a, b, c)
        dyads = [self._dyad(x, y) for x, y in combinations(nodes, 2)]
        m, asym, null = dyads.count("M"), dyads.count("A"), dyads.count("N")
        code = f"{m}{asym}{null}"
        if code in ("003", "012", "102", "201", "210", "300"):
            return code
        # the asymmetric arcs, as (tail, head), settle the letter
        arcs = [
            (x, y)
            for x, y in combinations(nodes, 2)
            for x, y in ((x, y), (y, x))
            if self.graph.has_edge(x, y) and not self.graph.has_edge(y, x)
        ]
        arcs = list(dict.fromkeys(arcs))
        if code == "030":
            tails = {t for t, _h in arcs}
            return "030C" if len(tails) == 3 else "030T"
        if code == "111":
            mutual = next(
                {x, y} for x, y in combinations(nodes, 2) if self._dyad(x, y) == "M"
            )
            tail, _head = arcs[0]
            return "111U" if tail in mutual else "111D"
        # 021 and 120: two asymmetric arcs meet at one node or form a chain
        tails = [t for t, _h in arcs]
        heads = [h for _t, h in arcs]
        if tails[0] == tails[1]:
            letter = "D"
        elif heads[0] == heads[1]:
            letter = "U"
        else:
            letter = "C"
        return f"{code}{letter}"

    def total(self) -> int:
        return sum(self.counts.values())

    def sums_to_choose_three(self) -> bool:
        return self.total() == comb(self.graph.node_count(), 3)

    def transitivity_share(self) -> float:
        # among triples with three arcs, the share that are transitive rather than cyclic
        three = self.counts["030T"] + self.counts["030C"]
        return self.counts["030T"] / three if three else 0.0

    def note(self) -> str:
        present = sorted(
            ((name, n) for name, n in self.counts.items() if n), key=lambda p: (-p[1], p[0])
        )
        listing = ", ".join(f"{name} x{n}" for name, n in present) or "no triples"
        return f"{self.total()} triple(s): {listing}"
