"""Permutation graphs: a pair of positions is an edge when the permutation inverts them.

Write a permutation as a row of numbers and draw a line from each
position in the identity row to where the number landed; two lines
cross exactly when the permutation puts a larger number before a
smaller one, an inversion, and the permutation graph has an edge for
every crossing. The class is worth having because its hard problems
are sequence problems in disguise. A clique is a set of pairwise
crossing lines, which is a decreasing run of numbers, so the clique
number is the longest decreasing subsequence. An independent set is a
set of pairwise non-crossing lines, an increasing run, so the
independence number is the longest increasing subsequence, and
patience sorting finds it in n log n. The complement of a permutation
graph is the permutation graph of the reversed permutation, since
reversing turns every inversion into a non-inversion and back. The
engine builds the graph from a permutation of zero through n minus
one, computes both subsequence lengths by patience sorting on the
sequence and its negation, and checks them against Bron-Kerbosch and
the complement module, which is what ties the sequence view to the
graph view. A sequence that is not a permutation of zero through n
minus one is refused with the first missing or repeated value named.
"""

from __future__ import annotations

from bisect import bisect_left
from itertools import combinations

from mesh.bronkerbosch import BronKerbosch
from mesh.complement import Complement
from mesh.errors import Invalid
from mesh.graph import Graph


class PermutationGraph:
    def __init__(self, permutation: list[int]) -> None:
        self.permutation = list(permutation)
        self._check()
        self.graph = self._build()

    def _check(self) -> None:
        n = len(self.permutation)
        seen: set[int] = set()
        for value in self.permutation:
            if value in seen:
                raise Invalid(f"value {value} is repeated")
            seen.add(value)
        for expected in range(n):
            if expected not in seen:
                raise Invalid(f"value {expected} is missing")

    def _build(self) -> Graph:
        g = Graph()
        n = len(self.permutation)
        for i in range(n):
            g.add_node(str(i))
        for i, j in combinations(range(n), 2):
            if self.permutation[i] > self.permutation[j]:
                g.add_edge(str(i), str(j))
        return g

    @staticmethod
    def longest_increasing(sequence: list[int]) -> int:
        # patience sorting: tails[k] is the smallest tail of an increasing run of length k+1
        tails: list[int] = []
        for value in sequence:
            spot = bisect_left(tails, value)
            if spot == len(tails):
                tails.append(value)
            else:
                tails[spot] = value
        return len(tails)

    def independence_number(self) -> int:
        return self.longest_increasing(self.permutation)

    def clique_number(self) -> int:
        return self.longest_increasing([-v for v in self.permutation])

    def inversions(self) -> int:
        return self.graph.edge_count()

    def reversed_graph(self) -> Graph:
        return PermutationGraph(list(reversed(self.permutation))).graph

    def complement_is_the_reverse(self) -> bool:
        # the reverse permutation's graph relabels positions, so compare edge counts and
        # the pair structure through the reversed position map
        n = len(self.permutation)
        comp = Complement(self.graph).complement
        rev = self.reversed_graph()
        if comp.edge_count() != rev.edge_count():
            return False
        return all(
            comp.has_edge(str(i), str(j)) == rev.has_edge(str(n - 1 - j), str(n - 1 - i))
            for i, j in combinations(range(n), 2)
        )

    def numbers_agree(self) -> bool:
        return (
            self.clique_number() == BronKerbosch(self.graph).clique_number()
            and self.independence_number() == Complement(self.graph).independence_number()
        )

    def note(self) -> str:
        return (
            f"permutation of {len(self.permutation)} with {self.inversions()} inversion(s): "
            f"clique number {self.clique_number()} from the longest decreasing run, "
            f"independence number {self.independence_number()} from the longest increasing"
        )
