"""Tournaments: every pair plays once, and the arrow points at the loser.

A tournament is a directed graph with exactly one arc between every pair
of nodes. Round-robin results, pairwise preference votes, and dominance
hierarchies all produce them, and they have structure that general
digraphs lack. Every tournament has a Hamiltonian path, which Redei
proved and which an insertion argument builds: keep an ordered list of
players, and insert each new player just before the first one it beats,
or at the end if it beats none, and the list is a path along the arcs.
A king is a player who beats everyone directly or through one
intermediary, and the player with the most wins is always a king,
because anyone who beat the top scorer has to have lost to someone the
top scorer beat, or the scores would not add up. The score sequence,
the sorted win counts, is realizable by some tournament exactly when
Landau's condition holds: the first k scores sum to at least k choose 2
for every k, with equality at k equal to n. A tournament is transitive,
meaning acyclic, exactly when its scores are 0 through n minus 1, one
each. The engine checks the shape, computes the scores, tests Landau
against them, builds the Hamiltonian path and verifies it walks arcs,
finds every king, and reports whether the tournament is transitive. A
graph that is undirected, or that misses a pair, or that has both arcs
of a pair, is refused with the offending pair named.
"""

from __future__ import annotations

from itertools import combinations, pairwise

from mesh.errors import Invalid
from mesh.graph import Graph


class Tournament:
    def __init__(self, graph: Graph) -> None:
        if not graph.directed:
            raise Invalid("a tournament is a directed graph")
        self.graph = graph
        self._check_shape()
        self.scores = {n: graph.out_degree(n) for n in graph.nodes()}

    def _check_shape(self) -> None:
        for a, b in combinations(self.graph.nodes(), 2):
            forward = self.graph.has_edge(a, b)
            backward = self.graph.has_edge(b, a)
            if forward and backward:
                raise Invalid(f"'{a}' and '{b}' play each other twice")
            if not forward and not backward:
                raise Invalid(f"'{a}' and '{b}' never play")
        for n in self.graph.nodes():
            if self.graph.has_edge(n, n):
                raise Invalid(f"'{n}' plays itself")

    def score_sequence(self) -> list[int]:
        return sorted(self.scores.values())

    @staticmethod
    def landau_holds(sequence: list[int]) -> bool:
        ordered = sorted(sequence)
        n = len(ordered)
        running = 0
        for k, score in enumerate(ordered, start=1):
            running += score
            if running < k * (k - 1) // 2:
                return False
        return running == n * (n - 1) // 2

    def hamiltonian_path(self) -> list[str]:
        # insert each player before the first one it beats, else at the end
        order: list[str] = []
        for player in self.graph.nodes():
            for i, other in enumerate(order):
                if self.graph.has_edge(player, other):
                    order.insert(i, player)
                    break
            else:
                order.append(player)
        return order

    def path_walks_arcs(self) -> bool:
        order = self.hamiltonian_path()
        if len(order) != self.graph.node_count():
            return False
        return all(self.graph.has_edge(a, b) for a, b in pairwise(order))

    def kings(self) -> list[str]:
        found = []
        for player in self.graph.nodes():
            beaten = set(self.graph.neighbors(player))
            within_two = set(beaten)
            for mid in beaten:
                within_two |= set(self.graph.neighbors(mid))
            within_two.discard(player)
            if len(within_two) == self.graph.node_count() - 1:
                found.append(player)
        return found

    def top_scorer(self) -> str:
        return max(self.graph.nodes(), key=lambda n: (self.scores[n], n))

    def is_transitive(self) -> bool:
        n = self.graph.node_count()
        return self.score_sequence() == list(range(n))

    def note(self) -> str:
        return (
            f"tournament of {self.graph.node_count()}: scores {self.score_sequence()}, "
            f"{len(self.kings())} king(s), top scorer {self.top_scorer()}, "
            f"{'transitive' if self.is_transitive() else 'with an upset cycle'}"
        )
