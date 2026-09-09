"""Graph burning: how many rounds a fire needs to reach every node when it can also be lit anew.

Burning models a rumor that spreads to every neighbor each round while
a new source is lit somewhere each round too. Round one lights one
node. In each later round every burning node's neighbors catch, and
then one more unburnt node is lit. The burning number is the fewest
rounds that burn everything, and it is small: a path on n nodes burns
in the ceiling of the square root of n rounds, a star in two, and a
complete graph in two, since the first node reaches everyone in round
two. Bonato, Janssen, and Roshanbin conjectured that every connected
graph burns in at most ceiling of root n rounds, which holds on every
tree tested here. A sequence of k sources burns the graph exactly when
every node is within distance k minus i of the i-th source for some
i, counting the first source as i equal to one, which is the check
the module runs on any proposed sequence. The exact number comes from
trying source sequences by length, and the greedy heuristic picks
each new source as the unburnt node that would newly cover the most
nodes with its remaining reach, which is never below the exact
number. The engine gives both, verifies the sequences, and reports
the gap. A disconnected graph still burns, since each round lights a
new source, though it can take longer than root n; a directed graph
is refused.
"""

from __future__ import annotations

from collections import deque
from itertools import permutations
from math import ceil, sqrt

from mesh.errors import Invalid
from mesh.graph import Graph


class Burning:
    def __init__(self, graph: Graph) -> None:
        if graph.directed:
            raise Invalid("burning spreads along undirected edges")
        self.graph = graph
        self.nodes = graph.nodes()
        self.table = {n: self._hops(n) for n in self.nodes}

    def _hops(self, start: str) -> dict[str, int]:
        dist = {start: 0}
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for m in self.graph.neighbors(node):
                if m not in dist:
                    dist[m] = dist[node] + 1
                    queue.append(m)
        return dist

    def burns(self, sources: list[str]) -> bool:
        k = len(sources)
        for node in self.nodes:
            reached = False
            for i, src in enumerate(sources, start=1):
                if self.table[src].get(node, k + 1) <= k - i:
                    reached = True
                    break
            if not reached:
                return False
        return True

    def exact(self) -> list[str]:
        if not self.nodes:
            return []
        if len(self.nodes) > 12:
            raise Invalid("the exact burning number tries sequences; keep it to twelve nodes")
        for k in range(1, len(self.nodes) + 1):
            for sources in permutations(self.nodes, k):
                if self.burns(list(sources)):
                    return list(sources)
        return list(self.nodes)

    def greedy(self, rounds: int | None = None) -> list[str]:
        # lay out the sources in order of reach: the first has the longest, so pick
        # each source to cover the most still-uncovered nodes with its reach
        if not self.nodes:
            return []
        k = rounds if rounds is not None else 1
        while True:
            # the list must be exactly k long, since each source's reach is k minus its
            # position; an early return of a shorter list shrank every reach and failed
            chosen: list[str] = []
            covered: set[str] = set()
            for i in range(1, k + 1):
                reach = k - i
                candidates = sorted(n for n in self.nodes if n not in chosen)
                if not candidates:
                    break
                best = max(
                    candidates,
                    key=lambda n: len(
                        {m for m, d in self.table[n].items() if d <= reach} - covered
                    ),
                )
                chosen.append(best)
                covered |= {m for m, d in self.table[best].items() if d <= reach}
            if len(covered) == len(self.nodes) or rounds is not None:
                return chosen
            k += 1

    def conjecture_bound(self) -> int:
        return ceil(sqrt(len(self.nodes)))

    def note(self) -> str:
        exact = self.exact()
        greedy = self.greedy()
        return (
            f"burns in {len(exact)} round(s) from {exact}; greedy took {len(greedy)}, "
            f"root-n bound {self.conjecture_bound()}"
        )
