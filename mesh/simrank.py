"""SimRank: two nodes are similar if the nodes that point to them are similar.

Most similarity measures compare two nodes by what they directly share.
SimRank is recursive: it says two nodes are similar to the degree that
their in-neighbors are similar to each other, and it bottoms out with
every node perfectly similar to itself. The definition reads as an
equation. The similarity of a pair is a decay constant times the average
similarity over all pairs of their in-neighbors, one drawn from each,
and a node's similarity to itself is one. Two pages cited by the same
papers are similar; two pages cited by different papers that are
themselves cited together are still somewhat similar, one step removed,
and so on with the decay shrinking each step. The equation is solved by
iteration from a start where only the diagonal is one, and each round
recomputes every pair from the previous round's values; the decay
constant, conventionally below one, is what makes the iteration converge
and what keeps distant relationships from counting as much as direct
ones. A pair where either node has no in-neighbors has nothing to
average over and stays at zero, which is the honest reading: nothing
points at one of them, so there is no structural evidence of any kind.
The cost is quadratic in the node count per round times the in-degree
product per pair, which is why SimRank is a measure for modest graphs
and why partial and sampled versions exist for large ones. The engine
iterates to a tolerance, returns the similarity of any pair, the most
similar partners of a node, refuses a decay outside the open unit
interval, and reports the largest off-diagonal similarity, because a
value near one is two nodes whose neighborhoods are near copies of each
other, the redundancy SimRank exists to reveal.
"""

from __future__ import annotations

from mesh.errors import Invalid, Missing
from mesh.graph import Graph


class SimRank:
    def __init__(
        self, graph: Graph, decay: float = 0.8, tolerance: float = 1e-8, max_rounds: int = 100
    ) -> None:
        if not 0.0 < decay < 1.0:
            raise Invalid("decay must lie strictly between 0 and 1")
        if graph.node_count() == 0:
            raise Invalid("an empty graph has no pairs to compare")
        self.graph = graph
        self.decay = decay
        self.nodes = graph.nodes()
        self._in: dict[str, list[str]] = {n: [] for n in self.nodes}
        for u, v, _w in graph.edges():
            self._in[v].append(u)
            if not graph.directed:
                self._in[u].append(v)
        self.rounds = 0
        self.sim = self._iterate(tolerance, max_rounds)

    def _iterate(self, tolerance: float, max_rounds: int) -> dict[str, dict[str, float]]:
        sim = {a: {b: (1.0 if a == b else 0.0) for b in self.nodes} for a in self.nodes}
        for _ in range(max_rounds):
            self.rounds += 1
            new = {a: dict(row) for a, row in sim.items()}
            change = 0.0
            for a in self.nodes:
                for b in self.nodes:
                    if a == b:
                        continue
                    ins_a, ins_b = self._in[a], self._in[b]
                    if not ins_a or not ins_b:
                        value = 0.0  # nothing points at one of them
                    else:
                        total = sum(sim[x][y] for x in ins_a for y in ins_b)
                        value = self.decay * total / (len(ins_a) * len(ins_b))
                    change += abs(value - sim[a][b])
                    new[a][b] = value
            sim = new
            if change < tolerance:
                break
        return sim

    def similarity(self, a: str, b: str) -> float:
        for n in (a, b):
            if n not in self.sim:
                raise Missing(f"node '{n}' is not in the graph")
        return self.sim[a][b]

    def most_similar(self, node: str, k: int = 3) -> list[tuple[str, float]]:
        if node not in self.sim:
            raise Missing(f"node '{node}' is not in the graph")
        others = [(m, s) for m, s in self.sim[node].items() if m != node]
        return sorted(others, key=lambda kv: (-kv[1], kv[0]))[:k]

    def largest_off_diagonal(self) -> float:
        return max(
            (self.sim[a][b] for a in self.nodes for b in self.nodes if a != b), default=0.0
        )

    def note(self) -> str:
        return (
            f"largest similarity between distinct nodes {self.largest_off_diagonal():.3f} "
            f"after {self.rounds} round(s) at decay {self.decay}; near one is two "
            "neighborhoods that are near copies of each other"
        )
