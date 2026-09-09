"""Kernighan-Lin bisection: swap pairs across the cut, then keep the best prefix of swaps.

Splitting a graph into two equal halves with the fewest crossing
edges is NP-hard, and the Kernighan-Lin heuristic from 1970 remains
the standard first answer. Start from any balanced split. For each
node compute its D value, external edges minus internal edges, which
is the gain from moving it alone. A pass picks, repeatedly, the
unlocked pair one from each side whose swap gains the most, where the
gain is D(a) plus D(b) minus twice the weight between them, swaps
them provisionally, locks both, and updates every D. After the pass,
the best prefix of the swap sequence is kept, even if the total gain
went negative later, which is what lets the method climb out of local
minima that single swaps cannot escape. Passes repeat until a pass
gains nothing. The engine runs it with weights, reports the cut
before and after, counts passes, and checks the halves stay balanced
and the cut it reports equals a direct count. On a small graph the
exact minimum bisection comes from trying every balanced split, and
the tests confirm the heuristic never lands above it and reaches it on
the shapes it should: two cliques joined by a bridge separate at the
bridge, and a ladder cuts at a rung. An odd node count is refused
because the halves would not be equal, and a directed graph is
refused.
"""

from __future__ import annotations

from itertools import combinations

from mesh.errors import Invalid
from mesh.graph import Graph


class KernighanLin:
    def __init__(self, graph: Graph, left: set[str] | None = None) -> None:
        if graph.directed:
            raise Invalid("bisection is defined on an undirected graph")
        if graph.node_count() % 2:
            raise Invalid("an equal bisection needs an even node count")
        self.graph = graph
        nodes = graph.nodes()
        half = len(nodes) // 2
        self.left = set(left) if left is not None else set(nodes[:half])
        if len(self.left) != half or not self.left <= set(nodes):
            raise Invalid("the starting side must hold exactly half the nodes")
        self.right = set(nodes) - self.left
        self.initial_cut = self.cut_weight()
        self.passes = 0
        self._run()

    def _weight(self, a: str, b: str) -> float:
        return self.graph.weight(a, b) if self.graph.has_edge(a, b) else 0.0

    def cut_weight(self) -> float:
        return sum(w for u, v, w in self.graph.edges() if (u in self.left) != (v in self.left))

    def _d_values(self) -> dict[str, float]:
        d: dict[str, float] = {}
        for node in self.graph.nodes():
            side = self.left if node in self.left else self.right
            external = sum(w for m, w in self.graph.neighbors(node).items() if m not in side)
            internal = sum(w for m, w in self.graph.neighbors(node).items() if m in side)
            d[node] = external - internal
        return d

    def _one_pass(self) -> float:
        d = self._d_values()
        left, right = set(self.left), set(self.right)
        swaps: list[tuple[str, str, float]] = []
        while left and right:
            best: tuple[str, str, float] | None = None
            for a in sorted(left):
                for b in sorted(right):
                    gain = d[a] + d[b] - 2 * self._weight(a, b)
                    if best is None or gain > best[2]:
                        best = (a, b, gain)
            assert best is not None
            a, b, gain = best
            swaps.append((a, b, gain))
            left.discard(a)
            right.discard(b)
            # every unlocked node's D shifts by its ties to the two moved nodes
            for x in left:
                d[x] += 2 * self._weight(x, a) - 2 * self._weight(x, b)
            for y in right:
                d[y] += 2 * self._weight(y, b) - 2 * self._weight(y, a)
        running = 0.0
        best_total = 0.0
        best_k = 0
        for k, (_a, _b, gain) in enumerate(swaps, start=1):
            running += gain
            if running > best_total + 1e-12:
                best_total, best_k = running, k
        for a, b, _gain in swaps[:best_k]:
            self.left.discard(a)
            self.right.discard(b)
            self.left.add(b)
            self.right.add(a)
        return best_total

    def _run(self) -> None:
        while True:
            self.passes += 1
            if self._one_pass() <= 0:
                break
            if self.passes > 50:
                break

    def balanced(self) -> bool:
        return len(self.left) == len(self.right)

    def exact_minimum(self) -> float:
        nodes = self.graph.nodes()
        if len(nodes) > 14:
            raise Invalid("the exact bisection tries every split; keep it to fourteen nodes")
        first = nodes[0]
        best = None
        for chosen in combinations(nodes[1:], len(nodes) // 2 - 1):
            side = {first, *chosen}
            cut = sum(w for u, v, w in self.graph.edges() if (u in side) != (v in side))
            if best is None or cut < best:
                best = cut
        return best if best is not None else 0.0

    def note(self) -> str:
        return (
            f"cut {self.initial_cut:g} to {self.cut_weight():g} in {self.passes} pass(es); "
            f"sides {sorted(self.left)} and {sorted(self.right)}"
        )
