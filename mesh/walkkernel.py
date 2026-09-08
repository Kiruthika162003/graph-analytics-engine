"""Random walk kernel: two graphs are alike when they share many walks.

A graph kernel scores the similarity of two graphs without aligning
their nodes. The random walk kernel counts walks the two graphs have in
common: a walk in the direct product graph, whose nodes are pairs of
nodes one from each graph and whose edges join two pairs when both
coordinates are adjacent, is exactly a pair of walks of the same length
taken in step. Summing over all lengths with a decay lambda gives the
geometric kernel, the sum over k of lambda to the k times the number of
product walks of length k, which converges when lambda is below one over
the product graph's largest degree. The engine builds the product graph
explicitly, counts walks by repeated adjacency multiplication with a
fixed number of terms, and normalises by the geometric mean of each
graph's self-similarity so the score lands in zero to one and a graph
against itself scores exactly one. Isomorphic graphs score one against
each other, since the product is the same up to relabeling, which is a
check the module runs against the WL hash. A truncation depth of ten is
enough for the small graphs this engine handles and the module reports
the last term's size so a reader can see how far from converged the
count was. Directed graphs are refused and a decay outside the open unit
interval is refused, because the sum has no meaning past it.
"""

from __future__ import annotations

from itertools import product

from mesh.errors import Invalid
from mesh.graph import Graph
from mesh.wlhash import WLHash

Pair = tuple[str, str]


class WalkKernel:
    def __init__(self, decay: float = 0.1, depth: int = 10) -> None:
        if not 0.0 < decay < 1.0:
            raise Invalid("the decay must lie strictly between zero and one")
        if depth < 1:
            raise Invalid("the depth must be at least one")
        self.decay = decay
        self.depth = depth
        self.last_term = 0.0

    @staticmethod
    def _product(a: Graph, b: Graph) -> tuple[list[Pair], dict[Pair, list[int]]]:
        if a.directed or b.directed:
            raise Invalid("the walk kernel compares undirected graphs")
        pairs = list(product(a.nodes(), b.nodes()))
        index = {p: i for i, p in enumerate(pairs)}
        adjacency: dict[Pair, list[int]] = {p: [] for p in pairs}
        for x, y in pairs:
            for xn in a.neighbors(x):
                for yn in b.neighbors(y):
                    adjacency[(x, y)].append(index[(xn, yn)])
            adjacency[(x, y)].sort()
        return pairs, adjacency

    def raw(self, a: Graph, b: Graph) -> float:
        pairs, adjacency = self._product(a, b)
        if not pairs:
            return 0.0
        # walks[i] counts walks of the current length ending at product node i
        walks = [1.0] * len(pairs)
        total = float(len(pairs))
        factor = 1.0
        for _ in range(self.depth):
            factor *= self.decay
            nxt = [0.0] * len(pairs)
            for i, p in enumerate(pairs):
                for j in adjacency[p]:
                    nxt[j] += walks[i]
            walks = nxt
            self.last_term = factor * sum(walks)
            total += self.last_term
        return total

    def similarity(self, a: Graph, b: Graph) -> float:
        cross = self.raw(a, b)
        aa = self.raw(a, a)
        bb = self.raw(b, b)
        if aa == 0.0 or bb == 0.0:
            return 0.0
        return cross / (aa * bb) ** 0.5

    @staticmethod
    def same_hash(a: Graph, b: Graph) -> bool:
        return WLHash(a).digest == WLHash(b).digest

    def note(self, a: Graph, b: Graph) -> str:
        return (
            f"walk kernel similarity {self.similarity(a, b):.4f} at decay {self.decay} over "
            f"{self.depth} terms, last term {self.last_term:.3g}"
        )
