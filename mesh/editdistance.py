"""Graph edit distance: the fewest node and edge insertions and deletions between two graphs.

Isomorphism says whether two graphs are the same; edit distance says
how far apart they are. With unit costs, the distance is the smallest
number of node insertions, node deletions, edge insertions, and edge
deletions that turn one graph into the other, and it is zero exactly
when the graphs are isomorphic. It is NP-hard, and the exact method
here is a search over partial node mappings: nodes of the first graph
are assigned in order to unused nodes of the second or to deletion,
each assignment's edge cost is charged as soon as both endpoints are
placed, and a branch is abandoned when its cost so far plus a cheap
lower bound, the difference in remaining node counts, cannot beat the
best complete mapping found. The mapping that achieves the distance
is kept, so a caller can see which nodes were matched. Small closed
forms pin the arithmetic: a path of three to a triangle costs one edge
insertion, a graph to itself costs zero, a graph to the empty graph
costs its nodes plus its edges, and the distance is symmetric, which
the tests check by computing it both ways. Above eight nodes on the
larger graph the search is refused, since the tree of mappings grows
factorially; the module is for measuring small motifs against one
another, not for comparing networks.
"""

from __future__ import annotations

from mesh.errors import Invalid
from mesh.graph import Graph


class EditDistance:
    def __init__(self, first: Graph, second: Graph) -> None:
        if first.directed != second.directed:
            raise Invalid("both graphs must be directed or both undirected")
        if max(first.node_count(), second.node_count()) > 8:
            raise Invalid("the exact search is factorial; keep both graphs to eight nodes")
        self.first = first
        self.second = second
        self.a = first.nodes()
        self.b = second.nodes()
        self.best = float("inf")
        self.mapping: dict[str, str | None] = {}
        self.branches = 0
        self._search(0, {}, set(), 0)
        self.distance = int(self.best)

    def _edge_cost(self, placed: dict[str, str | None], node: str, image: str | None) -> int:
        # edges between the newly placed node and earlier placed nodes; an undirected pair
        # is one edge, a directed pair is two arcs checked separately
        cost = 0
        for other, other_image in placed.items():
            images = {node: image, other: other_image}
            pairs = [(node, other), (other, node)] if self.first.directed else [(node, other)]
            for x, y in pairs:
                here = self.first.has_edge(x, y)
                there = False
                if images[x] is not None and images[y] is not None:
                    there = self.second.has_edge(images[x], images[y])
                cost += here != there
        return cost

    def _search(self, i: int, placed: dict[str, str | None], used: set[str], cost: int) -> None:
        self.branches += 1
        remaining_a = len(self.a) - i
        remaining_b = len(self.b) - len(used)
        if cost + abs(remaining_a - remaining_b) >= self.best:
            return
        if i == len(self.a):
            # unmatched nodes of the second graph are inserted with their internal edges
            leftover = [n for n in self.b if n not in used]
            extra = len(leftover) + sum(
                1
                for u, v, _w in self.second.edges()
                if u in leftover or v in leftover
            )
            total = cost + extra
            if total < self.best:
                self.best = total
                self.mapping = dict(placed)
            return
        node = self.a[i]
        for image in self.b:
            if image in used:
                continue
            step = self._edge_cost(placed, node, image)
            placed[node] = image
            used.add(image)
            self._search(i + 1, placed, used, cost + step)
            used.discard(image)
            del placed[node]
        step = 1 + self._edge_cost(placed, node, None)
        placed[node] = None
        self._search(i + 1, placed, used, cost + step)
        del placed[node]

    def matched_pairs(self) -> list[tuple[str, str]]:
        return [(a, b) for a, b in self.mapping.items() if b is not None]

    def note(self) -> str:
        kept = len(self.matched_pairs())
        return (
            f"edit distance {self.distance} with {kept} node(s) matched, "
            f"searched {self.branches} branch(es)"
        )
