"""Perfect matching counts: how many ways to pair everyone off, exactly.

A perfect matching pairs every node with exactly one neighbor, and
counting them is harder than finding one: on bipartite graphs the
count is the permanent of the biadjacency matrix, which Valiant showed
is as hard as counting gets, and on general graphs it is the hafnian.
Small graphs are still countable, and the honest way is a recursion
that pins the lowest remaining node: it must be paired with one of its
remaining neighbors, and each choice leaves a smaller graph to pair
off. A memo on the remaining node set keeps repeated subproblems from
being recounted, and the recursion is exact. On bipartite graphs the
module also computes the permanent by Ryser's inclusion-exclusion
formula, which sums over subsets of one side and shares no structure
with the recursion, so agreement between the two is the check that
matters. Known counts pin both: a complete graph on 2n nodes has the
double factorial (2n minus 1)!!, an even cycle has two, a path with an
even number of nodes has one, a ladder of n rungs has the Fibonacci
number F(n plus 1), and any graph with an odd number of nodes has
none. Above sixteen nodes the module refuses rather than run for an
hour, and a directed graph is refused since a matching ignores arrows.
"""

from __future__ import annotations

from collections import deque
from functools import cache
from itertools import combinations

from mesh.errors import Invalid
from mesh.graph import Graph


class PerfectMatchingCount:
    def __init__(self, graph: Graph) -> None:
        if graph.directed:
            raise Invalid("a matching ignores direction; pass an undirected graph")
        if graph.node_count() > 16:
            raise Invalid("counting perfect matchings is exponential; keep it to sixteen nodes")
        self.graph = graph
        self.nodes = graph.nodes()
        self._index = {n: i for i, n in enumerate(self.nodes)}
        self._nbr_mask = [0] * len(self.nodes)
        for u, v, _w in graph.edges():
            self._nbr_mask[self._index[u]] |= 1 << self._index[v]
            self._nbr_mask[self._index[v]] |= 1 << self._index[u]
        self.calls = 0
        self.count = self._count()

    def _count(self) -> int:
        n = len(self.nodes)
        if n % 2:
            return 0

        @cache
        def rec(remaining: int) -> int:
            self.calls += 1
            if remaining == 0:
                return 1
            lowest = (remaining & -remaining).bit_length() - 1
            rest = remaining & ~(1 << lowest)
            total = 0
            choices = self._nbr_mask[lowest] & rest
            while choices:
                bit = choices & -choices
                total += rec(rest & ~bit)
                choices &= ~bit
            return total

        return rec((1 << n) - 1)

    def sides(self) -> tuple[list[str], list[str]] | None:
        # a two-coloring by breadth-first search, or None when an odd cycle blocks it
        color: dict[str, int] = {}
        for start in self.nodes:
            if start in color:
                continue
            color[start] = 0
            queue = deque([start])
            while queue:
                node = queue.popleft()
                for other in self.graph.neighbors(node):
                    if other not in color:
                        color[other] = 1 - color[node]
                        queue.append(other)
                    elif color[other] == color[node]:
                        return None
        left = [n for n in self.nodes if color[n] == 0]
        right = [n for n in self.nodes if color[n] == 1]
        return left, right

    def permanent(self) -> int:
        # Ryser: over subsets S of the right side, (-1)^(n-|S|) times the product over
        # rows of the row's count of neighbors inside S
        split = self.sides()
        if split is None:
            raise Invalid("the permanent reading needs a bipartite graph")
        left, right = split
        if len(left) != len(right):
            return 0
        n = len(left)
        total = 0
        for size in range(1, n + 1):
            for subset in combinations(right, size):
                product = 1
                for u in left:
                    row = sum(1 for v in subset if self.graph.has_edge(u, v))
                    product *= row
                    if product == 0:
                        break
                total += (-1) ** (n - size) * product
        return total

    def note(self) -> str:
        return (
            f"{self.count} perfect matching(s) over {len(self.nodes)} node(s) in "
            f"{self.calls} call(s)"
        )
