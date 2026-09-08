"""Union-find: track a growing set of merges and answer same-group in near O(1).

Union-find, or disjoint-set, maintains a partition of elements into groups
under two operations: union, which merges the two groups containing two
elements, and find, which returns a canonical representative of an element's
group so that two elements are in the same group exactly when their
representatives match. It is the structure behind connected components as
edges arrive, behind Kruskal's spanning tree as it decides whether an edge
would close a cycle, and behind any incremental grouping. Each group is a
tree whose root is its representative, and find walks from an element to its
root. Two optimizations make the operations almost constant time, and both
matter. Union by rank attaches the shorter tree under the taller one when
merging, so the trees stay shallow instead of degenerating into a linked
list that makes find linear. Path compression, during a find, repoints
every node on the path directly at the root, so the next find on any of
them is immediate. Together they give an amortized cost per operation that
grows so slowly with the number of elements, the inverse Ackermann function,
that it is effectively a small constant for any input that will ever exist.
The structure adds elements, unions two of them and reports whether the
union actually merged two distinct groups or found them already together,
tells whether two elements are connected, and counts the groups. It refuses
an operation on an element it was never told about, because a silent
auto-add would put a typo in its own group and quietly inflate the count.
It reports the group count and the largest group size, the readings that a
components pass and a connectivity check are built directly on.
"""

from __future__ import annotations

from collections import Counter

from mesh.errors import Missing


class UnionFind:
    def __init__(self) -> None:
        self._parent: dict[str, str] = {}
        self._rank: dict[str, int] = {}
        self._groups = 0

    def add(self, element: str) -> None:
        if element not in self._parent:
            self._parent[element] = element
            self._rank[element] = 0
            self._groups += 1

    def find(self, element: str) -> str:
        if element not in self._parent:
            raise Missing(f"'{element}' was never added to the structure")
        # path compression: repoint the whole path at the root
        root = element
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[element] != root:
            self._parent[element], element = root, self._parent[element]
        return root

    def union(self, a: str, b: str) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False  # already together, this union merged nothing
        # union by rank: hang the shorter tree under the taller
        if self._rank[ra] < self._rank[rb]:
            ra, rb = rb, ra
        self._parent[rb] = ra
        if self._rank[ra] == self._rank[rb]:
            self._rank[ra] += 1
        self._groups -= 1
        return True

    def connected(self, a: str, b: str) -> bool:
        return self.find(a) == self.find(b)

    def group_count(self) -> int:
        return self._groups

    def largest_group_size(self) -> int:
        if not self._parent:
            return 0
        sizes = Counter(self.find(e) for e in self._parent)
        return max(sizes.values())

    def note(self) -> str:
        return (
            f"{self._groups} group(s), largest holds "
            f"{self.largest_group_size()}; a single group means everything is "
            "connected, and the largest is the giant component if one dominates"
        )
