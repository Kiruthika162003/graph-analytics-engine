"""Segment tree: range maximum with point updates, both in a logarithm.

Several tree algorithms in this engine flatten a tree into an array so
that a subtree or a chain becomes a contiguous range, and then need to
answer what is the largest value in this range while individual values
keep changing. A flat array answers the query by scanning, linear in the
range, and a single running maximum cannot survive a decrease, because
lowering the current maximum leaves it not knowing the runner-up. The
segment tree gives both operations a logarithm. It is a complete binary
tree stored in an array of twice the size, where the leaves hold the
values and every internal node holds the maximum of its two children. A
point update writes one leaf and recomputes its ancestors up to the
root, a logarithm of nodes, and because each ancestor is recomputed from
both children a decrease is handled correctly. A range query walks the
two boundaries inward from the leaves, taking a node's stored maximum
whole whenever the node's range lies inside the query, so it touches a
logarithm of nodes rather than every element. The layout is the
bottom-up one: leaf i lives at index i plus size, its parent at half
that, so no recursion is needed and the tree works for any size, not
only powers of two. An unset leaf reads as zero, which is the honest
default for the values this engine stores in it, node weights and lags
that have not been reported yet; the query accumulator starts at
negative infinity so a genuine value always wins over it. The tree
refuses an index out of range and an inverted query range, since a low
above its high has no meaning and would silently return the identity.
It reports the root, the maximum over everything, the one number a
running scalar could have tracked and the reason the rest of the tree
exists.
"""

from __future__ import annotations

from mesh.errors import Invalid

_NEG_INF = float("-inf")


class SegmentTree:
    def __init__(self, size: int) -> None:
        if size < 1:
            raise Invalid("a segment tree needs at least one element")
        self.size = size
        # bottom-up layout: leaf i at i + size, parent of p at p // 2
        self._tree: list[float] = [0.0] * (2 * size)

    def update(self, index: int, value: float) -> None:
        if not 0 <= index < self.size:
            raise Invalid(f"index {index} is out of range 0..{self.size - 1}")
        pos = index + self.size
        self._tree[pos] = value
        pos //= 2
        while pos >= 1:
            self._tree[pos] = max(self._tree[2 * pos], self._tree[2 * pos + 1])
            pos //= 2

    def range_max(self, low: int, high: int) -> float:
        if low > high:
            raise Invalid(f"range {low}..{high} is inverted; it would return the identity")
        if not (0 <= low < self.size and 0 <= high < self.size):
            raise Invalid(f"range {low}..{high} falls outside 0..{self.size - 1}")
        result = _NEG_INF
        lo = low + self.size
        hi = high + self.size + 1
        while lo < hi:
            if lo & 1:
                result = max(result, self._tree[lo])
                lo += 1
            if hi & 1:
                hi -= 1
                result = max(result, self._tree[hi])
            lo //= 2
            hi //= 2
        return result

    def overall_max(self) -> float:
        return self._tree[1]  # the root, even for a single leaf at index 1

    def note(self) -> str:
        return (
            f"root max {self.overall_max()} over {self.size} element(s); the root is "
            "what a running scalar could track, the rest handles decreases and ranges"
        )
