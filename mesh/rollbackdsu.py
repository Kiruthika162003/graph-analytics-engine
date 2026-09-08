"""Rollback union-find: merge, ask, then take the merge back.

Ordinary union-find only ever merges, which is right for edges that
arrive and stay. Some questions need edges that arrive and later leave:
is the network connected during this window, which pairs were joined
only while a temporary link was up, what does connectivity look like if
this edge is removed. Deletion is hard for union-find because path
compression rewrites parent pointers in ways that cannot be undone. The
rollback version gives up compression and keeps union by size only, so
every union changes exactly two things, one parent pointer and one
size, and records both on a stack. Undoing the last union pops the
record and restores the two fields, in constant time, and undoing back
to any earlier checkpoint is a sequence of pops. Without compression a
find walks up to the root without rewriting, and union by size keeps
that walk logarithmic, so the price of reversibility is a logarithm per
operation instead of the near-constant the compressed version enjoys.
That trade is what makes offline dynamic connectivity work: process the
edges in an order that nests their lifetimes, union on entry, answer the
queries inside, roll back on exit, and every query is answered against
exactly the edges alive at its moment. The structure unions, finds,
counts groups, checkpoints, rolls back to a checkpoint, and refuses a
rollback past the beginning of history. It reports the undo stack depth
against the union count, because a stack that never shrinks is a caller
using a rollback structure as a plain one, paying the logarithm for a
reversibility they never use.
"""

from __future__ import annotations

from mesh.errors import Invalid, Missing


class RollbackUnionFind:
    def __init__(self) -> None:
        self._parent: dict[str, str] = {}
        self._size: dict[str, int] = {}
        self._groups = 0
        self._history: list[tuple[str, str, int] | None] = []
        self.unions = 0

    def add(self, element: str) -> None:
        if element not in self._parent:
            self._parent[element] = element
            self._size[element] = 1
            self._groups += 1

    def find(self, element: str) -> str:
        if element not in self._parent:
            raise Missing(f"'{element}' was never added")
        # no path compression: the walk must leave the pointers intact
        while self._parent[element] != element:
            element = self._parent[element]
        return element

    def union(self, a: str, b: str) -> bool:
        ra, rb = self.find(a), self.find(b)
        self.unions += 1
        if ra == rb:
            self._history.append(None)  # a no-op union still takes a slot to undo
            return False
        if self._size[ra] < self._size[rb]:
            ra, rb = rb, ra
        # exactly two fields change, and both are recorded for the undo
        self._history.append((rb, ra, self._size[ra]))
        self._parent[rb] = ra
        self._size[ra] += self._size[rb]
        self._groups -= 1
        return True

    def connected(self, a: str, b: str) -> bool:
        return self.find(a) == self.find(b)

    def group_count(self) -> int:
        return self._groups

    def checkpoint(self) -> int:
        return len(self._history)

    def rollback(self, to: int | None = None) -> None:
        target = len(self._history) - 1 if to is None else to
        if target < 0 or target > len(self._history):
            raise Invalid(f"cannot roll back to {target}; history holds {len(self._history)}")
        while len(self._history) > target:
            record = self._history.pop()
            if record is None:
                continue
            child, root, old_size = record
            self._parent[child] = child
            self._size[root] = old_size
            self._groups += 1

    def depth(self) -> int:
        return len(self._history)

    def note(self) -> str:
        return (
            f"undo stack {self.depth()} deep after {self.unions} union(s), "
            f"{self._groups} group(s); a stack that never shrinks is paying for a "
            "reversibility never used"
        )
