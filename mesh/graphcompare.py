"""Graph comparison report: every similarity reading the engine has, on one pair, in one table.

The engine now measures how alike two graphs are in half a dozen
ways that do not agree with each other, and a reader wants them side
by side rather than one at a time. This module runs them all on a
pair and reports each with its scale: whether the Weisfeiler-Lehman
digests match, which is a fast no or a maybe; whether an exact
isomorphism exists, which settles the maybe; the edit distance, which
is zero exactly when isomorphic and counts operations otherwise; the
maximum common induced subgraph size, which is the node count when
isomorphic and less otherwise; the walk kernel similarity, which is
one when isomorphic and near one for graphs that share many walks;
and the distance between the two degree sequences, sorted and padded,
which is a bound on edit distance since every node insertion or edge
change moves at least one degree. The readings are consistent with
one another in ways the module checks: if the graphs are isomorphic,
the digests match, the edit distance is zero, the common subgraph is
everything, and the kernel reads one; if the digests differ, the
graphs are not isomorphic. Those implications are the module's own
test of the modules under it. Size limits from the exact modules
carry through, so a pair beyond eight nodes on either side skips the
edit distance and says so rather than failing.
"""

from __future__ import annotations

from mesh.commonsubgraph import CommonSubgraph
from mesh.editdistance import EditDistance
from mesh.graph import Graph
from mesh.isomorphism import Isomorphism
from mesh.walkkernel import WalkKernel
from mesh.wlhash import WLHash


class GraphCompare:
    def __init__(self, first: Graph, second: Graph) -> None:
        self.first = first
        self.second = second
        self.same_digest = WLHash(first).digest == WLHash(second).digest
        self.isomorphic = Isomorphism(first, second).isomorphic
        big = max(first.node_count(), second.node_count())
        self.edit: int | None = EditDistance(first, second).distance if big <= 8 else None
        self.common: int | None = (
            CommonSubgraph(first, second).size()
            if first.node_count() * second.node_count() <= 64
            else None
        )
        self.kernel = WalkKernel().similarity(first, second)
        self.degree_gap = self._degree_gap()

    def _degree_gap(self) -> int:
        a = sorted((self.first.degree(n) for n in self.first.nodes()), reverse=True)
        b = sorted((self.second.degree(n) for n in self.second.nodes()), reverse=True)
        size = max(len(a), len(b))
        a += [0] * (size - len(a))
        b += [0] * (size - len(b))
        return sum(abs(x - y) for x, y in zip(a, b, strict=True))

    def consistent(self) -> bool:
        if self.isomorphic:
            if not self.same_digest or abs(self.kernel - 1.0) > 1e-9:
                return False
            if self.edit is not None and self.edit != 0:
                return False
            if self.common is not None and self.common != self.first.node_count():
                return False
        if not self.same_digest and self.isomorphic:
            return False
        return not (self.edit is not None and self.edit == 0 and not self.isomorphic)

    def note(self) -> str:
        edit = "skipped" if self.edit is None else str(self.edit)
        common = "skipped" if self.common is None else str(self.common)
        return (
            f"digests {'match' if self.same_digest else 'differ'}, "
            f"{'isomorphic' if self.isomorphic else 'not isomorphic'}, edit distance {edit}, "
            f"common subgraph {common}, walk similarity {self.kernel:.3f}, "
            f"degree gap {self.degree_gap}"
        )
