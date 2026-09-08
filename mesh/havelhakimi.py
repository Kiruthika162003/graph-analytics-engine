"""Havel-Hakimi: can these degrees be a graph, and if so build one that has them.

A degree sequence lists how many neighbors each node should have. Not
every list can be realized: the degrees must sum to an even number, since
each edge contributes two, and no node can demand more neighbors than
there are other nodes, and even sequences that pass both checks can
still be impossible. Havel-Hakimi decides the question and constructs a
graph when the answer is yes. Sort the sequence descending, take the
largest degree d, remove it, and subtract one from each of the next d
largest degrees, connecting the removed node to those d nodes; repeat on
the shortened sequence. If the process ever needs to subtract from fewer
than d remaining degrees, or drives a degree negative, the sequence is
not graphical; if it reaches all zeros, it is, and the edges recorded
along the way form a graph with exactly the requested degrees. The
theorem behind it is that a sequence is graphical exactly when the
reduced sequence is, because whenever a realization exists there is one
in which the highest-degree node is joined to the next-highest ones, an
exchange argument that lets any other realization be rearranged into
that form without changing any degree. The construction is greedy and
deterministic, which means it produces one particular graph among the
many that might share the degrees, a fact the engine states rather than
implying the graph is unique. The realizer returns whether the sequence
is graphical, the graph it built, and the reason for a refusal, and it
verifies that the built graph's degrees match the request before
returning it. It reports the number of realization steps and whether the
sum check alone would have decided, because a sequence that passes the
even-sum test and still fails is the case where the full recursion earns
its keep over the cheap parity check.
"""

from __future__ import annotations

from mesh.errors import Invalid
from mesh.graph import Graph


class HavelHakimi:
    def __init__(self, degrees: list[int]) -> None:
        if any(d < 0 for d in degrees):
            raise Invalid("a degree cannot be negative")
        self.degrees = list(degrees)
        self.reason = ""
        self.steps = 0
        self.graph: Graph | None = None
        self.graphical = self._realize()

    def parity_ok(self) -> bool:
        return sum(self.degrees) % 2 == 0

    def _realize(self) -> bool:
        if not self.parity_ok():
            self.reason = "the degrees sum to an odd number; each edge adds two"
            return False
        n = len(self.degrees)
        if any(d >= n for d in self.degrees) and n > 0:
            self.reason = "a node asks for more neighbors than there are other nodes"
            return False
        # work on (remaining degree, node id) pairs, highest first each round
        remaining = [(d, str(i)) for i, d in enumerate(self.degrees)]
        g = Graph()
        for _d, node in remaining:
            g.add_node(node)
        while remaining:
            remaining.sort(key=lambda p: (-p[0], p[1]))
            d, node = remaining.pop(0)
            if d == 0:
                continue
            self.steps += 1
            if d > len(remaining):
                self.reason = (
                    f"node {node} needs {d} neighbors but only {len(remaining)} remain"
                )
                return False
            joined: list[tuple[int, str]] = []
            for i in range(d):
                nd, other = remaining[i]
                if nd == 0:
                    self.reason = f"node {node} would need a neighbor with no degree left"
                    return False
                g.add_edge(node, other)
                joined.append((nd - 1, other))
            remaining[:d] = joined
        if any(g.degree(str(i)) != d for i, d in enumerate(self.degrees)):
            self.reason = "the built graph did not match the requested degrees"
            return False
        self.graph = g
        return True

    def note(self) -> str:
        if not self.graphical:
            cheap = "the parity check alone decided" if not self.parity_ok() else \
                "the parity check passed; only the full recursion caught it"
            return f"not graphical: {self.reason}; {cheap}"
        return (
            f"graphical, realized in {self.steps} step(s) as one particular graph "
            "among the possible ones; the construction is greedy, not unique"
        )
