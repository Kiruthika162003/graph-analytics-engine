"""Interval graphs: intervals on a line, adjacent when they overlap.

An interval graph has a node per interval and an edge between two
intervals that share a point. Meeting schedules, gene segments, and
register lifetimes all produce them, and the questions asked of them
are the hard ones elsewhere: how many rooms do the meetings need, which
is the chromatic number, and how many meetings can run at once, which
is the clique number. On interval graphs both equal the maximum depth
of overlap at any point, and a single sweep across the sorted endpoints
finds it: walk the endpoints in order, count up at a left end and down
at a right end, and the highest count is the answer. Greedy coloring by
left endpoint uses exactly that many colors, which is what makes the
class perfect. The engine builds the graph from named closed intervals,
sweeps for the depth, colors greedily by left endpoint, and checks that
the coloring is proper and uses the depth's worth of colors. Interval
graphs are chordal, so the module confirms the chordal recognizer agrees
and that the clique number it reads off its elimination ordering equals
the sweep's depth, which ties two independent methods to one number.
Ties at a shared point are ordered so left ends come before right ends,
which is what makes a closed interval touching another at a single point
count as an overlap. An interval with its right end before its left end
is refused by name.
"""

from __future__ import annotations

from mesh.chordal import Chordal
from mesh.errors import Invalid
from mesh.graph import Graph


class IntervalGraph:
    def __init__(self, intervals: dict[str, tuple[float, float]]) -> None:
        for name, (lo, hi) in intervals.items():
            if hi < lo:
                raise Invalid(f"interval '{name}' ends at {hi} before it starts at {lo}")
        self.intervals = dict(intervals)
        self.graph = self._build()

    def _build(self) -> Graph:
        g = Graph()
        names = sorted(self.intervals, key=lambda n: (self.intervals[n], n))
        for n in names:
            g.add_node(n)
        for i, a in enumerate(names):
            lo_a, hi_a = self.intervals[a]
            for b in names[i + 1 :]:
                lo_b, hi_b = self.intervals[b]
                if lo_b <= hi_a and lo_a <= hi_b:
                    g.add_edge(a, b)
        return g

    def depth(self) -> int:
        # left ends sort before right ends at the same point, so touching counts
        events: list[tuple[float, int]] = []
        for lo, hi in self.intervals.values():
            events.append((lo, 0))
            events.append((hi, 1))
        events.sort()
        best = current = 0
        for _, kind in events:
            current += 1 if kind == 0 else -1
            best = max(best, current)
        return best

    def coloring(self) -> dict[str, int]:
        colors: dict[str, int] = {}
        for name in sorted(self.intervals, key=lambda n: (self.intervals[n][0], n)):
            taken = {colors[m] for m in self.graph.neighbors(name) if m in colors}
            color = 0
            while color in taken:
                color += 1
            colors[name] = color
        return colors

    def coloring_is_proper(self) -> bool:
        colors = self.coloring()
        return all(colors[u] != colors[v] for u, v, _w in self.graph.edges())

    def colors_used(self) -> int:
        colors = self.coloring()
        return len(set(colors.values())) if colors else 0

    def chordal_agrees(self) -> bool:
        chordal = Chordal(self.graph)
        if not chordal.is_chordal:
            return False
        return self.graph.node_count() == 0 or chordal.clique_number() == self.depth()

    def note(self) -> str:
        return (
            f"{len(self.intervals)} intervals, {self.graph.edge_count()} overlaps, depth "
            f"{self.depth()} so {self.colors_used()} colors suffice by the left-end greedy"
        )
