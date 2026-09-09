"""Benchmark: readings timed over graphs of growing size, with work counts that do not jitter.

Wall-clock times tell a reader how long a reading takes on a machine
and they jitter from run to run; work counts tell how the reading
grows and they do not. This module records both. A benchmark holds
named readings, each a callable on a graph, and an optional counting
version of each that returns the amount of work done, such as the
number of edge visits, so growth can be read from counts while times
are reported beside them. Graphs come from a generator given the
size, so the same benchmark runs on paths, cycles, random graphs, or
anything the caller builds. A run sweeps the sizes in order, times
each reading with the standard library's performance counter, and
records size, seconds, and work; the growth of a reading between two
sizes is the ratio of its work counts against the ratio of the
sizes, which is close to one for linear work and close to the size
ratio for quadratic work, and the module names the growth in words
by comparing the two. The tests check the frame rather than the
clock: sizes come back ascending, seconds are never negative, a
counting reading that visits every edge grows like the edge count, a
counting reading that compares every pair grows like the square, and
a reading that raises is reported as failed in its row rather than
ending the sweep. The table is rendered as fixed-width text so it
can be printed as it is.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from math import log

from mesh.errors import Invalid, MeshError
from mesh.graph import Graph

Reading = Callable[[Graph], object]
Counter = Callable[[Graph], int]
Row = tuple[str, int, float, int | None, str | None]


class Benchmark:
    def __init__(self, make: Callable[[int], Graph], sizes: list[int]) -> None:
        if not sizes or sizes != sorted(sizes) or sizes[0] < 1:
            raise Invalid("sizes must be a non-empty ascending list of positive counts")
        self.make = make
        self.sizes = list(sizes)
        self.readings: dict[str, tuple[Reading, Counter | None]] = {}
        self.rows: list[Row] = []

    def add(self, name: str, reading: Reading, counter: Counter | None = None) -> Benchmark:
        if name in self.readings:
            raise Invalid(f"a reading called '{name}' is already registered")
        self.readings[name] = (reading, counter)
        return self

    def run(self) -> list[Row]:
        self.rows = []
        for size in self.sizes:
            graph = self.make(size)
            for name, (reading, counter) in self.readings.items():
                start = time.perf_counter()
                failure: str | None = None
                try:
                    reading(graph)
                except MeshError as exc:
                    failure = str(exc)
                seconds = time.perf_counter() - start
                work = counter(graph) if counter is not None and failure is None else None
                self.rows.append((name, size, seconds, work, failure))
        return self.rows

    def growth(self, name: str) -> float | None:
        # exponent of work against size between the smallest and largest sizes
        mine = [row for row in self.rows if row[0] == name and row[4] is None]
        points = [(row[1], row[3]) for row in mine if row[3] is not None and row[3] > 0]
        if len(points) < 2 or points[0][0] == points[-1][0]:
            return None
        (s0, w0), (s1, w1) = points[0], points[-1]
        return log(w1 / w0) / log(s1 / s0)

    @staticmethod
    def describe(exponent: float | None) -> str:
        if exponent is None:
            return "unknown"
        if exponent < 0.5:
            return "constant"
        if exponent < 1.5:
            return "linear"
        if exponent < 2.5:
            return "quadratic"
        return "steeper than quadratic"

    def table(self) -> str:
        lines = [f"{'reading':<14}{'size':>8}{'seconds':>12}{'work':>10}  status"]
        for name, size, seconds, work, failure in self.rows:
            shown = "-" if work is None else str(work)
            status = "ok" if failure is None else f"failed: {failure}"
            lines.append(f"{name:<14}{size:>8}{seconds:>12.6f}{shown:>10}  {status}")
        return "\n".join(lines)

    def note(self) -> str:
        parts = [f"{name} {self.describe(self.growth(name))}" for name in self.readings]
        return f"{len(self.rows)} row(s) over sizes {self.sizes}; growth: " + ", ".join(parts)
