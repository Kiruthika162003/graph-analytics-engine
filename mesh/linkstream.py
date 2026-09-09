"""Link streams: a graph whose edges exist only during intervals, read over time.

A phone log, a contact tracing record, or a chat history is not a
graph but a stream of links, each alive from a start time to an end
time, and questions about it depend on when they are asked. This
module holds a link stream as a list of timed links, and answers
three kinds of question. The snapshot at a time is the graph of links
alive then, which is what any static reading needs. The aggregate over
a window is the graph of every pair that was linked at any moment in
the window, weighted by how long, which is what a summary over a week
wants. And the time-respecting reachability from a node at a time is
the set of nodes a message could reach if it can only cross a link
while the link is alive and never travels back in time, which is what
contact tracing asks and what a static aggregate gets wrong, because
the aggregate shows paths that never existed in that order. The
engine computes reachability by a breadth-first search over
(node, time) states that always waits for the earliest usable link,
so it also reports the earliest arrival time at every reached node.
Two facts anchor the tests: the aggregate's reach always contains the
time-respecting reach, and a link that is alive only before the start
time never helps. Links are refused when their end precedes their
start, and a query at a time with no live links gives an edgeless
snapshot rather than an error.
"""

from __future__ import annotations

from math import inf

from mesh.errors import Invalid
from mesh.graph import Graph

Link = tuple[str, str, float, float]


class LinkStream:
    def __init__(self, links: list[Link]) -> None:
        self.links: list[Link] = []
        self.nodes: list[str] = []
        seen: set[str] = set()
        for u, v, start, end in links:
            if end < start:
                raise Invalid(f"link {u}-{v} ends at {end} before it starts at {start}")
            self.links.append((u, v, start, end))
            for n in (u, v):
                if n not in seen:
                    seen.add(n)
                    self.nodes.append(n)
        self.links.sort(key=lambda t: (t[2], t[3], t[0], t[1]))

    def _base(self) -> Graph:
        g = Graph()
        for n in self.nodes:
            g.add_node(n)
        return g

    def snapshot(self, at: float) -> Graph:
        g = self._base()
        for u, v, start, end in self.links:
            if start <= at <= end and not g.has_edge(u, v):
                g.add_edge(u, v)
        return g

    def aggregate(self, window_start: float = -inf, window_end: float = inf) -> Graph:
        g = self._base()
        weight: dict[frozenset[str], float] = {}
        for u, v, start, end in self.links:
            overlap = min(end, window_end) - max(start, window_start)
            if overlap <= 0 and not (start == end and window_start <= start <= window_end):
                continue
            weight[frozenset((u, v))] = weight.get(frozenset((u, v)), 0.0) + max(overlap, 0.0)
        for pair, w in weight.items():
            u, v = sorted(pair)
            g.add_edge(u, v, w)
        return g

    def reach(self, source: str, at: float) -> dict[str, float]:
        # earliest arrival at every node reachable by a time-respecting path from (source, at)
        if source not in self.nodes:
            raise Invalid(f"'{source}' is not in the stream")
        arrival = {source: at}
        changed = True
        while changed:
            changed = False
            for u, v, start, end in self.links:
                for a, b in ((u, v), (v, u)):
                    if a not in arrival:
                        continue
                    ready = max(arrival[a], start)
                    if ready <= end and ready < arrival.get(b, inf):
                        arrival[b] = ready
                        changed = True
        return arrival

    def aggregate_reach(self, source: str) -> set[str]:
        g = self.aggregate()
        seen = {source}
        stack = [source]
        while stack:
            node = stack.pop()
            for other in g.neighbors(node):
                if other not in seen:
                    seen.add(other)
                    stack.append(other)
        return seen

    def note(self, source: str, at: float) -> str:
        reached = self.reach(source, at)
        static = self.aggregate_reach(source)
        return (
            f"from {source} at {at:g}: {len(reached)} node(s) reachable in time, "
            f"{len(static)} in the static aggregate; {len(self.links)} link(s) overall"
        )
