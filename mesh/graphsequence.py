"""Graph sequences: a series of snapshots read for churn, persistence, and the steady core.

A network observed at several times is a sequence of graphs, and
the questions it answers are about change: how many edges survive
from one snapshot to the next, how many nodes come and go, which
edges are present in every snapshot, and whether the sequence is
settling or churning. This module holds an ordered list of snapshots
and computes those readings. Edge persistence between consecutive
snapshots is the fraction of the earlier snapshot's edges still
present in the later one; node churn is the count of nodes that
appear or vanish; the core is the graph of edges present in every
snapshot and the union is the graph of edges present in any; and the
trend is the direction of the persistence series, rising when the
network is settling and falling when it is churning faster. Each
consecutive pair also yields a change report from the report module,
so the sequence can print its history step by step. The tests hold
the facts a sequence must respect: a constant sequence has
persistence one, no churn, and a core equal to any snapshot; a
sequence that replaces every edge each step has persistence zero and
an empty core; the core is a subgraph of every snapshot and every
snapshot is a subgraph of the union; and a sequence of one snapshot
has no steps to report. Mixed directions are refused at insertion.
"""

from __future__ import annotations

from mesh.errors import Invalid
from mesh.graph import Graph
from mesh.graphdiffreport import ChangeReport


class GraphSequence:
    def __init__(self, snapshots: list[Graph]) -> None:
        if not snapshots:
            raise Invalid("a sequence needs at least one snapshot")
        directed = snapshots[0].directed
        for i, g in enumerate(snapshots):
            if g.directed != directed:
                raise Invalid(f"snapshot {i} has a different direction from the first")
        self.snapshots = list(snapshots)
        self.directed = directed

    def _keys(self, g: Graph) -> set[tuple[str, str]]:
        if self.directed:
            return {(u, v) for u, v, _w in g.edges()}
        return {(min(u, v), max(u, v)) for u, v, _w in g.edges()}

    def persistence(self) -> list[float]:
        out = []
        for earlier, later in zip(self.snapshots, self.snapshots[1:], strict=False):
            before, after = self._keys(earlier), self._keys(later)
            out.append(len(before & after) / len(before) if before else 1.0)
        return out

    def churn(self) -> list[int]:
        out = []
        for earlier, later in zip(self.snapshots, self.snapshots[1:], strict=False):
            a, b = set(earlier.nodes()), set(later.nodes())
            out.append(len(a ^ b))
        return out

    def _build(self, nodes: list[str], keys: set[tuple[str, str]]) -> Graph:
        g = Graph(directed=self.directed)
        for n in nodes:
            g.add_node(n)
        for u, v in sorted(keys):
            g.add_edge(u, v)
        return g

    def core(self) -> Graph:
        keys = set.intersection(*(self._keys(g) for g in self.snapshots))
        everywhere = set.intersection(*(set(g.nodes()) for g in self.snapshots))
        nodes = [n for n in self.snapshots[0].nodes() if n in everywhere]
        return self._build(nodes, keys)

    def union(self) -> Graph:
        keys = set.union(*(self._keys(g) for g in self.snapshots))
        nodes: list[str] = []
        seen: set[str] = set()
        for g in self.snapshots:
            for n in g.nodes():
                if n not in seen:
                    seen.add(n)
                    nodes.append(n)
        return self._build(nodes, keys)

    def trend(self) -> str:
        series = self.persistence()
        if len(series) < 2:
            return "too short to read"
        first, last = series[0], series[-1]
        if last > first + 1e-9:
            return "settling"
        if last < first - 1e-9:
            return "churning"
        return "steady"

    def reports(self) -> list[list[str]]:
        return [
            ChangeReport(earlier, later).lines()
            for earlier, later in zip(self.snapshots, self.snapshots[1:], strict=False)
        ]

    def core_inside_every_snapshot(self) -> bool:
        core = self._keys(self.core())
        return all(core <= self._keys(g) for g in self.snapshots)

    def note(self) -> str:
        series = ", ".join(f"{p:.2f}" for p in self.persistence())
        return (
            f"{len(self.snapshots)} snapshot(s); persistence [{series or 'none'}], churn "
            f"{self.churn()}, core of {self.core().edge_count()} edge(s), union of "
            f"{self.union().edge_count()}; {self.trend()}"
        )
