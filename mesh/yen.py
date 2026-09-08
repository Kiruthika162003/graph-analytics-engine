"""Yen's k shortest paths: the best route, then the next best, without any loops.

One shortest path is often not enough. A router wants a backup if the
primary fails, a trip planner wants alternatives, a network analyst wants
to know how many nearly-as-good routes exist. Yen's algorithm lists the k
shortest loopless paths from a source to a target in order of length. It
starts with the single shortest path from Dijkstra. Then, for each path
already accepted, it treats every node along that path except the last as
a spur node: the prefix up to the spur is the root path, and the algorithm
temporarily removes every edge that any accepted path with the same root
uses to leave the spur node, plus every root-path node except the spur
itself, so a new path must branch off differently and cannot double back.
It runs Dijkstra from the spur to the target in that reduced graph, joins
the root path to the spur path, and adds the candidate to a pool. The
cheapest candidate not already accepted becomes the next shortest path,
and the process repeats until k paths are found or the pool runs dry. The
edge removals are what keep every path loopless and distinct, and the
root-path node removals are what stop a spur path from revisiting the
prefix. The cost is k times the path length times a Dijkstra run, so it is
meant for small k. The finder returns the paths and their costs in
nondecreasing order, refuses a negative edge for the same reason Dijkstra
does, and reports how many distinct paths it found against the k asked
for, because a graph that runs out before k is one with fewer routes than
the caller hoped, and the gap between the first and the k-th cost is how
much worse the backups are.
"""

from __future__ import annotations

from itertools import pairwise

from mesh.dijkstra import Dijkstra
from mesh.errors import Invalid, Missing, Unreachable
from mesh.graph import Graph


class YenKShortest:
    def __init__(self, graph: Graph, source: str, target: str, k: int) -> None:
        for endpoint in (source, target):
            if not graph.has_node(endpoint):
                raise Missing(f"'{endpoint}' is not in the graph")
        if k < 1:
            raise Invalid("k must be at least one")
        self.graph = graph
        self.source = source
        self.target = target
        self.k = k
        self.paths: list[tuple[float, list[str]]] = []
        self._run()

    def _restricted(
        self, removed_edges: set[tuple[str, str]], removed_nodes: set[str]
    ) -> Graph:
        g = Graph(directed=self.graph.directed)
        for n in self.graph.nodes():
            if n not in removed_nodes:
                g.add_node(n)
        for u, v, w in self.graph.edges():
            if u in removed_nodes or v in removed_nodes:
                continue
            if (u, v) in removed_edges or (not self.graph.directed and (v, u) in removed_edges):
                continue
            g.add_edge(u, v, w)
        return g

    def _cost(self, path: list[str]) -> float:
        return sum(self.graph.weight(u, v) for u, v in pairwise(path))

    def _run(self) -> None:
        first = Dijkstra(self.graph, self.source)
        if self.target not in first.distance:
            return
        # cost every path the same way so int weights stay int in the report
        best = first.path_to(self.target)
        self.paths.append((self._cost(best), best))
        pool: list[tuple[float, list[str]]] = []
        while len(self.paths) < self.k:
            _cost, last = self.paths[-1]
            for i in range(len(last) - 1):
                spur = last[i]
                root = last[: i + 1]
                # block every way an accepted path with this root leaves the spur
                removed_edges = {
                    (p[i], p[i + 1]) for _c, p in self.paths if p[: i + 1] == root
                }
                removed_nodes = set(root[:-1])
                reduced = self._restricted(removed_edges, removed_nodes)
                if not reduced.has_node(spur):
                    continue
                run = Dijkstra(reduced, spur)
                if self.target not in run.distance:
                    continue
                candidate = root[:-1] + run.path_to(self.target)
                entry = (self._cost(candidate), candidate)
                if entry not in pool and all(candidate != p for _c, p in self.paths):
                    pool.append(entry)
            if not pool:
                break
            pool.sort(key=lambda e: (e[0], e[1]))
            self.paths.append(pool.pop(0))

    def costs(self) -> list[float]:
        return [c for c, _p in self.paths]

    def found(self) -> int:
        return len(self.paths)

    def note(self) -> str:
        if not self.paths:
            raise Unreachable(f"'{self.target}' is unreachable from '{self.source}'")
        spread = self.paths[-1][0] - self.paths[0][0]
        return (
            f"found {self.found()} of {self.k} path(s), costs {self.costs()}; the "
            f"backups run up to {spread} worse than the best"
        )
