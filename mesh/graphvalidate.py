"""Validation: check a graph against a stated contract before an algorithm trusts it.

Half the refusals in this engine are the same handful of checks:
undirected, connected, no negative weights, no self loops, a node
count within a limit. A contract states those once, and validation
reports every breach at once with the offending node or edge, rather
than the first one an algorithm happens to hit. A contract is built
by chaining requirements: a direction, connectedness, weight bounds,
no self loops, no isolated nodes, a maximum node or edge count,
simple weights that are whole numbers, and a required set of node
names. Checking returns a list of breaches as sentences, and an
empty list means the graph meets the contract; a strict check raises
with all breaches joined, for callers that want a refusal. Each
requirement is tested on a graph that meets it and one that breaks
it, and the tests hold that the breaches are named precisely: the
self loop names its node, the negative weight names its edge and
value, the missing required node names itself, and the size limit
states both the count and the limit. The report also counts how
many requirements were checked, so a caller can see a contract with
nothing in it for what it is.
"""

from __future__ import annotations

from collections import deque

from mesh.errors import Invalid
from mesh.graph import Graph


class Contract:
    def __init__(self) -> None:
        self.rules: list[tuple[str, object]] = []

    def directed(self, value: bool = True) -> Contract:
        self.rules.append(("directed", value))
        return self

    def connected(self) -> Contract:
        self.rules.append(("connected", True))
        return self

    def weights_between(self, low: float, high: float) -> Contract:
        if low > high:
            raise Invalid("the weight bounds are reversed")
        self.rules.append(("weights", (low, high)))
        return self

    def no_self_loops(self) -> Contract:
        self.rules.append(("no_self_loops", True))
        return self

    def no_isolated(self) -> Contract:
        self.rules.append(("no_isolated", True))
        return self

    def at_most(self, nodes: int | None = None, edges: int | None = None) -> Contract:
        self.rules.append(("at_most", (nodes, edges)))
        return self

    def whole_weights(self) -> Contract:
        self.rules.append(("whole_weights", True))
        return self

    def requires(self, names: list[str]) -> Contract:
        self.rules.append(("requires", list(names)))
        return self

    def check(self, graph: Graph) -> list[str]:
        breaches: list[str] = []
        for kind, arg in self.rules:
            breaches.extend(self._one(kind, arg, graph))
        return breaches

    def enforce(self, graph: Graph) -> None:
        breaches = self.check(graph)
        if breaches:
            raise Invalid("; ".join(breaches))

    def _one(self, kind: str, arg: object, graph: Graph) -> list[str]:
        if kind == "directed":
            want = bool(arg)
            if graph.directed != want:
                return [f"the graph is {'undirected' if want else 'directed'}"]
            return []
        if kind == "connected":
            return [] if _connected(graph) else ["the graph is not connected"]
        if kind == "weights":
            low, high = arg  # type: ignore[misc]
            return [
                f"edge {u}-{v} has weight {w:g} outside {low:g}..{high:g}"
                for u, v, w in graph.edges()
                if w < low or w > high
            ]
        if kind == "no_self_loops":
            return [f"node {n} loops to itself" for n in graph.nodes() if graph.has_edge(n, n)]
        if kind == "no_isolated":
            return [f"node {n} has no edge" for n in graph.nodes() if graph.degree(n) == 0]
        if kind == "at_most":
            nodes, edges = arg  # type: ignore[misc]
            out = []
            if nodes is not None and graph.node_count() > nodes:
                out.append(f"{graph.node_count()} node(s) exceeds the limit of {nodes}")
            if edges is not None and graph.edge_count() > edges:
                out.append(f"{graph.edge_count()} edge(s) exceeds the limit of {edges}")
            return out
        if kind == "whole_weights":
            return [
                f"edge {u}-{v} has a fractional weight {w:g}"
                for u, v, w in graph.edges()
                if w != int(w)
            ]
        if kind == "requires":
            present = set(graph.nodes())
            return [f"required node {n} is missing" for n in arg if n not in present]  # type: ignore[union-attr]
        raise Invalid(f"unknown rule '{kind}'")

    def note(self, graph: Graph) -> str:
        breaches = self.check(graph)
        if not breaches:
            return f"meets all {len(self.rules)} requirement(s)"
        return f"{len(breaches)} breach(es) of {len(self.rules)} requirement(s): " + "; ".join(
            breaches
        )


def _connected(graph: Graph) -> bool:
    nodes = graph.nodes()
    if not nodes:
        return True
    seen = {nodes[0]}
    queue = deque([nodes[0]])
    while queue:
        node = queue.popleft()
        for other in graph.neighbors(node):
            if other not in seen:
                seen.add(other)
                queue.append(other)
    return len(seen) == len(nodes)
