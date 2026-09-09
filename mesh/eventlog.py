"""Event log: every edit to a graph recorded, replayable, and undoable back to any point.

A graph that is edited over a session is easier to trust when every
edit is written down. The event log records node and edge additions
and removals as events with a sequence number, applies each to the
live graph as it is recorded, and can rebuild the graph at any
earlier sequence number by replaying the events up to it from an
empty graph, which is what undo means here: not mutation backward
but reconstruction forward to an earlier point. Because the graph
module has no removal, removal is done by rebuilding without the
removed item, and the log keeps that honest by checking that a
removal names something present and an addition names something
absent, refusing both with the name. The log serialises to lines of
text and reads them back, so a session can be saved and resumed, and
a corrupt line is refused with its number. The tests hold the
invariants of an append-only log: replaying every event gives the
live graph, replaying up to an earlier point gives the graph as it
was then, a round trip through text gives an equal log, and the
sequence numbers are contiguous from one. The fingerprint from the
cache module makes the equality checks one-line comparisons.
"""

from __future__ import annotations

from mesh.errors import Invalid
from mesh.graph import Graph
from mesh.graphcache import fingerprint

Event = tuple[int, str, tuple[str, ...]]


class EventLog:
    def __init__(self, directed: bool = False) -> None:
        self.directed = directed
        self.events: list[Event] = []
        self.graph = Graph(directed=directed)

    def _record(self, kind: str, args: tuple[str, ...]) -> None:
        self.events.append((len(self.events) + 1, kind, args))

    def add_node(self, name: str) -> None:
        if name in self.graph.nodes():
            raise Invalid(f"node '{name}' is already present")
        self.graph.add_node(name)
        self._record("add_node", (name,))

    def add_edge(self, u: str, v: str, weight: float = 1.0) -> None:
        for n in (u, v):
            if n not in self.graph.nodes():
                raise Invalid(f"edge {u}-{v} names '{n}', which is absent")
        if self.graph.has_edge(u, v):
            raise Invalid(f"edge {u}-{v} is already present")
        self.graph.add_edge(u, v, weight)
        self._record("add_edge", (u, v, f"{weight:g}"))

    def remove_edge(self, u: str, v: str) -> None:
        if not self.graph.has_edge(u, v):
            raise Invalid(f"no edge {u}-{v} to remove")
        self._record("remove_edge", (u, v))
        self.graph = self.replay()

    def remove_node(self, name: str) -> None:
        if name not in self.graph.nodes():
            raise Invalid(f"no node '{name}' to remove")
        self._record("remove_node", (name,))
        self.graph = self.replay()

    def replay(self, upto: int | None = None) -> Graph:
        limit = len(self.events) if upto is None else upto
        if limit < 0 or limit > len(self.events):
            raise Invalid(f"sequence {limit} is outside 0..{len(self.events)}")
        nodes: list[str] = []
        edges: dict[tuple[str, str], float] = {}
        for seq, kind, args in self.events:
            if seq > limit:
                break
            if kind == "add_node":
                nodes.append(args[0])
            elif kind == "add_edge":
                edges[(args[0], args[1])] = float(args[2])
            elif kind == "remove_edge":
                key = next(
                    k for k in edges if k == (args[0], args[1]) or (
                        not self.directed and k == (args[1], args[0])
                    )
                )
                del edges[key]
            elif kind == "remove_node":
                nodes.remove(args[0])
                edges = {k: w for k, w in edges.items() if args[0] not in k}
        g = Graph(directed=self.directed)
        for n in nodes:
            g.add_node(n)
        for (u, v), w in edges.items():
            g.add_edge(u, v, w)
        return g

    def contiguous(self) -> bool:
        return [seq for seq, _k, _a in self.events] == list(range(1, len(self.events) + 1))

    def dump(self) -> str:
        head = "directed" if self.directed else "undirected"
        body = [f"{seq} {kind} {' '.join(args)}" for seq, kind, args in self.events]
        return "\n".join([head, *body]) + "\n"

    @classmethod
    def load(cls, text: str) -> EventLog:
        lines = [line for line in text.splitlines() if line.strip()]
        if not lines or lines[0] not in ("directed", "undirected"):
            raise Invalid("line 1 must say directed or undirected")
        log = cls(directed=lines[0] == "directed")
        for number, line in enumerate(lines[1:], start=2):
            fields = line.split()
            if len(fields) < 3 or not fields[0].isdigit():
                raise Invalid(f"line {number} is not an event")
            kind, args = fields[1], fields[2:]
            try:
                if kind == "add_node":
                    log.add_node(args[0])
                elif kind == "add_edge":
                    log.add_edge(args[0], args[1], float(args[2]))
                elif kind == "remove_edge":
                    log.remove_edge(args[0], args[1])
                elif kind == "remove_node":
                    log.remove_node(args[0])
                else:
                    raise Invalid(f"line {number}: unknown event '{kind}'")
            except (IndexError, ValueError) as exc:
                raise Invalid(f"line {number} is malformed") from exc
        return log

    def note(self) -> str:
        return (
            f"{len(self.events)} event(s); live graph {self.graph.node_count()} node(s) and "
            f"{self.graph.edge_count()} edge(s), fingerprint {fingerprint(self.graph)}"
        )
