"""Temporal paths: a route is only real if its edges happen in the right order.

A static graph says two nodes are connected if a chain of edges joins
them. A temporal graph attaches a time to every edge, the moment a
flight departs, an email is sent, a contact occurs, and a chain of edges
is only a real route if each edge happens no earlier than the one
before it arrives: you cannot take a connecting flight that left before
you landed. That single constraint breaks the intuitions the static
graph gives. Reachability is not symmetric even on undirected contacts,
because A can reach C through B only if the A-B contact precedes the
B-C contact, and the reverse order lets C reach A instead. Reachability
is not transitive: A reaches B and B reaches C does not mean A reaches
C if B's route to C happened before A arrived at B. The earliest-arrival
question, the soonest one can reach each node starting from a source at
a given time, has a clean streaming answer: process the contacts in time
order, and whenever a contact's start node has already been reached by
its time, mark its end node as reached at that time if that improves
it. One pass over the sorted contacts computes every earliest arrival,
because a contact can only help after its start node is reachable and
the sorted order presents contacts in exactly the order they could be
used. The engine stores contacts with timestamps, computes earliest
arrivals from a source and start time, reconstructs the route as a
timed sequence, and reports how many nodes the static graph says are
reachable against how many the timing actually allows, because that gap
is the whole difference between a network of possible connections and a
network of connections that ever line up.
"""

from __future__ import annotations

from mesh.bfs import BFS
from mesh.errors import Invalid, Missing, Unreachable
from mesh.graph import Graph


class TemporalGraph:
    def __init__(self, directed: bool = False) -> None:
        self.directed = directed
        self.nodes: set[str] = set()
        # (time, from, to), an undirected contact usable in either direction
        self.contacts: list[tuple[int, str, str]] = []

    def add_contact(self, u: str, v: str, time: int) -> None:
        if time < 0:
            raise Invalid("a contact time cannot be negative")
        self.nodes.update((u, v))
        self.contacts.append((time, u, v))

    def static_graph(self) -> Graph:
        g = Graph(directed=self.directed)
        for n in sorted(self.nodes):
            g.add_node(n)
        for _t, u, v in self.contacts:
            if not g.has_edge(u, v):
                g.add_edge(u, v)
        return g

    def earliest_arrival(self, source: str, start: int = 0) -> dict[str, int]:
        if source not in self.nodes:
            raise Missing(f"node '{source}' has no contacts")
        arrival = {source: start}
        self._parent: dict[str, tuple[str, int]] = {}
        # one pass in time order: a contact helps only once its tail is reached
        for time, u, v in sorted(self.contacts):
            if time < start:
                continue
            pairs = [(u, v)] if self.directed else [(u, v), (v, u)]
            for a, b in pairs:
                reached_in_time = a in arrival and arrival[a] <= time
                if reached_in_time and time < arrival.get(b, float("inf")):
                    arrival[b] = time
                    self._parent[b] = (a, time)
        self._arrival = arrival
        return arrival

    def route(self, source: str, target: str, start: int = 0) -> list[tuple[str, int]]:
        arrival = self.earliest_arrival(source, start)
        if target not in arrival:
            raise Unreachable(f"'{target}' is never reached from '{source}' after {start}")
        # each step carries the node's own arrival time; a first version
        # stamped it with the departure time of the contact leaving it
        steps: list[tuple[str, int]] = [(target, arrival[target])]
        node = target
        while node != source:
            node, _time = self._parent[node]
            steps.append((node, arrival[node]))
        steps.reverse()
        return steps

    def temporal_reach(self, source: str, start: int = 0) -> int:
        return len(self.earliest_arrival(source, start)) - 1

    def static_reach(self, source: str) -> int:
        return len(BFS(self.static_graph(), source).reachable_nodes()) - 1

    def note(self, source: str, start: int = 0) -> str:
        static = self.static_reach(source)
        timed = self.temporal_reach(source, start)
        return (
            f"from '{source}' at {start}: {timed} node(s) reachable in time against "
            f"{static} the static graph promises; the gap is connections that never line up"
        )
