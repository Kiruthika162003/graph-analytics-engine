"""Reading cache: compute each graph reading once per graph version and reuse it until an edit.

The recipes and reports call the same expensive readings several
times on one graph, a breadth-first table here, a spectrum there,
and a caller who asks two questions pays twice. A reading cache holds
the answers keyed by a fingerprint of the graph, so the same question
on the same graph is answered from memory, and an edited graph, which
has a new fingerprint, gets fresh answers. The fingerprint is the
direction flag, the sorted node list, and the sorted canonical edge
list with weights, hashed, which is exactly what the I/O module's
comparison reads, so two graphs the comparison calls equal share a
fingerprint even when built in different orders. Readings are
registered by name with the function that computes them, and the
cache records hits and misses so a report can say how much work it
saved. The tests check the invariants that make a cache safe: the
same graph asked twice computes once, a graph edited between asks
computes again, two graphs equal in content but built in different
orders share a fingerprint, and an unregistered reading is refused by
name. The fingerprint is also useful on its own as a content
identity for a graph, which the module exposes, and it is stable
across processes since it uses a fixed hash rather than Python's
salted one.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from typing import Any

from mesh.errors import Invalid
from mesh.graph import Graph

Reading = Callable[[Graph], Any]


def fingerprint(graph: Graph) -> str:
    parts = ["directed" if graph.directed else "undirected"]
    parts.extend(sorted(graph.nodes()))
    edges = []
    for u, v, w in graph.edges():
        a, b = (u, v) if graph.directed else (min(u, v), max(u, v))
        edges.append(f"{a}\t{b}\t{w:g}")
    parts.extend(sorted(edges))
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:16]


class ReadingCache:
    def __init__(self) -> None:
        self.readings: dict[str, Reading] = {}
        self.store: dict[tuple[str, str], Any] = {}
        self.hits = 0
        self.misses = 0

    def register(self, name: str, reading: Reading) -> None:
        if name in self.readings:
            raise Invalid(f"a reading called '{name}' is already registered")
        self.readings[name] = reading

    def get(self, name: str, graph: Graph) -> Any:
        if name not in self.readings:
            raise Invalid(f"no reading called '{name}'; registered: {sorted(self.readings)}")
        key = (name, fingerprint(graph))
        if key in self.store:
            self.hits += 1
            return self.store[key]
        self.misses += 1
        value = self.readings[name](graph)
        self.store[key] = value
        return value

    def forget(self, graph: Graph) -> int:
        print_ = fingerprint(graph)
        doomed = [key for key in self.store if key[1] == print_]
        for key in doomed:
            del self.store[key]
        return len(doomed)

    def note(self) -> str:
        total = self.hits + self.misses
        rate = self.hits / total if total else 0.0
        return (
            f"{len(self.readings)} reading(s) registered, {len(self.store)} answer(s) held, "
            f"{self.hits} hit(s) and {self.misses} miss(es) ({rate:.0%} served from memory)"
        )
