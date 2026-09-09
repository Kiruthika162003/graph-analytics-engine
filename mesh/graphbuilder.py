"""Graph builder: a fluent way to assemble a graph from names, pairs, and small phrases.

Most graphs in tests and examples are typed by hand, and the typing
is where mistakes creep in: a node declared twice, an edge naming a
node that was never added, a weight that is a string. The builder
takes that work over. It accepts nodes one at a time or in a batch,
edges as pairs or triples with a weight, whole shapes by name from
the factories module joined onto what is already there with a
prefix, and a compact text form where each line is either a lone name
or two names with an optional weight, which is the edge list format
the I/O module reads. Every call returns the builder so calls chain,
and build returns the graph and refuses to build twice, so a builder
is used once and its graph is not silently shared. The builder keeps
a log of what it added so a report can say how a graph was made, and
it refuses the mistakes named above with a message that says which
line or which name was wrong. Direction is fixed at construction and
checked whenever a shape is merged in, since a directed shape cannot
be added to an undirected graph. The tests build the same small graph
four ways and confirm the results are equal through the I/O module's
comparison, which is the point: however a graph is typed, the same
edges come out.
"""

from __future__ import annotations

from collections.abc import Iterable

from mesh import factories
from mesh.errors import Invalid
from mesh.graph import Graph


class GraphBuilder:
    def __init__(self, directed: bool = False) -> None:
        self.graph = Graph(directed=directed)
        self.log: list[str] = []
        self.built = False

    def _check_open(self) -> None:
        if self.built:
            raise Invalid("this builder already built its graph; make a new one")

    def node(self, name: str) -> GraphBuilder:
        self._check_open()
        if not name or name != name.strip():
            raise Invalid(f"node name {name!r} is empty or has surrounding spaces")
        if name in self.graph.nodes():
            raise Invalid(f"node '{name}' was already added")
        self.graph.add_node(name)
        self.log.append(f"node {name}")
        return self

    def nodes(self, names: Iterable[str]) -> GraphBuilder:
        for name in names:
            self.node(name)
        return self

    def edge(self, a: str, b: str, weight: float = 1.0) -> GraphBuilder:
        self._check_open()
        for name in (a, b):
            if name not in self.graph.nodes():
                raise Invalid(f"edge {a}-{b} names '{name}', which was never added")
        if not isinstance(weight, int | float) or isinstance(weight, bool):
            raise Invalid(f"weight {weight!r} on {a}-{b} is not a number")
        self.graph.add_edge(a, b, float(weight))
        self.log.append(f"edge {a}-{b} {float(weight):g}")
        return self

    def edges(self, pairs: Iterable[tuple[str, str] | tuple[str, str, float]]) -> GraphBuilder:
        for pair in pairs:
            self.edge(*pair)
        return self

    def shape(self, name: str, *args: int, prefix: str = "") -> GraphBuilder:
        self._check_open()
        maker = getattr(factories, name, None)
        if maker is None or not callable(maker) or name.startswith("_"):
            raise Invalid(f"no shape called '{name}' in the factories")
        piece = maker(*args, prefix=prefix)
        if piece.directed != self.graph.directed:
            raise Invalid(f"shape '{name}' has the wrong direction for this builder")
        for n in piece.nodes():
            self.node(n)
        for u, v, w in piece.edges():
            self.edge(u, v, w)
        self.log.append(f"shape {name}{args} as '{prefix}'")
        return self

    def text(self, body: str) -> GraphBuilder:
        for number, raw in enumerate(body.splitlines(), start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) == 1:
                self.node(fields[0])
            elif len(fields) in (2, 3):
                for name in fields[:2]:
                    if name not in self.graph.nodes():
                        self.node(name)
                try:
                    weight = float(fields[2]) if len(fields) == 3 else 1.0
                except ValueError as exc:
                    bad = fields[2]
                    raise Invalid(f"line {number}: weight '{bad}' is not a number") from exc
                self.edge(fields[0], fields[1], weight)
            else:
                raise Invalid(f"line {number}: expected a name or a pair, got '{line}'")
        return self

    def build(self) -> Graph:
        self._check_open()
        self.built = True
        return self.graph

    def note(self) -> str:
        kind = "directed" if self.graph.directed else "undirected"
        return (
            f"{kind} graph of {self.graph.node_count()} node(s) and {self.graph.edge_count()} "
            f"edge(s) from {len(self.log)} step(s)"
        )
