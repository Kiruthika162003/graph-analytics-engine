"""Graph query: a small language for asking a graph questions by name from the command line.

Every module in the engine answers one question, and a reader at a
terminal wants to ask them without writing Python. This module
parses a short query, one verb and its arguments, and dispatches to
the module that answers it, returning the answer as text. The verbs
are the ones a first session needs: degree of a node, whether two
nodes are adjacent, the shortest hop path between two nodes, the
distance between them, the neighbors of a node, the count of
nodes or edges, the component of a node, and the summary of the whole
graph. A query is a list of words, so it can come from a shell line
split on spaces or from a script; the first word picks the verb, the
rest are its arguments, and a wrong count of arguments or an unknown
verb or node is reported as text that names the problem rather than
as an exception, because a terminal session should never end in a
traceback over a typo. The module keeps the verbs in a table so the
help verb can list them, and each verb's entry carries the argument
count it expects, which is what the argument check reads. The tests
run every verb on a small graph and check both the answers and the
messages for every kind of mistake, and they confirm that an empty
query and the help verb both produce the verb list.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable

from mesh.graph import Graph
from mesh.graphsummary import GraphSummary


class GraphQuery:
    def __init__(self, graph: Graph) -> None:
        self.graph = graph
        self.verbs: dict[str, tuple[int, Callable[..., str]]] = {
            "degree": (1, self._degree),
            "adjacent": (2, self._adjacent),
            "path": (2, self._path),
            "distance": (2, self._distance),
            "neighbors": (1, self._neighbors),
            "nodes": (0, lambda: str(self.graph.node_count())),
            "edges": (0, lambda: str(self.graph.edge_count())),
            "component": (1, self._component),
            "summary": (0, lambda: GraphSummary(self.graph).note()),
            "help": (0, self._help),
        }

    def ask(self, words: list[str]) -> str:
        if not words:
            return self._help()
        verb, args = words[0], words[1:]
        if verb not in self.verbs:
            return f"unknown verb '{verb}'; try help"
        count, fn = self.verbs[verb]
        if len(args) != count:
            return f"'{verb}' takes {count} argument(s), got {len(args)}"
        for name in args:
            if name not in self.graph.nodes():
                return f"no node called '{name}'"
        return fn(*args)

    def _help(self) -> str:
        return "verbs: " + ", ".join(f"{v}/{c}" for v, (c, _fn) in self.verbs.items())

    def _degree(self, node: str) -> str:
        return str(self.graph.degree(node))

    def _adjacent(self, a: str, b: str) -> str:
        return "yes" if self.graph.has_edge(a, b) else "no"

    def _bfs(self, start: str) -> dict[str, str | None]:
        parent: dict[str, str | None] = {start: None}
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for other in self.graph.neighbors(node):
                if other not in parent:
                    parent[other] = node
                    queue.append(other)
        return parent

    def _path(self, a: str, b: str) -> str:
        parent = self._bfs(a)
        if b not in parent:
            return f"no path from {a} to {b}"
        trail = [b]
        while trail[-1] != a:
            step = parent[trail[-1]]
            assert step is not None
            trail.append(step)
        return " > ".join(reversed(trail))

    def _distance(self, a: str, b: str) -> str:
        parent = self._bfs(a)
        if b not in parent:
            return "unreachable"
        hops = 0
        node = b
        while node != a:
            node = parent[node]  # type: ignore[assignment]
            hops += 1
        return str(hops)

    def _neighbors(self, node: str) -> str:
        nbrs = sorted(self.graph.neighbors(node))
        return ", ".join(nbrs) if nbrs else "none"

    def _component(self, node: str) -> str:
        return ", ".join(sorted(self._bfs(node)))
