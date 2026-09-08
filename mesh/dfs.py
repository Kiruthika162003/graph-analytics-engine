"""Depth-first search: go deep first, and the timestamps expose the structure.

Depth-first search follows one path as far as it can before backing up,
the opposite discipline to breadth-first's rings. On its own that order is
less useful for distances, but the pair of timestamps DFS stamps on each
node, when it was first discovered and when it was finished after all its
descendants were explored, is what makes DFS the workhorse behind
topological sort, strongly connected components, and cycle detection. The
discovery and finish times nest like well-formed parentheses: one node is
a descendant of another exactly when its whole discover-finish interval
sits inside the other's, the parenthesis theorem, and that nesting is what
the later algorithms read. The search also classifies each edge by the
state of the node it points at when the edge is traversed. An edge to an
undiscovered node is a tree edge, part of the DFS forest. An edge to an
ancestor still on the recursion stack, discovered but not yet finished, is
a back edge, and a back edge is exactly a cycle, which is how DFS detects
one. An edge to an already-finished descendant is a forward edge, and an
edge to an already-finished node in another branch is a cross edge. In an
undirected graph only tree and back edges occur, so a back edge there too
means a cycle. The search runs iteratively to avoid a recursion-depth
limit on deep graphs, records discovery and finish times and parents,
classifies the edges, and reports whether a back edge was seen. It refuses
a source not in the graph. It reports the preorder and postorder, the two
vertex orderings the later algorithms consume, postorder reversed being a
topological order when the graph is acyclic.
"""

from __future__ import annotations

from mesh.errors import Missing
from mesh.graph import Graph


class DFS:
    def __init__(self, graph: Graph, source: str) -> None:
        if not graph.has_node(source):
            raise Missing(f"source '{source}' is not in the graph")
        self.graph = graph
        self.source = source
        self.discover: dict[str, int] = {}
        self.finish: dict[str, int] = {}
        self.parent: dict[str, str | None] = {source: None}
        self.preorder: list[str] = []
        self.postorder: list[str] = []
        self.has_back_edge = False
        self._time = 0
        self._run()

    def _run(self) -> None:
        # an explicit stack of (node, neighbor-iterator) simulates recursion
        self.discover[self.source] = self._tick()
        self.preorder.append(self.source)
        stack: list[tuple[str, list[str]]] = [
            (self.source, list(self.graph.neighbors(self.source)))
        ]
        while stack:
            node, pending = stack[-1]
            if pending:
                neighbor = pending.pop(0)
                self._classify(node, neighbor)
                if neighbor not in self.discover:
                    self.parent[neighbor] = node
                    self.discover[neighbor] = self._tick()
                    self.preorder.append(neighbor)
                    stack.append((neighbor, list(self.graph.neighbors(neighbor))))
            else:
                self.finish[node] = self._tick()
                self.postorder.append(node)
                stack.pop()

    def _classify(self, node: str, neighbor: str) -> None:
        if neighbor not in self.discover:
            return  # a tree edge is handled when the neighbor is pushed
        if neighbor not in self.finish:
            # neighbor discovered but not finished: it is an ancestor on the stack
            if not self.graph.directed and self.parent.get(node) == neighbor:
                return  # the edge back to the parent is not a cycle in undirected
            self.has_back_edge = True

    def _tick(self) -> int:
        self._time += 1
        return self._time

    def is_ancestor(self, ancestor: str, descendant: str) -> bool:
        # the parenthesis theorem: nesting of discover-finish intervals
        if ancestor not in self.discover or descendant not in self.discover:
            return False
        return (
            self.discover[ancestor] < self.discover[descendant]
            and self.finish[descendant] < self.finish[ancestor]
        )

    def reachable_nodes(self) -> set[str]:
        return set(self.discover)

    def note(self) -> str:
        return (
            f"reached {len(self.discover)} node(s) from '{self.source}', "
            f"back edge seen: {self.has_back_edge}; reversed postorder is a "
            "topological order when there is no back edge"
        )
