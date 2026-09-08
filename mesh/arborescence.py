"""Minimum arborescence: the cheapest directed tree that reaches everything from a root.

A spanning tree of an undirected graph has a directed cousin, the
spanning arborescence: a set of edges in which every node except the
root has exactly one incoming edge and every node is reachable from the
root by following edges forward. It is the cheapest way to broadcast
from one source when links are one-way and priced. Kruskal and Prim do
not carry over, because the greedy choice that is safe for undirected
cuts is not safe under direction, and the algorithm that works is
Chu-Liu/Edmonds. Every non-root node picks its cheapest incoming edge.
If those picks form no cycle they are the arborescence and the total is
their sum. If they form a cycle, no arborescence can contain the whole
cycle, so the cycle is contracted into a single super-node: every edge
entering the cycle from outside has its cost reduced by the cost of the
cycle edge it would displace, the cheapest edge already chosen into that
node, so that choosing it later accounts for dropping the cycle edge.
The algorithm recurses on the contracted graph, and when it returns, the
one edge chosen into the super-node determines which cycle edge to drop:
the rest of the cycle is kept. Each contraction removes at least one
node, so the recursion is bounded by the node count, and with a plain
scan per level the running time is nodes times edges. The reduction of
entering edges is the whole idea, and it is what makes the eventual
total equal the sum of chosen original edges rather than something
merely close. The builder finds the arborescence, returns its edges and
total weight, refuses a root that cannot reach every node since no
arborescence exists then, and reports the total against the sum of every
node's cheapest incoming edge, because that sum is a lower bound and the
gap above it is exactly the price the cycles imposed.
"""

from __future__ import annotations

from mesh.bfs import BFS
from mesh.errors import Invalid, Missing
from mesh.graph import Graph


class Arborescence:
    def __init__(self, graph: Graph, root: str) -> None:
        if not graph.directed:
            raise Invalid("an arborescence is a directed spanning tree; use a digraph")
        if not graph.has_node(root):
            raise Missing(f"root '{root}' is not in the graph")
        reached = BFS(graph, root).reachable_nodes()
        if len(reached) != graph.node_count():
            raise Invalid("the root cannot reach every node, so no arborescence exists")
        self.graph = graph
        self.root = root
        self.lower_bound = self._lower_bound()
        edges = [(u, v, w) for u, v, w in graph.edges()]
        self.total = self._solve(edges, graph.nodes(), root)

    def _lower_bound(self) -> float:
        total = 0.0
        for v in self.graph.nodes():
            if v == self.root:
                continue
            incoming = [w for u, x, w in self.graph.edges() if x == v]
            total += min(incoming)
        return total

    def _solve(
        self, edges: list[tuple[str, str, float]], nodes: list[str], root: str
    ) -> float:
        # each non-root node takes its cheapest incoming edge
        best_in: dict[str, tuple[str, float]] = {}
        for u, v, w in edges:
            if v in (root, u):
                continue  # nothing enters the root, and self loops never help
            if v not in best_in or w < best_in[v][1]:
                best_in[v] = (u, w)
        total = sum(w for _u, w in best_in.values())
        # look for a cycle among the chosen edges
        cycle = self._find_cycle(best_in, root)
        if cycle is None:
            return total
        # contract the cycle into one super-node with reduced entering costs
        super_node = "\0cycle:" + ",".join(sorted(cycle))
        in_cycle = set(cycle)
        contracted: list[tuple[str, str, float]] = []
        for u, v, w in edges:
            if u in in_cycle and v in in_cycle:
                continue  # inside the cycle: gone after contraction
            if v in in_cycle:
                reduced = w - best_in[v][1]  # displaces the cycle edge into v
                contracted.append((u, super_node, reduced))
            elif u in in_cycle:
                contracted.append((super_node, v, w))
            else:
                contracted.append((u, v, w))
        new_nodes = [n for n in nodes if n not in in_cycle] + [super_node]
        cycle_cost = sum(best_in[v][1] for v in cycle)
        return cycle_cost + self._solve(contracted, new_nodes, root)

    @staticmethod
    def _find_cycle(best_in: dict[str, tuple[str, float]], root: str) -> list[str] | None:
        visited: set[str] = set()
        for start in best_in:
            path: list[str] = []
            node = start
            while node not in visited and node != root and node in best_in:
                visited.add(node)
                path.append(node)
                node = best_in[node][0]
            if node in path:
                return path[path.index(node) :]
        return None

    def cycle_penalty(self) -> float:
        return self.total - self.lower_bound

    def note(self) -> str:
        return (
            f"arborescence from '{self.root}' costs {self.total} against a "
            f"cheapest-in-edge bound of {self.lower_bound}; the gap of "
            f"{self.cycle_penalty()} is the price the cycles imposed"
        )
