"""Edge coloring: schedule the edges so no node is busy twice at once.

An edge coloring gives every edge a color such that no two edges sharing
a node get the same one. It is round-robin scheduling: teams that must
each play the others, with each team playing at most one match per
round, need as many rounds as the edge coloring needs colors. Every
graph needs at least its maximum degree many colors, since the edges at
the busiest node must all differ, and Vizing's theorem says the maximum
degree plus one always suffices, so the chromatic index is one of just
two values; deciding which is NP-hard in general. On a bipartite graph
the answer is known exactly, Konig's edge theorem: the maximum degree
always suffices. The constructive proof is the algorithm here. Color the
edges one by one; for an edge between u and v, find a color free at u
and a color free at v. If some color is free at both, use it. Otherwise
take the color free at u, call it alpha, and the color free at v, beta,
and walk from v along the path that alternates alpha and beta edges;
that path cannot reach u in a bipartite graph, because it would close an
odd cycle, so swapping alpha and beta along it frees alpha at v without
disturbing anything else, and the edge takes alpha. For a general graph
the engine does not have an exact method and says so, falling back to
greedy, which assigns each edge the smallest color free at both ends
and can use up to twice the maximum degree minus one in the worst case.
The colorer verifies the result is proper, reports the color count, and
sets it against the maximum degree and Vizing's bound, because a
bipartite graph landing exactly on the maximum degree is the theorem
made visible, and a general graph landing above the degree plus one is
the greedy fallback paying for its simplicity.
"""

from __future__ import annotations

from mesh.bipartite import Bipartite
from mesh.errors import Invalid
from mesh.graph import Graph


class EdgeColoring:
    def __init__(self, graph: Graph) -> None:
        if graph.directed:
            raise Invalid("edge coloring is defined on an undirected graph")
        self.graph = graph
        self.max_degree = max((graph.degree(n) for n in graph.nodes()), default=0)
        self.exact = Bipartite(graph).is_bipartite
        # color of each edge, keyed by the unordered pair
        self.color: dict[frozenset[str], int] = {}
        # at[node][color] = the neighbor that edge goes to
        self._at: dict[str, dict[int, str]] = {n: {} for n in graph.nodes()}
        if self.exact:
            self._konig()
        else:
            self._greedy()

    def _free(self, node: str, limit: int) -> int:
        c = 0
        while c in self._at[node] and c < limit:
            c += 1
        return c

    def _set(self, u: str, v: str, c: int) -> None:
        self.color[frozenset((u, v))] = c
        self._at[u][c] = v
        self._at[v][c] = u

    def _unset(self, u: str, v: str) -> None:
        c = self.color.pop(frozenset((u, v)))
        del self._at[u][c]
        del self._at[v][c]

    def _konig(self) -> None:
        d = self.max_degree
        for u, v, _w in self.graph.edges():
            alpha = self._free(u, d)
            beta = self._free(v, d)
            if alpha != beta and alpha in self._at[v]:
                self._swap_path(v, alpha, beta)  # free alpha at v
            self._set(u, v, alpha)

    def _swap_path(self, start: str, alpha: int, beta: int) -> None:
        # walk alpha, beta, alpha... from start and exchange the two colors
        path: list[tuple[str, str]] = []
        node, want = start, alpha
        while want in self._at[node]:
            nxt = self._at[node][want]
            path.append((node, nxt))
            node, want = nxt, beta if want == alpha else alpha
        for a, b in path:
            self._unset(a, b)
        # the walk began with an alpha edge, so even steps were alpha: flip each
        for i, (a, b) in enumerate(path):
            self._set(a, b, beta if i % 2 == 0 else alpha)

    def _greedy(self) -> None:
        for u, v, _w in self.graph.edges():
            c = 0
            while c in self._at[u] or c in self._at[v]:
                c += 1
            self._set(u, v, c)

    def is_proper(self) -> bool:
        for node in self.graph.nodes():
            seen = [self.color[frozenset((node, m))] for m in self.graph.neighbors(node)]
            if len(seen) != len(set(seen)):
                return False
        return True

    def color_count(self) -> int:
        return len(set(self.color.values()))

    def vizing_bound(self) -> int:
        return self.max_degree + 1

    def note(self) -> str:
        regime = "exact by Konig" if self.exact else "greedy fallback"
        return (
            f"{self.color_count()} color(s), {regime}, against max degree "
            f"{self.max_degree} and Vizing's bound {self.vizing_bound()}; a general "
            "graph above the bound is greedy paying for its simplicity"
        )
