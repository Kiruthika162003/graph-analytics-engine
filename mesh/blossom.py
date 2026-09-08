"""Blossom matching: maximum matching in any graph by shrinking the odd cycles that fool it.

Hopcroft-Karp finds maximum matchings in bipartite graphs, and its
augmenting-path search relies on the graph having no odd cycle. In a
general graph an odd cycle can hide an augmenting path from a plain
search: the alternating tree grown from a free node can reach a node by
an even-length path and again by an odd-length one, and the search,
having labelled it once, never sees the second route. Edmonds's insight
was that such an odd cycle, a blossom, can be shrunk to a single node,
the search continued on the contracted graph, and any augmenting path
found there lifted back to the original by walking around the blossom
in the direction that keeps the alternation. The engine implements the
classic breadth-first form: from each free node it grows an alternating
tree, and whenever an edge joins two even-level nodes of the same tree,
it finds their lowest common ancestor as the blossom's base, marks the
cycle, and contracts it by pointing every member's base at that
ancestor, re-queuing members so their edges are explored again. When
the tree reaches a free node, the path is flipped and the matching
grows by one. The matching is maximum when no free node yields a path,
by Berge's theorem again, now proven for general graphs by the
contraction argument. The cost is cubic in the node count in this form,
fine for the graphs a matching question usually involves, and the
result is checked against exhaustive search on small graphs. The
matcher returns the matching and its size, and reports how many
blossoms it contracted, because a run with none was a graph the
bipartite search would have handled and every contraction is an odd
cycle that would have hidden an augmenting path from it.
"""

from __future__ import annotations

from collections import deque

from mesh.errors import Invalid
from mesh.graph import Graph


class Blossom:
    def __init__(self, graph: Graph) -> None:
        if graph.directed:
            raise Invalid("matching is defined on an undirected graph")
        self.graph = graph
        self.nodes = graph.nodes()
        self._idx = {n: i for i, n in enumerate(self.nodes)}
        self._adj = [[self._idx[m] for m in graph.neighbors(n)] for n in self.nodes]
        self._match = [-1] * len(self.nodes)
        self.blossoms = 0
        self._solve()

    def _lca(self, a: int, b: int, base: list[int], parent: list[int]) -> int:
        seen = [False] * len(self.nodes)
        while True:
            a = base[a]
            seen[a] = True
            if self._match[a] == -1:
                break
            a = parent[self._match[a]]
        while True:
            b = base[b]
            if seen[b]:
                return b
            b = parent[self._match[b]]

    def _mark(self, v: int, ancestor: int, child: int, base: list[int],
              parent: list[int], in_blossom: list[bool]) -> None:
        while base[v] != ancestor:
            in_blossom[base[v]] = in_blossom[base[self._match[v]]] = True
            parent[v] = child
            child = self._match[v]
            v = parent[self._match[v]]

    def _find_path(self, root: int) -> tuple[int, list[int]]:
        n = len(self.nodes)
        base = list(range(n))
        parent = [-1] * n
        used = [False] * n
        used[root] = True
        queue: deque[int] = deque([root])
        while queue:
            v = queue.popleft()
            for to in self._adj[v]:
                if base[v] == base[to] or self._match[v] == to:
                    continue
                if to == root or (self._match[to] != -1 and parent[self._match[to]] != -1):
                    # an edge between two even nodes of the tree: a blossom
                    ancestor = self._lca(v, to, base, parent)
                    in_blossom = [False] * n
                    self._mark(v, ancestor, to, base, parent, in_blossom)
                    self._mark(to, ancestor, v, base, parent, in_blossom)
                    self.blossoms += 1
                    for i in range(n):
                        if in_blossom[base[i]]:
                            base[i] = ancestor
                            if not used[i]:
                                used[i] = True
                                queue.append(i)
                elif parent[to] == -1:
                    parent[to] = v
                    if self._match[to] == -1:
                        return to, parent  # a free node: an augmenting path
                    nxt = self._match[to]
                    used[nxt] = True
                    queue.append(nxt)
        return -1, parent

    def _solve(self) -> None:
        for root in range(len(self.nodes)):
            if self._match[root] != -1:
                continue
            v, parent = self._find_path(root)
            while v != -1:
                # flip the alternating path from the free end back to the root:
                # v takes its tree parent, whose old partner continues the walk
                pv = parent[v]
                ppv = self._match[pv]
                self._match[v] = pv
                self._match[pv] = v
                v = ppv

    def matching(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for i, j in enumerate(self._match):
            if j != -1 and i < j:
                out[self.nodes[i]] = self.nodes[j]
        return out

    def size(self) -> int:
        return len(self.matching())

    def is_valid(self) -> bool:
        pairs = self.matching()
        used: set[str] = set()
        for u, v in pairs.items():
            if not self.graph.has_edge(u, v) or u in used or v in used:
                return False
            used.update((u, v))
        return True

    def note(self) -> str:
        return (
            f"matching of size {self.size()} after {self.blossoms} blossom "
            f"contraction(s); none means a bipartite search would have done"
        )
