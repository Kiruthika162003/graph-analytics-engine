"""Clique percolation: communities as chains of overlapping cliques.

Most community detectors partition the nodes, so each belongs to exactly
one group. Real memberships overlap: a person is in a family, a
workplace, and a club at once. Clique percolation is the method built
for that. Fix a clique size k, typically three or four. Find every
k-clique, a set of k nodes all mutually adjacent. Call two k-cliques
adjacent if they share k minus one nodes, so that one can be reached
from the other by swapping a single node. A community is a connected
set of k-cliques under that adjacency, the union of its cliques' nodes,
and because a node can sit in cliques from different chains, it can
belong to several communities, which is the point. Nodes in no k-clique
belong to no community at all, which is honest for a method that
defines community by dense overlap rather than by assigning everyone
somewhere. The engine enumerates maximal cliques with Bron-Kerbosch and
expands each of size at least k into its k-subsets, which is where the
cost lives: a maximal clique of size m contains m choose k k-cliques,
so the method suits sparse graphs with modest cliques and the engine
states that limit. Union-find over the k-cliques with the shared-node
test builds the communities. The detector returns the communities, the
communities of a node, the nodes left out, and reports how many nodes
sit in more than one community, because that count is what percolation
finds that a partition cannot, and a graph where it is zero would have
been served just as well by Louvain.
"""

from __future__ import annotations

from itertools import combinations

from mesh.bronkerbosch import BronKerbosch
from mesh.errors import Invalid, Missing
from mesh.graph import Graph
from mesh.unionfind import UnionFind


class CliquePercolation:
    def __init__(self, graph: Graph, k: int = 3) -> None:
        if graph.directed:
            raise Invalid("clique percolation runs on an undirected graph")
        if k < 2:
            raise Invalid("k must be at least two; a 1-clique is a lone node")
        self.graph = graph
        self.k = k
        self.cliques = self._k_cliques()
        self.communities = self._percolate()

    def _k_cliques(self) -> list[frozenset[str]]:
        found: set[frozenset[str]] = set()
        for maximal in BronKerbosch(self.graph).cliques:
            if len(maximal) < self.k:
                continue
            for subset in combinations(sorted(maximal), self.k):
                found.add(frozenset(subset))
        return sorted(found, key=sorted)

    def _percolate(self) -> list[set[str]]:
        uf = UnionFind()
        keys = [",".join(sorted(c)) for c in self.cliques]
        for key in keys:
            uf.add(key)
        # two k-cliques are adjacent when they share k-1 nodes
        for i, a in enumerate(self.cliques):
            for j in range(i + 1, len(self.cliques)):
                if len(a & self.cliques[j]) == self.k - 1:
                    uf.union(keys[i], keys[j])
        groups: dict[str, set[str]] = {}
        for key, clique in zip(keys, self.cliques, strict=True):
            groups.setdefault(uf.find(key), set()).update(clique)
        return sorted(groups.values(), key=lambda s: (-len(s), sorted(s)))

    def communities_of(self, node: str) -> list[set[str]]:
        if not self.graph.has_node(node):
            raise Missing(f"node '{node}' is not in the graph")
        return [c for c in self.communities if node in c]

    def left_out(self) -> set[str]:
        covered = set().union(*self.communities) if self.communities else set()
        return set(self.graph.nodes()) - covered

    def overlapping_nodes(self) -> set[str]:
        return {n for n in self.graph.nodes() if len(self.communities_of(n)) > 1}

    def note(self) -> str:
        return (
            f"{len(self.communities)} communit(ies) from {len(self.cliques)} "
            f"{self.k}-clique(s), {len(self.overlapping_nodes())} node(s) in more than "
            f"one, {len(self.left_out())} in none; zero overlap would have been served "
            "by a partition"
        )
