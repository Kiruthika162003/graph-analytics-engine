"""Louvain: climb modularity by moving nodes, then collapse communities and climb again.

Modularity scores a partition by how many more edges fall inside its
communities than a random graph with the same degrees would put there,
and Louvain is the method that made optimizing it practical on large
networks. It works in two alternating phases. In the local moving phase
every node, visited in turn, considers leaving its community for the
community of each neighbor and takes the move with the largest gain in
modularity, staying put if no move helps; the gain has a closed form in
terms of the edge weight from the node into the candidate community, the
community's total degree, and the node's own degree, so each evaluation
is cheap. The sweep repeats until no node moves. In the aggregation
phase each community collapses to a single node, edges between
communities become weighted edges between the new nodes, and edges inside
a community become a self-loop of their total weight, so the modularity
of the collapsed graph equals the modularity of the partition it
represents. The moving phase then runs again on the collapsed graph,
merging whole communities, and the two phases alternate until a full
pass produces no improvement, which gives a hierarchy of partitions
from fine to coarse. Louvain is a heuristic, greedy at every step, and
the result depends on the order nodes are visited; the engine fixes that
order by a seeded shuffle so a run is reproducible, and states the
limit rather than presenting the partition as optimal. The detector
returns the communities, the modularity reached, the number of levels
the hierarchy took, and compares its modularity against the trivial
partition of one community per node, because that comparison is what
says the structure found is real and not the score of an arbitrary split.
"""

from __future__ import annotations

import random

from mesh.errors import Invalid, Missing
from mesh.graph import Graph


class Louvain:
    def __init__(self, graph: Graph, seed: int = 0) -> None:
        if graph.directed:
            raise Invalid("Louvain here runs on an undirected graph")
        if graph.edge_count() == 0:
            raise Invalid("with no edges there is no modularity to climb")
        self.graph = graph
        self._rng = random.Random(seed)
        self.levels = 0
        # membership of original nodes, refined level by level
        self.community: dict[str, int] = {n: i for i, n in enumerate(graph.nodes())}
        self._run()

    def _run(self) -> None:
        # weighted adjacency of the current (possibly collapsed) graph
        adj: dict[str, dict[str, float]] = {
            n: dict(self.graph.neighbors(n)) for n in self.graph.nodes()
        }
        mapping = {n: n for n in self.graph.nodes()}  # original node -> current node
        while True:
            self.levels += 1
            assignment = self._local_moving(adj)
            distinct = len(set(assignment.values()))
            if distinct == len(adj):
                break  # no node moved anywhere: the hierarchy is done
            adj = self._aggregate(adj, assignment)
            mapping = {orig: str(assignment[cur]) for orig, cur in mapping.items()}
            self.community = {orig: int(cur) for orig, cur in mapping.items()}

    def _local_moving(self, adj: dict[str, dict[str, float]]) -> dict[str, int]:
        nodes = list(adj)
        two_m = sum(sum(nbrs.values()) for nbrs in adj.values())
        degree = {n: sum(adj[n].values()) for n in nodes}
        comm = {n: i for i, n in enumerate(nodes)}
        comm_degree = dict(enumerate(degree[n] for n in nodes))
        moved = True
        while moved:
            moved = False
            self._rng.shuffle(nodes)
            for n in nodes:
                own = comm[n]
                comm_degree[own] -= degree[n]
                # weight from n into each neighboring community
                into: dict[int, float] = {}
                for nbr, w in adj[n].items():
                    if nbr != n:
                        into[comm[nbr]] = into.get(comm[nbr], 0.0) + w
                best, best_gain = own, into.get(own, 0.0) - comm_degree[own] * degree[n] / two_m
                for c, w in into.items():
                    gain = w - comm_degree[c] * degree[n] / two_m
                    if gain > best_gain + 1e-12:
                        best, best_gain = c, gain
                comm_degree[best] += degree[n]
                if best != own:
                    comm[n] = best
                    moved = True
        # relabel communities densely
        labels = {c: i for i, c in enumerate(sorted(set(comm.values())))}
        return {n: labels[c] for n, c in comm.items()}

    @staticmethod
    def _aggregate(
        adj: dict[str, dict[str, float]], assignment: dict[str, int]
    ) -> dict[str, dict[str, float]]:
        collapsed: dict[str, dict[str, float]] = {}
        for u, nbrs in adj.items():
            cu = str(assignment[u])
            collapsed.setdefault(cu, {})
            for v, w in nbrs.items():
                cv = str(assignment[v])
                collapsed[cu][cv] = collapsed[cu].get(cv, 0.0) + w
        return collapsed

    def communities(self) -> list[set[str]]:
        groups: dict[int, set[str]] = {}
        for n, c in self.community.items():
            groups.setdefault(c, set()).add(n)
        return sorted(groups.values(), key=lambda s: (-len(s), sorted(s)))

    def community_of(self, node: str) -> set[str]:
        if node not in self.community:
            raise Missing(f"node '{node}' is not in the graph")
        return {n for n, c in self.community.items() if c == self.community[node]}

    def modularity(self, membership: dict[str, int] | None = None) -> float:
        member = membership if membership is not None else self.community
        m = self.graph.edge_count()
        degree = {n: self.graph.degree(n) for n in self.graph.nodes()}
        total = 0.0
        for u in self.graph.nodes():
            for v in self.graph.nodes():
                if member[u] == member[v]:
                    actual = 1.0 if self.graph.has_edge(u, v) else 0.0
                    total += actual - degree[u] * degree[v] / (2 * m)
        return total / (2 * m)

    def trivial_modularity(self) -> float:
        return self.modularity({n: i for i, n in enumerate(self.graph.nodes())})

    def note(self) -> str:
        return (
            f"{len(self.communities())} communit(ies) in {self.levels} level(s), "
            f"modularity {self.modularity():.3f} against {self.trivial_modularity():.3f} "
            "for one node per community; greedy, so a local optimum not the global"
        )
