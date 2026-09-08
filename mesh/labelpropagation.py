"""Label propagation: let every node adopt its neighbors' majority label until it settles.

Community detection asks for groups of nodes that are densely connected
inside and sparsely connected between, without being told how many groups
there are or where they lie. Label propagation is the simplest algorithm
that finds them and it runs in near-linear time. Every node starts with
its own unique label. Then, repeatedly, each node looks at the labels its
neighbors currently hold and adopts the most common one, breaking ties at
random. A label that is common in a dense region spreads through it fast,
because most of a node's neighbors are inside the region and share it,
while it struggles to cross a sparse boundary, because a node on the far
side has most of its neighbors carrying the other side's label. After a
few sweeps the labels stop changing, and the nodes sharing a label are a
community. The randomness is real and matters: the tie-breaks and the
order nodes are updated in can change the result, so two runs can give
different partitions of the same graph, and a graph with no community
structure can be split arbitrarily. This engine takes a seed so a result
is reproducible, updates nodes asynchronously in a shuffled order each
sweep, which converges where synchronous updates can oscillate on a
bipartite piece, and stops when a sweep changes nothing or the sweep cap is
hit. The detector returns the communities, the label of a node, the sweep
count, and reports the modularity of the partition, the standard score of
how much denser the communities are inside than a random graph with the
same degrees would be, because a modularity near zero says the labels
settled into a split the structure does not support.
"""

from __future__ import annotations

import random
from collections import Counter

from mesh.errors import Invalid, Missing
from mesh.graph import Graph


class LabelPropagation:
    def __init__(self, graph: Graph, seed: int = 0, max_sweeps: int = 100) -> None:
        if graph.directed:
            raise Invalid("label propagation here runs on an undirected graph")
        self.graph = graph
        self._rng = random.Random(seed)
        self.label: dict[str, str] = {n: n for n in graph.nodes()}
        self.sweeps = 0
        self._run(max_sweeps)

    def _run(self, max_sweeps: int) -> None:
        nodes = self.graph.nodes()
        for _ in range(max_sweeps):
            self.sweeps += 1
            self._rng.shuffle(nodes)  # asynchronous updates in a fresh order
            changed = False
            for node in nodes:
                nbrs = self.graph.neighbors(node)
                if not nbrs:
                    continue
                counts = Counter(self.label[m] for m in nbrs)
                top = max(counts.values())
                best = [lab for lab, c in counts.items() if c == top]
                chosen = self._rng.choice(sorted(best))
                if chosen != self.label[node]:
                    self.label[node] = chosen
                    changed = True
            if not changed:
                break

    def communities(self) -> list[set[str]]:
        groups: dict[str, set[str]] = {}
        for node, lab in self.label.items():
            groups.setdefault(lab, set()).add(node)
        return sorted(groups.values(), key=lambda s: (-len(s), sorted(s)))

    def community_of(self, node: str) -> set[str]:
        if node not in self.label:
            raise Missing(f"node '{node}' is not in the graph")
        return {n for n, lab in self.label.items() if lab == self.label[node]}

    def modularity(self) -> float:
        m = self.graph.edge_count()
        if m == 0:
            return 0.0
        degree = {n: self.graph.degree(n) for n in self.graph.nodes()}
        total = 0.0
        for u in self.graph.nodes():
            for v in self.graph.nodes():
                if self.label[u] != self.label[v]:
                    continue
                actual = 1.0 if self.graph.has_edge(u, v) else 0.0
                expected = degree[u] * degree[v] / (2 * m)
                total += actual - expected
        return total / (2 * m)

    def note(self) -> str:
        return (
            f"{len(self.communities())} communit(ies) after {self.sweeps} sweep(s), "
            f"modularity {self.modularity():.3f}; near zero means a split the "
            "structure does not support"
        )
