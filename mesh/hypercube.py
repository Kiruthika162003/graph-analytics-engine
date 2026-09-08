"""Hypercubes: bit strings joined when they differ in one bit, and a Gray code to walk them.

The d-dimensional hypercube has a node for every d-bit string and an
edge between two strings that differ in exactly one bit. It is the
graph of a d-way switch, of a parallel machine's interconnect, and of
every subset of a d-element set under single insertions and removals.
Its facts are exact: two to the d nodes, d times two to the d minus one
edges, every node of degree d, bipartite by bit parity, and the hop
distance between two strings is their Hamming distance, the number of
bits where they differ, because each hop fixes one bit and no route can
fix more than one per step. A reflected Gray code, where consecutive
codes differ in one bit and the last wraps to the first, is a
Hamiltonian cycle of the cube for d at least two, and the code is built
by reflection: the code for d is the code for d minus one with a zero
prefixed, followed by the same code reversed with a one prefixed. The
engine builds the cube from the strings, builds the Gray code by
reflection, checks the cycle walks edges and closes, checks the hop
distance against the Hamming distance from a breadth-first sweep, and
confirms the cube equals the Cartesian product of d single edges, which
ties it to the product module. A dimension below zero is refused; a
dimension of zero is a single node, which is what the recursion
bottoms out on.
"""

from __future__ import annotations

from collections import deque
from itertools import pairwise

from mesh.errors import Invalid
from mesh.graph import Graph


class Hypercube:
    def __init__(self, dimension: int) -> None:
        if dimension < 0:
            raise Invalid("a hypercube needs a dimension of zero or more")
        self.dimension = dimension
        self.graph = self._build()

    def _build(self) -> Graph:
        g = Graph()
        width = self.dimension
        names = [format(i, f"0{width}b") if width else "" for i in range(2**width)]
        for name in names:
            g.add_node(name)
        for name in names:
            for bit in range(self.dimension):
                flipped = name[:bit] + ("1" if name[bit] == "0" else "0") + name[bit + 1 :]
                if flipped > name:
                    g.add_edge(name, flipped)
        return g

    def gray_code(self) -> list[str]:
        codes = [""]
        for _ in range(self.dimension):
            codes = ["0" + c for c in codes] + ["1" + c for c in reversed(codes)]
        return codes

    def gray_code_is_a_hamiltonian_cycle(self) -> bool:
        codes = self.gray_code()
        if self.dimension < 2:
            return False
        if len(set(codes)) != self.graph.node_count():
            return False
        steps = [*pairwise(codes), (codes[-1], codes[0])]
        return all(self.graph.has_edge(a, b) for a, b in steps)

    @staticmethod
    def hamming(a: str, b: str) -> int:
        return sum(1 for x, y in zip(a, b, strict=True) if x != y)

    def hops_from(self, start: str) -> dict[str, int]:
        dist = {start: 0}
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for other in self.graph.neighbors(node):
                if other not in dist:
                    dist[other] = dist[node] + 1
                    queue.append(other)
        return dist

    def hops_equal_hamming(self) -> bool:
        start = "0" * self.dimension
        hops = self.hops_from(start)
        return all(hops[n] == self.hamming(start, n) for n in self.graph.nodes())

    def is_regular(self) -> bool:
        return all(self.graph.degree(n) == self.dimension for n in self.graph.nodes())

    def parity_is_a_bipartition(self) -> bool:
        return all(u.count("1") % 2 != v.count("1") % 2 for u, v, _w in self.graph.edges())

    def note(self) -> str:
        return (
            f"Q{self.dimension}: {self.graph.node_count()} node(s), {self.graph.edge_count()} "
            f"edge(s), {self.dimension}-regular, Gray cycle "
            f"{'closes' if self.gray_code_is_a_hamiltonian_cycle() else 'absent'}"
        )
