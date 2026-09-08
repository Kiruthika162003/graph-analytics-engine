"""Prufer sequences: every labeled tree is a short list of numbers, and back again.

A labeled tree on n nodes can be written as a sequence of n minus two
labels, and every such sequence names exactly one tree: the Prufer
code is a bijection, and counting the sequences, n to the power n minus
two, is the shortest proof of Cayley's formula for the number of labeled
trees. Encoding removes leaves: repeatedly take the leaf with the
smallest label, write down its one neighbor, and delete it, until two
nodes remain; the written neighbors are the code. A node appears in the
code exactly its degree minus one times, so leaves never appear and the
hub of a star appears every time. Decoding reverses it with the same
rule. Keep the degree of each node as one plus its count in the code;
for each code entry, find the smallest-labeled node of degree one, join
it to the entry, and lower both degrees; when the code is exhausted,
exactly two nodes of degree one remain and they are joined. The
smallest-leaf rule on both sides is what makes the round trip exact,
because the decoder reconstructs the encoder's choices in the same
order. Beyond the counting proof, the code is a compact way to store,
transmit, or randomly generate trees: a uniformly random sequence
decodes to a uniformly random labeled tree, which is how the engine's
random tree generator draws without bias. The coder encodes a tree,
decodes a sequence, refuses a graph that is not a tree and a sequence
whose entries fall outside the label range, and reports the code beside
the degrees it implies, because a label's count plus one is its degree
and that identity is the check that the encoding was read correctly.
"""

from __future__ import annotations

import heapq
import random
from collections import Counter

from mesh.connectedcomponents import ConnectedComponents
from mesh.errors import Invalid
from mesh.graph import Graph


class Prufer:
    @staticmethod
    def encode(tree: Graph) -> list[int]:
        n = tree.node_count()
        if tree.directed or tree.edge_count() != n - 1 or n < 2:
            raise Invalid("Prufer coding needs an undirected tree on at least two nodes")
        if not ConnectedComponents(tree).is_connected():
            raise Invalid("the graph is disconnected, so it is not a tree")
        try:
            ordered = sorted(tree.nodes(), key=int)
        except ValueError as exc:
            raise Invalid("Prufer labels must be integers written as node names") from exc
        labels = {node: i for i, node in enumerate(ordered)}
        degree = {node: tree.degree(node) for node in tree.nodes()}
        nbrs = {node: set(tree.neighbors(node)) for node in tree.nodes()}
        leaves = [labels[node] for node, d in degree.items() if d == 1]
        heapq.heapify(leaves)
        by_label = {i: node for node, i in labels.items()}
        code: list[int] = []
        for _ in range(n - 2):
            # the smallest leaf goes; its one neighbor is written down
            leaf = by_label[heapq.heappop(leaves)]
            parent = next(iter(nbrs[leaf]))
            code.append(labels[parent])
            nbrs[parent].discard(leaf)
            degree[parent] -= 1
            if degree[parent] == 1:
                heapq.heappush(leaves, labels[parent])
        return code

    @staticmethod
    def decode(code: list[int]) -> Graph:
        n = len(code) + 2
        if any(not 0 <= c < n for c in code):
            raise Invalid(f"every code entry must lie in 0..{n - 1}")
        degree = [1] * n
        for c in code:
            degree[c] += 1
        leaves = [i for i in range(n) if degree[i] == 1]
        heapq.heapify(leaves)
        g = Graph()
        for i in range(n):
            g.add_node(str(i))
        for c in code:
            leaf = heapq.heappop(leaves)
            g.add_edge(str(leaf), str(c))
            degree[c] -= 1
            if degree[c] == 1:
                heapq.heappush(leaves, c)
        a, b = heapq.heappop(leaves), heapq.heappop(leaves)
        g.add_edge(str(a), str(b))
        return g

    @staticmethod
    def random_tree(n: int, seed: int = 0) -> Graph:
        if n < 2:
            raise Invalid("a random tree needs at least two nodes")
        rng = random.Random(seed)
        return Prufer.decode([rng.randrange(n) for _ in range(n - 2)])

    @staticmethod
    def implied_degrees(code: list[int]) -> dict[int, int]:
        n = len(code) + 2
        counts = Counter(code)
        return {i: counts.get(i, 0) + 1 for i in range(n)}

    @staticmethod
    def note(code: list[int]) -> str:
        degrees = Prufer.implied_degrees(code)
        hub = max(degrees, key=lambda i: (degrees[i], -i))
        return (
            f"code {code} names one tree on {len(code) + 2} labeled nodes; a label's count "
            f"plus one is its degree, so {hub} has degree {degrees[hub]}"
        )
