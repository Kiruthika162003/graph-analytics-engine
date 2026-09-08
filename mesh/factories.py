"""Graph factories: the named shapes every algorithm gets tested against.

The same handful of graphs appear in every test and every example,
because each one isolates a property: the complete graph has every
edge, the cycle has exactly one, the path is a tree with two leaves,
the star is a tree with one hub, the grid is planar and bipartite with
a large diameter, the wheel is a cycle plus a hub and so has triangles
everywhere, the complete bipartite graph is the densest triangle-free
shape, and the Petersen graph is the standard counterexample, three-
regular, non-planar, with girth five and no Hamiltonian cycle. Building
them in one place, with one naming convention, means a test's intent is
its first line rather than a loop the reader has to decode, and it
means a claimed property can be checked on the factory once instead of
trusted in every file that hand-builds the shape. Each factory takes a
size and an optional label prefix, returns a Graph, and refuses a size
too small for the shape to exist: a cycle needs three nodes, a wheel
four, a grid a side of one or more. The Petersen graph is fixed at ten
nodes and built from its outer pentagon, inner pentagram, and five
spokes. The module also reports, for any factory output, the counts a
reader can verify by hand, nodes and edges, because the edge count of
each shape follows a formula and the formula is the first thing a
factory should get right.
"""

from __future__ import annotations

from itertools import combinations

from mesh.errors import Invalid
from mesh.graph import Graph


def _labelled(count: int, prefix: str) -> tuple[Graph, list[str]]:
    g = Graph()
    nodes = [f"{prefix}{i}" for i in range(count)]
    for n in nodes:
        g.add_node(n)
    return g, nodes


def complete(n: int, prefix: str = "") -> Graph:
    if n < 1:
        raise Invalid("a complete graph needs at least one node")
    g, nodes = _labelled(n, prefix)
    for a, b in combinations(nodes, 2):
        g.add_edge(a, b)
    return g


def cycle(n: int, prefix: str = "") -> Graph:
    if n < 3:
        raise Invalid("a cycle needs at least three nodes")
    g, nodes = _labelled(n, prefix)
    for i in range(n):
        g.add_edge(nodes[i], nodes[(i + 1) % n])
    return g


def path(n: int, prefix: str = "") -> Graph:
    if n < 1:
        raise Invalid("a path needs at least one node")
    g, nodes = _labelled(n, prefix)
    for i in range(n - 1):
        g.add_edge(nodes[i], nodes[i + 1])
    return g


def star(leaves: int, prefix: str = "") -> Graph:
    if leaves < 1:
        raise Invalid("a star needs at least one leaf")
    g, nodes = _labelled(leaves + 1, prefix)
    for leaf in nodes[1:]:
        g.add_edge(nodes[0], leaf)
    return g


def wheel(rim: int, prefix: str = "") -> Graph:
    if rim < 3:
        raise Invalid("a wheel needs a rim of at least three nodes")
    g = cycle(rim, prefix)
    hub = f"{prefix}hub"
    g.add_node(hub)
    for n in list(g.nodes()):
        if n != hub:
            g.add_edge(hub, n)
    return g


def grid(side: int, prefix: str = "") -> Graph:
    if side < 1:
        raise Invalid("a grid needs a side of at least one")
    g = Graph()
    for r in range(side):
        for c in range(side):
            g.add_node(f"{prefix}{r},{c}")
    for r in range(side):
        for c in range(side):
            if c + 1 < side:
                g.add_edge(f"{prefix}{r},{c}", f"{prefix}{r},{c + 1}")
            if r + 1 < side:
                g.add_edge(f"{prefix}{r},{c}", f"{prefix}{r + 1},{c}")
    return g


def complete_bipartite(left: int, right: int, prefix: str = "") -> Graph:
    if left < 1 or right < 1:
        raise Invalid("both sides need at least one node")
    g = Graph()
    lefts = [f"{prefix}l{i}" for i in range(left)]
    rights = [f"{prefix}r{i}" for i in range(right)]
    for n in lefts + rights:
        g.add_node(n)
    for a in lefts:
        for b in rights:
            g.add_edge(a, b)
    return g


def petersen(prefix: str = "") -> Graph:
    # outer pentagon, inner pentagram, and five spokes
    g = Graph()
    outer = [f"{prefix}o{i}" for i in range(5)]
    inner = [f"{prefix}i{i}" for i in range(5)]
    for n in outer + inner:
        g.add_node(n)
    for i in range(5):
        g.add_edge(outer[i], outer[(i + 1) % 5])
        g.add_edge(inner[i], inner[(i + 2) % 5])
        g.add_edge(outer[i], inner[i])
    return g


def note(g: Graph, name: str) -> str:
    return f"{name}: {g.node_count()} node(s), {g.edge_count()} edge(s)"
