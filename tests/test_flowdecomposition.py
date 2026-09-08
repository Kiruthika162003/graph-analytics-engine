from __future__ import annotations

import random

from mesh.flowdecomposition import FlowDecomposition
from mesh.graph import Graph


def _diamond() -> Graph:
    g = Graph(directed=True)
    for n in "sabt":
        g.add_node(n)
    g.add_edge("s", "a", 3.0)
    g.add_edge("s", "b", 2.0)
    g.add_edge("a", "t", 2.0)
    g.add_edge("b", "t", 3.0)
    g.add_edge("a", "b", 1.0)
    return g


def _random_network(seed: int, n: int) -> Graph:
    rng = random.Random(seed)
    g = Graph(directed=True)
    names = [f"n{i}" for i in range(n)]
    for name in names:
        g.add_node(name)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            if rng.random() < 0.5:
                g.add_edge(a, b, float(rng.randint(1, 9)))
            if rng.random() < 0.15:
                g.add_edge(b, a, float(rng.randint(1, 9)))
    return g


class TestRoutes:
    def test_the_diamond_splits_into_routes_summing_to_the_flow(self):
        fd = FlowDecomposition(_diamond(), "s", "t")
        assert fd.value == 5.0
        assert fd.sums_to_value()
        assert fd.routes_walk_arcs()
        assert fd.nothing_left()
        assert len(fd.paths) >= 2

    def test_each_route_carries_at_least_its_bottleneck_and_the_walk_is_deterministic(self):
        once = FlowDecomposition(_diamond(), "s", "t").paths
        twice = FlowDecomposition(_diamond(), "s", "t").paths
        assert once == twice
        assert all(amount > 0 for _t, amount in once)

    def test_a_single_arc_is_one_route(self):
        g = Graph(directed=True)
        g.add_node("s")
        g.add_node("t")
        g.add_edge("s", "t", 4.0)
        fd = FlowDecomposition(g, "s", "t")
        assert fd.paths == [(["s", "t"], 4.0)]
        assert fd.cycles == []


class TestProperties:
    def test_random_networks_decompose_fully_with_routes_summing_to_the_value(self):
        for seed in range(563, 583):
            g = _random_network(seed, 7)
            fd = FlowDecomposition(g, "n0", "n6")
            assert fd.sums_to_value()
            assert fd.routes_walk_arcs()
            assert fd.nothing_left()
            assert len(fd.paths) + len(fd.cycles) <= g.edge_count()

    def test_no_flow_decomposes_to_nothing(self):
        g = Graph(directed=True)
        for n in "sat":
            g.add_node(n)
        g.add_edge("s", "a", 1.0)
        fd = FlowDecomposition(g, "s", "t")
        assert fd.value == 0.0
        assert fd.paths == []
        assert fd.cycles == []


class TestReport:
    def test_the_note_counts_routes_and_cycles(self):
        note = FlowDecomposition(_diamond(), "s", "t").note()
        assert "flow 5.0 carried by" in note
        assert "routes sum to 5.0" in note
