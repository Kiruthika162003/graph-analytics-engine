from __future__ import annotations

import random
from itertools import combinations

import pytest

from mesh.errors import Invalid
from mesh.factories import complete, cycle, path, star
from mesh.graph import Graph
from mesh.nullmodel import NullModel, assortativity, transitivity


def _ring_of_cliques(cliques: int, size: int) -> Graph:
    g = Graph()
    for c in range(cliques):
        names = [f"c{c}n{i}" for i in range(size)]
        for n in names:
            g.add_node(n)
        for a, b in combinations(names, 2):
            g.add_edge(a, b)
    for c in range(cliques):
        g.add_edge(f"c{c}n0", f"c{(c + 1) % cliques}n1")
    return g


def _random_graph(seed: int, n: int, p: float) -> Graph:
    rng = random.Random(seed)
    g = Graph()
    nodes = [str(i) for i in range(n)]
    for node in nodes:
        g.add_node(node)
    for a, b in combinations(nodes, 2):
        if rng.random() < p:
            g.add_edge(a, b)
    return g


class TestMeasures:
    def test_transitivity_is_one_on_a_clique_and_zero_on_a_tree(self):
        assert transitivity(complete(5)) == 1.0
        assert transitivity(star(4)) == 0.0
        assert transitivity(path(2)) == 0.0

    def test_assortativity_is_negative_on_a_star_and_zero_on_a_regular_graph(self):
        assert assortativity(star(5)) < 0
        assert assortativity(cycle(6)) == 0.0
        assert assortativity(Graph()) == 0.0


class TestComparison:
    def test_a_ring_of_cliques_clusters_far_beyond_its_degrees(self):
        nm = NullModel(_ring_of_cliques(6, 4), samples=20, seed=3)
        t = nm.transitivity()
        assert t["observed"] > t["mean"]
        assert t["z"] > 2
        assert t["p"] == 0.0
        assert nm.degrees_preserved
        assert "clusters beyond its degrees" in nm.note()

    def test_a_random_graph_sits_near_its_own_null(self):
        nm = NullModel(_random_graph(971, 24, 0.2), samples=20, seed=5)
        t = nm.transitivity()
        assert abs(t["z"]) < 2.5
        assert nm.degrees_preserved
        assert "as its degrees allow" in nm.note()

    def test_every_shuffle_keeps_the_degree_sequence(self):
        g = _random_graph(977, 16, 0.3)
        nm = NullModel(g, samples=10, seed=7)
        for shuffled in nm.shuffles:
            assert sorted(shuffled.degree(n) for n in g.nodes()) == sorted(
                g.degree(n) for n in g.nodes()
            )

    def test_assortativity_comparison_reports_all_fields(self):
        report = NullModel(_ring_of_cliques(4, 3), samples=8, seed=1).assortativity()
        assert set(report) == {"observed", "mean", "spread", "z", "p"}
        assert 0.0 <= report["p"] <= 1.0


class TestRefusal:
    def test_directed_graphs_and_single_samples_are_refused(self):
        with pytest.raises(Invalid):
            NullModel(Graph(directed=True))
        with pytest.raises(Invalid):
            NullModel(cycle(4), samples=1)

    def test_the_same_seed_reproduces_the_report(self):
        g = _random_graph(983, 12, 0.3)
        a = NullModel(g, samples=6, seed=9).transitivity()
        b = NullModel(g, samples=6, seed=9).transitivity()
        assert a == b
