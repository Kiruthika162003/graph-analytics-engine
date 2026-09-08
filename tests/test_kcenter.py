from __future__ import annotations

import random
from itertools import combinations, pairwise

import pytest

from mesh.errors import Invalid
from mesh.graph import Graph
from mesh.kcenter import KCenter


def _path(k: int) -> Graph:
    g = Graph()
    nodes = [str(i) for i in range(k)]
    for n in nodes:
        g.add_node(n)
    for a, b in pairwise(nodes):
        g.add_edge(a, b, 1)
    return g


class TestPlacement:
    def test_one_center_on_a_path_lands_at_the_middle(self):
        # with k=1 the traversal never picks a farthest node, so the start is
        # the whole answer; started at node 0 it measured radius 6 against an
        # optimum of 3, the factor-two bound met exactly, which refuted the
        # first version of this test and moved the start to the graph's center
        kc = KCenter(_path(7), k=1)
        assert kc.centers == ["3"]
        assert kc.radius == kc.optimum() == 3

    def test_two_centers_halve_the_reach(self):
        kc = KCenter(_path(8), k=2)
        assert kc.radius <= 2 * kc.optimum()
        assert kc.optimum() == 2

    def test_k_equal_to_node_count_has_radius_zero(self):
        kc = KCenter(_path(4), k=4)
        assert kc.radius == 0

    def test_centers_are_distinct_nodes_of_the_graph(self):
        g = _path(6)
        kc = KCenter(g, k=3)
        assert len(set(kc.centers)) == 3
        assert set(kc.centers) <= set(g.nodes())

    def test_the_radius_is_the_worst_nearest_center_distance(self):
        g = _path(5)
        kc = KCenter(g, k=2)
        worst = max(min(abs(int(n) - int(c)) for c in kc.centers) for n in g.nodes())
        assert kc.radius == worst


class TestGuarantee:
    def test_the_greedy_radius_stays_within_twice_the_optimum(self):
        rng = random.Random(359)
        for _ in range(20):
            g = Graph()
            nodes = [str(i) for i in range(8)]
            for n in nodes:
                g.add_node(n)
            shuffled = nodes[:]
            rng.shuffle(shuffled)
            for a, b in pairwise(shuffled):
                g.add_edge(a, b, rng.randint(1, 9))
            for a, b in combinations(nodes, 2):
                if not g.has_edge(a, b) and rng.random() < 0.2:
                    g.add_edge(a, b, rng.randint(1, 9))
            for k in (1, 2, 3):
                kc = KCenter(g, k)
                best = kc.optimum()
                assert best <= kc.radius <= 2 * best


class TestRefusals:
    def test_a_directed_graph_is_refused(self):
        with pytest.raises(Invalid):
            KCenter(Graph(directed=True), 1)

    def test_k_out_of_range_is_refused(self):
        with pytest.raises(Invalid):
            KCenter(_path(3), 0)
        with pytest.raises(Invalid):
            KCenter(_path(3), 4)

    def test_a_disconnected_graph_is_refused(self):
        g = _path(3)
        g.add_node("island")
        with pytest.raises(Invalid):
            KCenter(g, 1)

    def test_the_exact_optimum_is_capped(self):
        kc = KCenter(_path(13), 2)
        with pytest.raises(Invalid):
            kc.optimum()


class TestReport:
    def test_the_note_states_the_measured_ratio(self):
        note = KCenter(_path(7), 1).note()
        assert "1.00 times the optimum" in note
