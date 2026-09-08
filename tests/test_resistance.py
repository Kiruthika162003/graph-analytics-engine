from __future__ import annotations

import random
from itertools import combinations

import pytest

from mesh.errors import Invalid, Missing
from mesh.factories import complete, cycle, path
from mesh.graph import Graph
from mesh.resistance import EffectiveResistance


class TestLaws:
    def test_a_path_reads_its_length_in_series(self):
        assert EffectiveResistance(path(5)).between("0", "4") == pytest.approx(4.0)

    def test_two_parallel_paths_read_half(self):
        # a 6-cycle: from 0 to 3 there are two paths of length 3 in parallel
        assert EffectiveResistance(cycle(6)).between("0", "3") == pytest.approx(1.5)

    def test_a_complete_graph_has_resistance_two_over_n(self):
        assert EffectiveResistance(complete(5)).between("0", "1") == pytest.approx(2 / 5)

    def test_resistance_is_symmetric_and_zero_to_itself(self):
        er = EffectiveResistance(cycle(5))
        assert er.between("0", "2") == pytest.approx(er.between("2", "0"))
        assert er.between("1", "1") == 0.0

    def test_a_heavier_edge_conducts_more(self):
        g = Graph()
        g.add_node("a")
        g.add_node("b")
        g.add_edge("a", "b", 4)
        assert EffectiveResistance(g).between("a", "b") == pytest.approx(0.25)


class TestTriangleInequality:
    def test_resistance_obeys_the_triangle_inequality_on_random_graphs(self):
        rng = random.Random(419)
        for _ in range(15):
            g = Graph()
            nodes = [str(i) for i in range(7)]
            for n in nodes:
                g.add_node(n)
            for i in range(6):
                g.add_edge(nodes[i], nodes[i + 1])
            for a, b in combinations(nodes, 2):
                if not g.has_edge(a, b) and rng.random() < 0.3:
                    g.add_edge(a, b)
            er = EffectiveResistance(g)
            for a, b, c in combinations(nodes, 3):
                assert er.between(a, c) <= er.between(a, b) + er.between(b, c) + 1e-9


class TestCommuteAndRedundancy:
    def test_commute_time_is_twice_edges_times_resistance(self):
        er = EffectiveResistance(path(4))
        assert er.commute_time("0", "3") == pytest.approx(2 * 3 * 3)

    def test_redundancy_is_one_on_a_single_route_and_below_on_parallel(self):
        assert EffectiveResistance(path(4)).redundancy("0", "3") == pytest.approx(1.0)
        assert EffectiveResistance(cycle(6)).redundancy("0", "3") == pytest.approx(0.5)


class TestRefusals:
    def test_a_directed_graph_is_refused(self):
        with pytest.raises(Invalid):
            EffectiveResistance(Graph(directed=True))

    def test_a_disconnected_graph_is_refused(self):
        g = path(3)
        g.add_node("island")
        with pytest.raises(Invalid):
            EffectiveResistance(g)

    def test_a_missing_node_is_refused(self):
        with pytest.raises(Missing):
            EffectiveResistance(path(3)).between("0", "ghost")

    def test_a_non_positive_weight_is_refused(self):
        g = Graph()
        g.add_node("a")
        g.add_node("b")
        g.add_edge("a", "b", 0)
        with pytest.raises(Invalid):
            EffectiveResistance(g).between("a", "b")


class TestReport:
    def test_the_note_states_resistance_and_redundancy(self):
        note = EffectiveResistance(cycle(6)).note("0", "3")
        assert "resistance 1.500" in note
        assert "redundancy 0.50" in note
