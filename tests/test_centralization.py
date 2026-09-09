from __future__ import annotations

import random
from itertools import combinations

import pytest

from mesh.centralization import Centralization
from mesh.errors import Invalid
from mesh.factories import complete, cycle, path, star
from mesh.graph import Graph


class TestExtremes:
    def test_a_star_scores_one_on_every_reading(self):
        c = Centralization(star(6))
        assert c.degree() == pytest.approx(1.0)
        assert c.betweenness() == pytest.approx(1.0)
        assert c.harmonic() == pytest.approx(1.0)
        assert c.most_central() == "0"

    def test_cycles_and_complete_graphs_score_zero(self):
        for g in (cycle(6), complete(5)):
            c = Centralization(g)
            assert c.degree() == 0.0
            assert c.betweenness() == pytest.approx(0.0)
            assert c.harmonic() == pytest.approx(0.0)

    def test_a_path_sits_between(self):
        c = Centralization(path(5))
        assert 0 < c.degree() < 1
        assert 0 < c.betweenness() < 1
        assert 0 < c.harmonic() < 1


class TestRange:
    def test_every_reading_stays_within_the_unit_interval_on_random_graphs(self):
        rng = random.Random(911)
        for _ in range(12):
            g = Graph()
            nodes = [str(i) for i in range(8)]
            for n in nodes:
                g.add_node(n)
            for a, b in combinations(nodes, 2):
                if rng.random() < 0.35:
                    g.add_edge(a, b)
            c = Centralization(g)
            for reading in (c.degree(), c.betweenness(), c.harmonic()):
                assert -1e-9 <= reading <= 1 + 1e-9

    def test_adding_a_hub_raises_degree_centralization(self):
        g = cycle(6)
        before = Centralization(g).degree()
        g.add_node("hub")
        for n in cycle(6).nodes():
            g.add_edge("hub", n)
        assert Centralization(g).degree() > before


class TestSmall:
    def test_tiny_graphs_score_zero_and_the_empty_graph_has_no_center(self):
        c = Centralization(path(2))
        assert c.degree() == 0.0
        assert c.betweenness() == 0.0
        assert Centralization(Graph()).most_central() is None

    def test_a_directed_graph_is_refused(self):
        with pytest.raises(Invalid):
            Centralization(Graph(directed=True))


class TestReport:
    def test_the_note_prints_all_three(self):
        note = Centralization(star(4)).note()
        assert "by degree 1.000, betweenness 1.000, harmonic 1.000" in note
        assert "busiest node 0" in note
