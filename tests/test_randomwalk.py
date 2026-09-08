from __future__ import annotations

from itertools import combinations

import pytest

from mesh.errors import Invalid, Missing
from mesh.graph import Graph
from mesh.randomwalk import RandomWalk


def _hub_graph() -> Graph:
    # a triangle a,b,c with a pendant d on a: degrees 3,2,2,1
    g = Graph()
    for n in "abcd":
        g.add_node(n)
    for x, y in combinations("abc", 2):
        g.add_edge(x, y)
    g.add_edge("a", "d")
    return g


class TestStationary:
    def test_the_stationary_law_is_degree_over_twice_edges(self):
        s = RandomWalk(_hub_graph(), "a", steps=10).stationary()
        assert s["a"] == pytest.approx(3 / 8)
        assert s["d"] == pytest.approx(1 / 8)

    def test_the_stationary_distribution_sums_to_one(self):
        s = RandomWalk(_hub_graph(), "a", steps=10).stationary()
        assert sum(s.values()) == pytest.approx(1.0)

    def test_a_long_walk_lands_near_the_theory(self):
        w = RandomWalk(_hub_graph(), "a", steps=50000, seed=1)
        assert w.total_variation() < 3 * w.noise_scale()

    def test_the_busiest_node_by_degree_is_visited_most(self):
        w = RandomWalk(_hub_graph(), "b", steps=20000, seed=2)
        assert max(w.visits, key=w.visits.get) == "a"

    def test_a_short_walk_can_sit_far_from_theory(self):
        # first guess: a few dozen steps is plenty. Measured: it is not.
        short = RandomWalk(_hub_graph(), "d", steps=20, seed=3)
        long = RandomWalk(_hub_graph(), "d", steps=50000, seed=3)
        assert short.total_variation() > long.total_variation()


class TestCounts:
    def test_visits_sum_to_the_step_count(self):
        w = RandomWalk(_hub_graph(), "a", steps=777, seed=4)
        assert sum(w.visits.values()) == 777

    def test_the_same_seed_reproduces_the_walk(self):
        a = RandomWalk(_hub_graph(), "a", steps=500, seed=9).visits
        b = RandomWalk(_hub_graph(), "a", steps=500, seed=9).visits
        assert a == b


class TestBipartite:
    def test_a_bipartite_graph_is_flagged_as_alternating(self):
        g = Graph()
        for n in "abcd":
            g.add_node(n)
        for u, v in [("a", "b"), ("b", "c"), ("c", "d"), ("d", "a")]:
            g.add_edge(u, v)
        w = RandomWalk(g, "a", steps=100)
        assert w.alternates
        assert "alternates" in w.note()


class TestRefusals:
    def test_a_directed_graph_is_refused(self):
        with pytest.raises(Invalid):
            RandomWalk(Graph(directed=True), "a", steps=1)

    def test_a_missing_start_is_refused(self):
        with pytest.raises(Missing):
            RandomWalk(_hub_graph(), "ghost", steps=1)

    def test_zero_steps_is_refused(self):
        with pytest.raises(Invalid):
            RandomWalk(_hub_graph(), "a", steps=0)

    def test_a_disconnected_graph_is_refused(self):
        g = _hub_graph()
        g.add_node("island")
        with pytest.raises(Invalid):
            RandomWalk(g, "a", steps=10)


class TestReport:
    def test_the_note_gives_a_mixing_verdict(self):
        note = RandomWalk(_hub_graph(), "a", steps=50000, seed=5).note()
        assert "total variation" in note
        assert "mixed" in note
