from __future__ import annotations

from itertools import pairwise

import pytest

from mesh.errors import Invalid
from mesh.factories import cycle, path, star
from mesh.graph import Graph
from mesh.sampling import Sampler


def _hub_and_ring() -> Graph:
    g = cycle(10)
    g.add_node("hub")
    for n in cycle(10).nodes():
        g.add_edge("hub", n)
    return g


class TestMethods:
    def test_node_sampling_keeps_exactly_the_count_and_only_inner_edges(self):
        s = Sampler(cycle(8), seed=1)
        sample = s.by_nodes(4)
        assert sample.node_count() == 4
        for u, v, _w in sample.edges():
            assert cycle(8).has_edge(u, v)

    def test_edge_sampling_keeps_the_chosen_endpoints(self):
        s = Sampler(star(6), seed=2)
        sample = s.by_edges(3)
        assert sample.edge_count() >= 3
        assert "0" in sample.nodes()
        assert sample.node_count() == 4

    def test_snowball_takes_everything_within_the_hops(self):
        s = Sampler(path(9), seed=3)
        sample = s.snowball("4", 2)
        assert sorted(sample.nodes()) == ["2", "3", "4", "5", "6"]
        assert sample.edge_count() == 4
        assert s.snowball("4", 0).node_count() == 1

    def test_a_walk_follows_edges_step_by_step(self):
        s = Sampler(cycle(6), seed=4)
        visits = s.walk(30, start="0")
        assert len(visits) == 31
        for a, b in pairwise(visits):
            assert cycle(6).has_edge(a, b)


class TestBias:
    def test_the_walk_visits_the_hub_in_proportion_to_its_degree(self):
        g = _hub_and_ring()
        s = Sampler(g, seed=5)
        shares = s.visit_shares(20000, start="hub")
        degree_share = g.degree("hub") / (2 * g.edge_count())
        assert shares["hub"] == pytest.approx(degree_share, abs=0.03)

    def test_degree_correction_flattens_the_walk_to_uniform(self):
        g = _hub_and_ring()
        s = Sampler(g, seed=6)
        corrected = s.corrected_shares(20000, start="hub")
        assert corrected["hub"] == pytest.approx(1 / 11, abs=0.03)
        assert corrected["3"] == pytest.approx(1 / 11, abs=0.03)

    def test_edge_sampling_leans_toward_hubs_more_than_node_sampling(self):
        g = _hub_and_ring()
        s = Sampler(g, seed=7)
        by_edges = sum(s.degree_bias(s.by_edges(4)) for _ in range(20)) / 20
        by_nodes = sum(s.degree_bias(s.by_nodes(4)) for _ in range(20)) / 20
        assert by_edges > by_nodes


class TestDirected:
    def test_a_walker_stuck_in_a_sink_restarts_and_the_restarts_are_counted(self):
        g = Graph(directed=True)
        for n in "abc":
            g.add_node(n)
        g.add_edge("a", "b")
        g.add_edge("b", "c")
        s = Sampler(g, seed=8)
        s.walk(10, start="a")
        assert s.restarts > 0


class TestRefusal:
    def test_bad_counts_seeds_and_hops_are_refused(self):
        s = Sampler(path(3))
        with pytest.raises(Invalid):
            s.by_nodes(4)
        with pytest.raises(Invalid):
            s.by_edges(-1)
        with pytest.raises(Invalid):
            s.snowball("zz", 1)
        with pytest.raises(Invalid):
            s.snowball("0", -1)
        with pytest.raises(Invalid):
            s.walk(-1)
        with pytest.raises(Invalid):
            s.walk(2, start="zz")

    def test_an_empty_graph_walks_nowhere(self):
        s = Sampler(Graph())
        assert s.walk(5) == []
        assert s.degree_bias(Graph()) == 0.0


class TestReport:
    def test_the_note_names_the_method_and_the_bias(self):
        s = Sampler(star(4), seed=9)
        note = s.note(s.snowball("0", 1), "snowball")
        assert note.startswith("snowball sample of 5 node(s) and 4 edge(s)")
        assert "mean original degree 1.00x" in note
