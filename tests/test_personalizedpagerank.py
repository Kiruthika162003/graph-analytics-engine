from __future__ import annotations

from itertools import combinations

import pytest

from mesh.errors import Invalid, Missing
from mesh.graph import Graph
from mesh.pagerank import PageRank
from mesh.personalizedpagerank import PersonalizedPageRank


def _two_cliques_bridged() -> Graph:
    g = Graph(directed=True)
    left = [f"l{i}" for i in range(4)]
    right = [f"r{i}" for i in range(4)]
    for n in left + right:
        g.add_node(n)
    for a, b in combinations(left, 2):
        g.add_edge(a, b)
        g.add_edge(b, a)
    for a, b in combinations(right, 2):
        g.add_edge(a, b)
        g.add_edge(b, a)
    g.add_edge("l0", "r0")
    g.add_edge("r0", "l0")
    return g


class TestDistribution:
    def test_ranks_sum_to_one(self):
        ppr = PersonalizedPageRank(_two_cliques_bridged(), ["l1"])
        assert ppr.total() == pytest.approx(1.0)

    def test_the_seed_scores_highest(self):
        ppr = PersonalizedPageRank(_two_cliques_bridged(), ["l1"])
        assert max(ppr.rank, key=ppr.rank.get) == "l1"

    def test_the_seeds_own_clique_outranks_the_far_one(self):
        ppr = PersonalizedPageRank(_two_cliques_bridged(), ["l1"])
        near = sum(ppr.rank[f"l{i}"] for i in range(4))
        far = sum(ppr.rank[f"r{i}"] for i in range(4))
        assert near > 3 * far

    def test_recommendations_exclude_the_seeds_and_favor_the_neighborhood(self):
        ppr = PersonalizedPageRank(_two_cliques_bridged(), ["l1"])
        recs = [n for n, _s in ppr.recommendations(3)]
        assert "l1" not in recs
        assert set(recs) <= {"l0", "l2", "l3"}

    def test_dangling_mass_returns_to_the_seeds(self):
        g = Graph(directed=True)
        for n in "abc":
            g.add_node(n)
        g.add_edge("a", "b")
        g.add_edge("b", "c")  # c is a dead end
        ppr = PersonalizedPageRank(g, ["a"])
        assert ppr.total() == pytest.approx(1.0)
        assert ppr.rank["a"] > ppr.rank["c"]


class TestAgainstPlainPageRank:
    def test_seeding_every_node_equally_recovers_plain_pagerank(self):
        g = _two_cliques_bridged()
        g.add_node("x")
        g.add_edge("l3", "x")
        ppr = PersonalizedPageRank(g, g.nodes(), tolerance=1e-12)
        plain = PageRank(g, tolerance=1e-12)
        for n in g.nodes():
            assert ppr.rank[n] == pytest.approx(plain.rank[n], abs=1e-8)


class TestRefusals:
    def test_an_empty_seed_set_is_refused(self):
        with pytest.raises(Invalid):
            PersonalizedPageRank(_two_cliques_bridged(), [])

    def test_a_seed_outside_the_graph_is_refused(self):
        with pytest.raises(Missing):
            PersonalizedPageRank(_two_cliques_bridged(), ["ghost"])

    def test_a_damping_outside_the_unit_interval_is_refused(self):
        with pytest.raises(Invalid):
            PersonalizedPageRank(_two_cliques_bridged(), ["l0"], damping=1.0)


class TestReport:
    def test_the_note_states_the_neighborhood_share(self):
        note = PersonalizedPageRank(_two_cliques_bridged(), ["l1"]).note()
        assert "of the mass within one hop" in note
        assert "top recommendation" in note
