from __future__ import annotations

import pytest

from mesh.errors import Invalid, Missing
from mesh.graph import Graph
from mesh.simrank import SimRank


def _citations() -> Graph:
    # papers p1, p2 both cite a and b; p3 cites only c
    g = Graph(directed=True)
    for n in ["p1", "p2", "p3", "a", "b", "c"]:
        g.add_node(n)
    for u, v in [("p1", "a"), ("p1", "b"), ("p2", "a"), ("p2", "b"), ("p3", "c")]:
        g.add_edge(u, v)
    return g


class TestSimilarity:
    def test_a_node_is_fully_similar_to_itself(self):
        assert SimRank(_citations()).similarity("a", "a") == 1.0

    def test_two_nodes_cited_by_the_same_papers_are_similar(self):
        # a and b share citers p1 and p2, which are themselves uncited, so
        # sim(a,b) = 0.8 * (sim(p1,p1) + sim(p1,p2) + sim(p2,p1) + sim(p2,p2)) / 4
        # = 0.8 * (1 + 0 + 0 + 1) / 4 = 0.4. A first guess of "above 0.5"
        # was refuted by working the definition through.
        s = SimRank(_citations())
        assert s.similarity("a", "b") == pytest.approx(0.4, abs=1e-6)

    def test_the_equation_holds_at_the_fixed_point(self):
        g = _citations()
        s = SimRank(g, tolerance=1e-12)
        ins = {"a": ["p1", "p2"], "b": ["p1", "p2"]}
        expected = 0.8 * sum(s.similarity(x, y) for x in ins["a"] for y in ins["b"]) / 4
        assert s.similarity("a", "b") == pytest.approx(expected, abs=1e-9)

    def test_a_node_nothing_points_at_has_zero_similarity_to_others(self):
        s = SimRank(_citations())
        assert s.similarity("p1", "p2") == 0.0  # no paper cites a paper here
        assert s.similarity("a", "c") == 0.0  # c's citer shares nothing with a's

    def test_similarity_is_symmetric(self):
        s = SimRank(_citations())
        assert s.similarity("a", "b") == pytest.approx(s.similarity("b", "a"))

    def test_the_most_similar_partner_is_the_co_cited_node(self):
        s = SimRank(_citations())
        assert s.most_similar("a", 1)[0][0] == "b"


class TestDecay:
    def test_a_smaller_decay_lowers_every_similarity(self):
        g = _citations()
        strong = SimRank(g, decay=0.9).similarity("a", "b")
        weak = SimRank(g, decay=0.3).similarity("a", "b")
        assert weak < strong


class TestRefusals:
    def test_a_decay_outside_the_unit_interval_is_refused(self):
        with pytest.raises(Invalid):
            SimRank(_citations(), decay=1.0)

    def test_an_empty_graph_is_refused(self):
        with pytest.raises(Invalid):
            SimRank(Graph(directed=True))

    def test_a_missing_node_is_refused(self):
        with pytest.raises(Missing):
            SimRank(_citations()).similarity("a", "ghost")


class TestReport:
    def test_the_note_states_the_largest_similarity(self):
        note = SimRank(_citations()).note()
        assert "largest similarity between distinct nodes" in note
