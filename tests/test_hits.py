from __future__ import annotations

import math

import pytest

from mesh.errors import Invalid
from mesh.graph import Graph
from mesh.hits import HITS


def _directory_and_answers() -> Graph:
    # two directories d1, d2 each point at answers a1, a2, a3; answers point nowhere
    g = Graph(directed=True)
    for n in ["d1", "d2", "a1", "a2", "a3"]:
        g.add_node(n)
    for d in ("d1", "d2"):
        for a in ("a1", "a2", "a3"):
            g.add_edge(d, a)
    return g


class TestScores:
    def test_directories_are_hubs_and_answers_are_authorities(self):
        h = HITS(_directory_and_answers())
        assert h.top_hubs(1)[0][0] in {"d1", "d2"}
        assert h.top_authorities(1)[0][0] in {"a1", "a2", "a3"}

    def test_a_pure_hub_has_zero_authority(self):
        h = HITS(_directory_and_answers())
        assert h.authority["d1"] == 0.0
        assert h.hub["a1"] == 0.0

    def test_both_vectors_have_unit_length(self):
        h = HITS(_directory_and_answers())
        assert math.sqrt(sum(x * x for x in h.hub.values())) == pytest.approx(1.0)
        assert math.sqrt(sum(x * x for x in h.authority.values())) == pytest.approx(1.0)

    def test_an_answer_cited_by_more_hubs_ranks_higher(self):
        g = _directory_and_answers()
        g.add_node("d3")
        g.add_edge("d3", "a1")  # a1 now has three citers, the others two
        h = HITS(g)
        assert h.top_authorities(1)[0][0] == "a1"

    def test_the_result_is_a_fixed_point_of_the_update(self):
        g = _directory_and_answers()
        g.add_node("d3")
        g.add_edge("d3", "a1")
        h = HITS(g, tolerance=1e-13)
        # authority is proportional to the sum of incoming hubs
        raw_auth = {n: sum(h.hub[u] for u in g.nodes() if g.has_edge(u, n)) for n in g.nodes()}
        norm = math.sqrt(sum(x * x for x in raw_auth.values()))
        for n in g.nodes():
            assert raw_auth[n] / norm == pytest.approx(h.authority[n], abs=1e-8)


class TestStructure:
    def test_the_one_sided_count_covers_every_node_in_a_bipartite_layout(self):
        h = HITS(_directory_and_answers())
        assert h.one_sided_count() == 5

    def test_a_reciprocal_pair_is_strong_on_both_axes(self):
        g = Graph(directed=True)
        g.add_node("a")
        g.add_node("b")
        g.add_edge("a", "b")
        g.add_edge("b", "a")
        h = HITS(g)
        assert h.one_sided_count() == 0


class TestRefusals:
    def test_an_undirected_graph_is_refused(self):
        with pytest.raises(Invalid):
            HITS(Graph())

    def test_an_edgeless_graph_is_refused(self):
        g = Graph(directed=True)
        g.add_node("a")
        with pytest.raises(Invalid):
            HITS(g)


class TestReport:
    def test_the_note_names_a_hub_and_an_authority(self):
        note = HITS(_directory_and_answers()).note()
        assert "top hub" in note
        assert "top authority" in note
