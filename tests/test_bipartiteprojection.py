from __future__ import annotations

import pytest

from mesh.bipartiteprojection import BipartiteProjection
from mesh.errors import Invalid, Missing
from mesh.graph import Graph


def _authors_papers() -> Graph:
    # ana and bo wrote p1 and p2 together; bo, cy, dev all wrote p3
    g = Graph()
    for n in ["ana", "bo", "cy", "dev", "p1", "p2", "p3"]:
        g.add_node(n)
    for a, p in [("ana", "p1"), ("bo", "p1"), ("ana", "p2"), ("bo", "p2"),
                 ("bo", "p3"), ("cy", "p3"), ("dev", "p3")]:
        g.add_edge(a, p)
    return g


AUTHORS = ["ana", "bo", "cy", "dev"]


class TestProjection:
    def test_coauthors_are_joined_with_the_shared_paper_count(self):
        proj = BipartiteProjection(_authors_papers(), AUTHORS).projection
        assert proj.weight("ana", "bo") == 2
        assert proj.weight("bo", "cy") == 1
        assert not proj.has_edge("ana", "cy")

    def test_a_three_author_paper_projects_to_a_triangle(self):
        proj = BipartiteProjection(_authors_papers(), AUTHORS).projection
        for a, b in [("bo", "cy"), ("cy", "dev"), ("bo", "dev")]:
            assert proj.has_edge(a, b)

    def test_the_newman_weight_discounts_the_crowded_paper(self):
        proj = BipartiteProjection(_authors_papers(), AUTHORS, newman=True).projection
        assert proj.weight("ana", "bo") == pytest.approx(2.0)  # two papers of two authors
        assert proj.weight("bo", "cy") == pytest.approx(0.5)  # one paper of three

    def test_the_strongest_tie_is_the_most_shared_pair(self):
        u, v, w = BipartiteProjection(_authors_papers(), AUTHORS).strongest_tie()
        assert {u, v} == {"ana", "bo"}
        assert w == 2

    def test_projecting_the_other_side_gives_papers_linked_by_shared_authors(self):
        proj = BipartiteProjection(_authors_papers(), ["p1", "p2", "p3"]).projection
        assert proj.weight("p1", "p2") == 2  # ana and bo
        assert proj.weight("p1", "p3") == 1  # bo


class TestRefusals:
    def test_a_directed_graph_is_refused(self):
        with pytest.raises(Invalid):
            BipartiteProjection(Graph(directed=True), [])

    def test_a_non_bipartite_graph_is_refused(self):
        g = Graph()
        for n in "abc":
            g.add_node(n)
        g.add_edge("a", "b")
        g.add_edge("b", "c")
        g.add_edge("c", "a")
        with pytest.raises(Invalid):
            BipartiteProjection(g, ["a"])

    def test_a_kept_side_that_is_not_a_part_is_refused(self):
        with pytest.raises(Invalid):
            BipartiteProjection(_authors_papers(), ["ana", "p1"])

    def test_a_missing_node_is_refused(self):
        with pytest.raises(Missing):
            BipartiteProjection(_authors_papers(), ["ghost"])


class TestReport:
    def test_the_note_compares_ties_to_clique_edges(self):
        note = BipartiteProjection(_authors_papers(), AUTHORS).note()
        # p1 and p2 give 1 clique edge each, p3 gives 3: 5, projecting to 4 ties
        assert "projected 4 tie(s)" in note
        assert "from 5 clique edge(s)" in note
