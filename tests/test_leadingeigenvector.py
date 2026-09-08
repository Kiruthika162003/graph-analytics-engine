from __future__ import annotations

from itertools import combinations

import pytest

from mesh.errors import Invalid
from mesh.factories import complete
from mesh.graph import Graph
from mesh.leadingeigenvector import LeadingEigenvector
from mesh.louvain import Louvain


def _two_cliques_bridged() -> Graph:
    g = Graph()
    left = [f"l{i}" for i in range(5)]
    right = [f"r{i}" for i in range(5)]
    for n in left + right:
        g.add_node(n)
    for a, b in combinations(left, 2):
        g.add_edge(a, b)
    for a, b in combinations(right, 2):
        g.add_edge(a, b)
    g.add_edge("l0", "r0")
    return g


class TestSplit:
    def test_two_bridged_cliques_split_along_the_bridge(self):
        le = LeadingEigenvector(_two_cliques_bridged())
        pos, neg = le.sides()
        left = {f"l{i}" for i in range(5)}
        assert left in (pos, neg)
        assert le.worth_splitting()

    def test_the_split_modularity_is_high_on_clear_structure(self):
        le = LeadingEigenvector(_two_cliques_bridged())
        assert le.modularity() > 0.4

    def test_the_vector_is_an_eigenvector_of_the_modularity_matrix(self):
        g = _two_cliques_bridged()
        le = LeadingEigenvector(g, tolerance=1e-13)
        bx = le._modularity_times(le.vector)
        for n in g.nodes():
            assert bx[n] == pytest.approx(le.eigenvalue * le.vector[n], abs=1e-6)

    def test_a_complete_graph_has_no_seam(self):
        le = LeadingEigenvector(complete(6))
        assert not le.worth_splitting()
        assert "one piece" in le.note()

    def test_the_spectral_split_is_measured_against_louvain(self):
        g = _two_cliques_bridged()
        le = LeadingEigenvector(g)
        lv = Louvain(g, seed=1)
        # a relaxation rounded to signs can trail a greedy climb, never lead by much
        assert le.modularity() <= lv.modularity() + 1e-9


class TestRefusals:
    def test_a_directed_graph_is_refused(self):
        with pytest.raises(Invalid):
            LeadingEigenvector(Graph(directed=True))

    def test_an_edgeless_graph_is_refused(self):
        g = Graph()
        g.add_node("a")
        with pytest.raises(Invalid):
            LeadingEigenvector(g)


class TestReport:
    def test_the_note_states_the_eigenvalue_and_modularity(self):
        note = LeadingEigenvector(_two_cliques_bridged()).note()
        assert "leading eigenvalue" in note
        assert "split modularity" in note
