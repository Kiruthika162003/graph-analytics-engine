from __future__ import annotations

from itertools import combinations

import pytest

from mesh.errors import Invalid, Missing
from mesh.graph import Graph
from mesh.labelpropagation import LabelPropagation


def _two_cliques(bridge: bool = True) -> Graph:
    # two 5-cliques joined by a single edge
    g = Graph()
    left = [f"l{i}" for i in range(5)]
    right = [f"r{i}" for i in range(5)]
    for n in left + right:
        g.add_node(n)
    for a, b in combinations(left, 2):
        g.add_edge(a, b)
    for a, b in combinations(right, 2):
        g.add_edge(a, b)
    if bridge:
        g.add_edge("l0", "r0")
    return g


class TestCommunities:
    def test_two_bridged_cliques_split_into_two_communities(self):
        lp = LabelPropagation(_two_cliques(), seed=1)
        comms = lp.communities()
        assert len(comms) == 2
        assert {f"l{i}" for i in range(5)} in comms

    def test_members_of_a_clique_share_a_community(self):
        lp = LabelPropagation(_two_cliques(), seed=2)
        assert lp.community_of("l1") == {f"l{i}" for i in range(5)}

    def test_a_single_clique_becomes_one_community(self):
        g = Graph()
        for n in "abcd":
            g.add_node(n)
        for a, b in combinations("abcd", 2):
            g.add_edge(a, b)
        assert len(LabelPropagation(g, seed=3).communities()) == 1

    def test_an_isolated_node_keeps_its_own_label(self):
        g = _two_cliques()
        g.add_node("hermit")
        lp = LabelPropagation(g, seed=4)
        assert lp.community_of("hermit") == {"hermit"}


class TestDeterminism:
    def test_the_same_seed_reproduces_the_partition(self):
        a = LabelPropagation(_two_cliques(), seed=11).communities()
        b = LabelPropagation(_two_cliques(), seed=11).communities()
        assert a == b

    def test_it_converges_in_few_sweeps_on_clear_structure(self):
        lp = LabelPropagation(_two_cliques(), seed=5)
        assert lp.sweeps < 20


class TestModularity:
    def test_a_clear_split_scores_well_above_zero(self):
        lp = LabelPropagation(_two_cliques(), seed=6)
        assert lp.modularity() > 0.3

    def test_an_edgeless_graph_scores_zero(self):
        g = Graph()
        g.add_node("a")
        assert LabelPropagation(g).modularity() == 0.0


class TestRefusals:
    def test_a_directed_graph_is_refused(self):
        with pytest.raises(Invalid):
            LabelPropagation(Graph(directed=True))

    def test_a_missing_node_is_refused(self):
        with pytest.raises(Missing):
            LabelPropagation(_two_cliques()).community_of("ghost")


class TestReport:
    def test_the_note_states_communities_and_modularity(self):
        note = LabelPropagation(_two_cliques(), seed=7).note()
        assert "2 communit(ies)" in note
        assert "modularity" in note
