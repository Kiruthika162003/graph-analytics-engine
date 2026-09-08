from __future__ import annotations

from itertools import combinations

import pytest

from mesh.errors import Invalid, Missing
from mesh.graph import Graph
from mesh.labelpropagation import LabelPropagation
from mesh.louvain import Louvain


def _three_cliques() -> Graph:
    g = Graph()
    groups = [[f"{c}{i}" for i in range(4)] for c in "abc"]
    for group in groups:
        for n in group:
            g.add_node(n)
        for x, y in combinations(group, 2):
            g.add_edge(x, y)
    g.add_edge("a0", "b0")
    g.add_edge("b1", "c0")
    return g


class TestCommunities:
    def test_three_bridged_cliques_are_recovered(self):
        lv = Louvain(_three_cliques(), seed=1)
        comms = lv.communities()
        assert len(comms) == 3
        assert {f"a{i}" for i in range(4)} in comms

    def test_members_of_a_clique_share_a_community(self):
        lv = Louvain(_three_cliques(), seed=2)
        assert lv.community_of("c2") == {f"c{i}" for i in range(4)}

    def test_a_single_clique_is_one_community(self):
        g = Graph()
        for n in "abcd":
            g.add_node(n)
        for x, y in combinations("abcd", 2):
            g.add_edge(x, y)
        assert len(Louvain(g).communities()) == 1

    def test_the_same_seed_reproduces_the_partition(self):
        a = Louvain(_three_cliques(), seed=5).communities()
        b = Louvain(_three_cliques(), seed=5).communities()
        assert a == b


class TestModularity:
    def test_the_found_partition_beats_the_trivial_one(self):
        lv = Louvain(_three_cliques(), seed=3)
        assert lv.modularity() > lv.trivial_modularity()

    def test_modularity_is_high_on_clear_structure(self):
        assert Louvain(_three_cliques(), seed=4).modularity() > 0.5

    def test_it_matches_or_beats_label_propagation(self):
        g = _three_cliques()
        lv = Louvain(g, seed=6)
        lp = LabelPropagation(g, seed=6)
        assert lv.modularity() >= lp.modularity() - 1e-9

    def test_modularity_of_the_true_partition_is_computed_directly(self):
        g = _three_cliques()
        lv = Louvain(g, seed=7)
        truth = {n: "abc".index(n[0]) for n in g.nodes()}
        # 3 cliques of 6 internal edges each, 2 bridges, 20 edges total
        assert lv.modularity(truth) == pytest.approx(lv.modularity(), abs=1e-9)


class TestRefusals:
    def test_a_directed_graph_is_refused(self):
        with pytest.raises(Invalid):
            Louvain(Graph(directed=True))

    def test_an_edgeless_graph_is_refused(self):
        g = Graph()
        g.add_node("a")
        with pytest.raises(Invalid):
            Louvain(g)

    def test_a_missing_node_is_refused(self):
        with pytest.raises(Missing):
            Louvain(_three_cliques()).community_of("ghost")


class TestReport:
    def test_the_note_states_communities_and_levels(self):
        note = Louvain(_three_cliques(), seed=8).note()
        assert "3 communit(ies)" in note
        assert "local optimum" in note
