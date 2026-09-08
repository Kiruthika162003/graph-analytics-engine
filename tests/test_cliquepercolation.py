from __future__ import annotations

from itertools import combinations

import pytest

from mesh.cliquepercolation import CliquePercolation
from mesh.errors import Invalid, Missing
from mesh.graph import Graph


def _two_cliques_sharing_a_node() -> Graph:
    # K4 on a,b,c,d and K4 on d,e,f,g share d: two 3-clique communities
    # that overlap at d, since their cliques share only one node
    g = Graph()
    for n in "abcdefg":
        g.add_node(n)
    for x, y in combinations("abcd", 2):
        g.add_edge(x, y)
    for x, y in combinations("defg", 2):
        g.add_edge(x, y)
    g.add_node("hermit")
    return g


class TestCommunities:
    def test_two_cliques_sharing_one_node_stay_separate_at_k3(self):
        cp = CliquePercolation(_two_cliques_sharing_a_node(), k=3)
        assert len(cp.communities) == 2
        assert {"a", "b", "c", "d"} in cp.communities

    def test_the_shared_node_belongs_to_both(self):
        cp = CliquePercolation(_two_cliques_sharing_a_node(), k=3)
        assert len(cp.communities_of("d")) == 2
        assert cp.overlapping_nodes() == {"d"}

    def test_a_node_in_no_clique_is_left_out(self):
        cp = CliquePercolation(_two_cliques_sharing_a_node(), k=3)
        assert cp.left_out() == {"hermit"}

    def test_a_shared_edge_merges_two_triangles_at_k3(self):
        # triangles a,b,c and b,c,d share the edge b-c: two nodes, so they chain
        g = Graph()
        for n in "abcd":
            g.add_node(n)
        for x, y in [("a", "b"), ("b", "c"), ("c", "a"), ("b", "d"), ("c", "d")]:
            g.add_edge(x, y)
        cp = CliquePercolation(g, k=3)
        assert cp.communities == [{"a", "b", "c", "d"}]

    def test_k_controls_how_dense_a_community_must_be(self):
        g = _two_cliques_sharing_a_node()
        assert len(CliquePercolation(g, k=4).communities) == 2
        assert CliquePercolation(g, k=5).communities == []

    def test_every_k_clique_is_fully_connected(self):
        g = _two_cliques_sharing_a_node()
        for clique in CliquePercolation(g, k=3).cliques:
            for x, y in combinations(sorted(clique), 2):
                assert g.has_edge(x, y)


class TestRefusals:
    def test_a_directed_graph_is_refused(self):
        with pytest.raises(Invalid):
            CliquePercolation(Graph(directed=True))

    def test_k_below_two_is_refused(self):
        with pytest.raises(Invalid):
            CliquePercolation(Graph(), k=1)

    def test_a_missing_node_is_refused(self):
        with pytest.raises(Missing):
            CliquePercolation(_two_cliques_sharing_a_node()).communities_of("ghost")


class TestReport:
    def test_the_note_counts_overlap_and_left_out(self):
        note = CliquePercolation(_two_cliques_sharing_a_node(), k=3).note()
        assert "1 node(s) in more than one, 1 in none" in note
