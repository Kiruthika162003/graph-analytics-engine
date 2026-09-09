from __future__ import annotations

from itertools import combinations

import pytest

from mesh.errors import Invalid
from mesh.factories import complete, cycle, path
from mesh.graph import Graph
from mesh.spectralclustering import SpectralClustering


def _cliques(groups: list[str], bridges: list[tuple[str, str]]) -> Graph:
    g = Graph()
    for group in groups:
        for n in group:
            g.add_node(n)
        for a, b in combinations(group, 2):
            g.add_edge(a, b)
    for a, b in bridges:
        g.add_edge(a, b)
    return g


class TestRecovery:
    def test_three_cliques_on_single_bridges_come_back_as_three_groups(self):
        g = _cliques(["abcd", "efgh", "ijkl"], [("d", "e"), ("h", "i")])
        sc = SpectralClustering(g, 3)
        assert sc.groups == [list("abcd"), list("efgh"), list("ijkl")]
        assert sc.crossing_edges() == 2

    def test_components_come_back_exactly_with_nothing_crossing(self):
        g = _cliques(["abc", "xyz"], [])
        sc = SpectralClustering(g, 2)
        assert sc.groups == [list("abc"), list("xyz")]
        assert sc.crossing_edges() == 0

    def test_two_cliques_of_unequal_size_still_separate(self):
        g = _cliques(["abcde", "xyz"], [("e", "x")])
        sc = SpectralClustering(g, 2)
        assert sc.groups == [list("abcde"), list("xyz")]

    def test_one_group_is_everything(self):
        sc = SpectralClustering(cycle(5), 1)
        assert sc.groups == [cycle(5).nodes()]
        assert sc.crossing_edges() == 0


class TestShapes:
    def test_a_path_split_in_two_cuts_one_edge(self):
        sc = SpectralClustering(path(8), 2)
        assert sc.crossing_edges() == 1
        assert sorted(len(g) for g in sc.groups) == [4, 4]

    def test_a_complete_graph_gives_k_groups_covering_everything(self):
        sc = SpectralClustering(complete(6), 3)
        assert sum(len(g) for g in sc.groups) == 6
        assert len(sc.groups) <= 3


class TestRefusal:
    def test_bad_k_and_directed_graphs_are_refused(self):
        with pytest.raises(Invalid):
            SpectralClustering(cycle(4), 0)
        with pytest.raises(Invalid):
            SpectralClustering(cycle(4), 5)
        with pytest.raises(Invalid):
            SpectralClustering(Graph(directed=True), 1)


class TestReport:
    def test_the_note_gives_sizes_rounds_and_crossings(self):
        g = _cliques(["abcd", "efgh", "ijkl"], [("d", "e"), ("h", "i")])
        note = SpectralClustering(g, 3).note()
        assert "3 group(s) of sizes [4, 4, 4]" in note
        assert "2 edge(s) cross between groups" in note
