from __future__ import annotations

from math import inf

import pytest

from mesh.errors import Invalid
from mesh.factories import complete, cycle, path, star
from mesh.graph import Graph
from mesh.vitality import Vitality


class TestNodes:
    def test_a_star_leaf_scores_two_n_minus_three_and_the_hub_disconnects(self):
        n = 6
        v = Vitality(star(n - 1))
        assert v.node("1") == 2 * n - 3
        assert v.node("0") == inf
        assert v.most_vital() == "0"

    def test_every_node_of_a_complete_graph_scores_n_minus_one(self):
        v = Vitality(complete(5))
        assert all(score == 4 for score in v.all_nodes().values())

    def test_a_path_end_scores_its_own_distance_sum(self):
        n = 6
        v = Vitality(path(n))
        assert v.node("0") == n * (n - 1) / 2
        assert v.node("0") == v.own_distance_sum("0")
        assert v.node("2") == inf

    def test_a_cycle_node_scores_less_than_its_own_distances(self):
        # the guess was more: removing a cycle node forces detours, and detours lengthen
        # the remaining distances, so the score is the node's own sum of 9 minus the
        # growth of 2, which is 7. the own sum is the ceiling, not the floor
        v = Vitality(cycle(6))
        assert v.own_distance_sum("0") == 9
        assert v.node("0") == 7
        assert v.node("0") < v.own_distance_sum("0")


class TestEdges:
    def test_a_bridge_scores_infinity_and_a_chord_can_score_zero(self):
        v = Vitality(path(4))
        assert v.edge("1", "2") == inf
        g = complete(4)
        assert Vitality(g).edge("0", "1") == 1.0

    def test_an_edge_of_a_cycle_adds_the_detours(self):
        # cutting one edge of a square turns two distances of 1 and 2 into 3 and ... the
        # pair across the cut goes from 1 to 3, the diagonals stay 2, so growth is 2
        v = Vitality(cycle(4))
        assert v.edge("0", "1") == 2.0

    def test_edge_scores_cover_every_edge(self):
        # the guess was 2 per edge as on the square; cutting a pentagon edge turns the
        # index from 15 to the path's 20, so each edge scores 5
        scores = Vitality(cycle(5)).all_edges()
        assert len(scores) == 5
        assert all(s == 5.0 for s in scores.values())


class TestDisconnected:
    def test_a_disconnected_graph_scores_infinity_everywhere(self):
        g = Graph()
        for n in "abcd":
            g.add_node(n)
        g.add_edge("a", "b")
        g.add_edge("c", "d")
        v = Vitality(g)
        assert v.node("a") == inf
        assert v.edge("a", "b") == inf


class TestRefusal:
    def test_absent_items_and_directed_graphs_are_refused(self):
        v = Vitality(path(3))
        with pytest.raises(Invalid):
            v.node("zz")
        with pytest.raises(Invalid):
            v.edge("0", "2")
        with pytest.raises(Invalid):
            Vitality(Graph(directed=True))


class TestReport:
    def test_the_note_names_the_most_vital_node(self):
        note = Vitality(star(3)).note()
        assert "most vital node 0: removing it disconnects the graph" in note
        assert "adds 4 to the distance sum" in Vitality(complete(5)).note()
        assert Vitality(Graph()).note() == "no nodes, nothing vital"
