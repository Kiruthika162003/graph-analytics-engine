from __future__ import annotations

import random
from math import comb, inf

import pytest

from mesh.errors import Invalid
from mesh.factories import complete, cycle, path, star
from mesh.graph import Graph
from mesh.prufer import Prufer
from mesh.wienerindex import WienerIndex


class TestClosedForms:
    def test_a_path_gives_n_minus_one_n_n_plus_one_over_six(self):
        for n in range(2, 9):
            assert WienerIndex(path(n)).wiener() == (n - 1) * n * (n + 1) / 6

    def test_a_complete_graph_gives_n_choose_two(self):
        assert WienerIndex(complete(6)).wiener() == comb(6, 2)

    def test_a_star_gives_leaves_squared(self):
        # n nodes means n - 1 leaves; the form is (n - 1)^2
        assert WienerIndex(star(5)).wiener() == 25

    def test_a_cycle_of_six(self):
        # each node sees distances 1,1,2,2,3 summing to 9; 6 * 9 / 2 = 27
        assert WienerIndex(cycle(6)).wiener() == 27


class TestRelatives:
    def test_average_distance_is_the_index_over_the_pairs(self):
        wi = WienerIndex(path(4))
        assert wi.average_distance() == pytest.approx(10 / 6)

    def test_the_harary_index_weights_close_pairs_more(self):
        assert WienerIndex(complete(4)).harary() == pytest.approx(6.0)
        assert WienerIndex(path(3)).harary() == pytest.approx(2.5)

    def test_weights_enter_when_asked(self):
        g = path(3)
        g.add_edge("0", "1", 5.0)
        assert WienerIndex(g).wiener() == 4
        assert WienerIndex(g, weighted=True).wiener() == 5 + 1 + 6


class TestTrees:
    def test_the_edge_cut_identity_holds_on_random_trees(self):
        rng = random.Random(673)
        for _ in range(20):
            tree = Prufer.decode([rng.randrange(9) for _ in range(7)])
            assert WienerIndex(tree).tree_edge_identity_holds()

    def test_the_identity_is_refused_off_a_tree(self):
        with pytest.raises(Invalid):
            WienerIndex(cycle(4)).tree_edge_identity_holds()


class TestEdges:
    def test_a_disconnected_graph_is_infinite_for_wiener_but_finite_for_harary(self):
        g = Graph()
        for n in "abcd":
            g.add_node(n)
        g.add_edge("a", "b")
        g.add_edge("c", "d")
        wi = WienerIndex(g)
        assert wi.wiener() == inf
        assert wi.harary() == 2.0
        assert "disconnected" in wi.note()

    def test_tiny_graphs_read_zero(self):
        assert WienerIndex(Graph()).wiener() == 0.0
        one = Graph()
        one.add_node("x")
        assert WienerIndex(one).average_distance() == 0.0

    def test_a_directed_graph_is_refused(self):
        with pytest.raises(Invalid):
            WienerIndex(Graph(directed=True))


class TestReport:
    def test_the_note_states_the_three_readings(self):
        # the guess was a Harary index of 3.833; three pairs at 1, two at 2, one at 3
        # give 3 + 1 + 1/3 = 4.333
        note = WienerIndex(path(4)).note()
        assert "Wiener index 10, average distance 1.667, Harary index 4.333" in note
