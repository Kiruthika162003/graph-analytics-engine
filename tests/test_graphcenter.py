from __future__ import annotations

import random
from math import inf

import pytest

from mesh.errors import Invalid
from mesh.factories import complete, cycle, path, star
from mesh.graph import Graph
from mesh.graphcenter import GraphCenter
from mesh.prufer import Prufer


class TestNumbers:
    def test_a_path_of_five_centers_on_the_middle(self):
        gc = GraphCenter(path(5))
        assert gc.radius() == 2
        assert gc.diameter() == 4
        assert gc.center() == ["2"]
        assert gc.periphery() == ["0", "4"]

    def test_a_path_of_four_has_two_center_nodes(self):
        assert GraphCenter(path(4)).center() == ["1", "2"]

    def test_a_star_centers_on_the_hub_with_every_leaf_peripheral(self):
        gc = GraphCenter(star(4))
        assert gc.center() == ["0"]
        assert gc.periphery() == ["1", "2", "3", "4"]
        assert gc.radius() == 1
        assert gc.diameter() == 2

    def test_a_cycle_and_a_clique_are_all_center(self):
        assert GraphCenter(cycle(6)).center() == cycle(6).nodes()
        assert GraphCenter(complete(4)).radius() == 1

    def test_weights_change_the_answer(self):
        g = path(3)
        g.add_edge("0", "1", 10.0)
        g.add_edge("1", "2", 1.0)
        assert GraphCenter(g).center() == ["1"]
        # the guess was a center of 1 and 2; node 2 is 11 from node 0, so only the
        # middle node reaches the radius of 10, and both ends sit at 11
        weighted = GraphCenter(g, weighted=True)
        assert weighted.radius() == 10.0
        assert weighted.center() == ["1"]
        assert weighted.periphery() == ["0", "2"]
        assert weighted.diameter() == 11.0


class TestBounds:
    def test_the_diameter_never_exceeds_twice_the_radius(self):
        rng = random.Random(607)
        for _ in range(20):
            code = [rng.randrange(9) for _ in range(7)]
            tree = Prufer.decode(code)
            assert GraphCenter(tree).doubling_bound_holds()

    def test_peeling_a_tree_lands_on_the_eccentricity_center(self):
        rng = random.Random(613)
        for _ in range(20):
            code = [rng.randrange(10) for _ in range(8)]
            tree = Prufer.decode(code)
            gc = GraphCenter(tree)
            assert gc.peel_tree() == gc.center()

    def test_peeling_refuses_a_graph_with_a_cycle(self):
        with pytest.raises(Invalid):
            GraphCenter(cycle(4)).peel_tree()


class TestDisconnected:
    def test_a_disconnected_graph_has_infinite_eccentricity_and_no_center(self):
        g = Graph()
        for n in "abcd":
            g.add_node(n)
        g.add_edge("a", "b")
        g.add_edge("c", "d")
        gc = GraphCenter(g)
        assert gc.radius() == inf
        assert gc.doubling_bound_holds()
        assert "no center" in gc.note()


class TestRefusal:
    def test_a_directed_graph_is_refused(self):
        with pytest.raises(Invalid):
            GraphCenter(Graph(directed=True))


class TestReport:
    def test_the_note_states_radius_diameter_and_both_sets(self):
        note = GraphCenter(path(5)).note()
        assert "radius 2 and diameter 4 in hop(s)" in note
        assert "center ['2'], periphery ['0', '4']" in note
