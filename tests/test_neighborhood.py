from __future__ import annotations

import pytest

from mesh.errors import Invalid
from mesh.factories import complete, cycle, path, star
from mesh.graph import Graph
from mesh.neighborhood import Neighborhood


class TestEgo:
    def test_radius_one_takes_the_neighbors_and_their_ties(self):
        g = complete(4)
        g.add_node("far")
        g.add_edge("3", "far")
        ego = Neighborhood(g).ego("0", 1)
        assert sorted(ego.nodes()) == ["0", "1", "2", "3"]
        assert ego.edge_count() == 6
        assert Neighborhood.density(ego) == 1.0

    def test_dropping_the_center_leaves_the_circle_and_radius_two_reaches_further(self):
        nb = Neighborhood(path(5))
        circle = nb.ego("2", 1, keep_center=False)
        assert sorted(circle.nodes()) == ["1", "3"]
        assert circle.edge_count() == 0
        assert sorted(nb.ego("2", 2).nodes()) == ["0", "1", "2", "3", "4"]
        assert nb.ego("2", 0).node_count() == 1

    def test_unknown_nodes_negative_radii_and_directed_graphs_are_refused(self):
        nb = Neighborhood(path(3))
        with pytest.raises(Invalid):
            nb.ego("zz")
        with pytest.raises(Invalid):
            nb.ego("0", -1)
        with pytest.raises(Invalid):
            Neighborhood(Graph(directed=True))


class TestConstraint:
    def test_a_star_hub_is_constrained_one_over_its_leaves(self):
        nb = Neighborhood(star(5))
        assert nb.constraint("0") == pytest.approx(1 / 5)
        assert nb.effective_size("0") == 5.0

    def test_a_leaf_with_one_tie_is_fully_constrained(self):
        nb = Neighborhood(star(3))
        assert nb.constraint("1") == pytest.approx(1.0)
        assert nb.effective_size("1") == 1.0

    def test_a_complete_graph_constrains_more_as_it_shrinks(self):
        small = Neighborhood(complete(4)).constraint("0")
        large = Neighborhood(complete(8)).constraint("0")
        assert small > large
        assert Neighborhood(complete(4)).effective_size("0") == pytest.approx(1.0)

    def test_a_cycle_node_brokers_its_two_strangers(self):
        nb = Neighborhood(cycle(6))
        assert nb.constraint("0") == pytest.approx(0.5)
        assert nb.effective_size("0") == 2.0

    def test_an_isolated_node_is_refused(self):
        g = path(2)
        g.add_node("alone")
        with pytest.raises(Invalid, match="no ties"):
            Neighborhood(g).constraint("alone")
        assert Neighborhood(g).effective_size("alone") == 0.0


class TestReport:
    def test_the_note_reads_the_circle_and_the_brokerage(self):
        note = Neighborhood(star(4)).note("0")
        assert note.startswith("0: 4 contact(s) with density 0.00, constraint 0.250")
        assert note.endswith("effective size 4.00")
