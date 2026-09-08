from __future__ import annotations

import math

import pytest

from mesh.bipartite import Bipartite
from mesh.diameter import Diameter
from mesh.errors import Invalid
from mesh.factories import (
    complete,
    complete_bipartite,
    cycle,
    grid,
    note,
    path,
    petersen,
    star,
    wheel,
)
from mesh.girth import Girth
from mesh.hamiltonian import Hamiltonian
from mesh.planarity import PlanarityCheck
from mesh.triangles import Triangles


class TestCounts:
    def test_complete_has_n_choose_two_edges(self):
        assert complete(6).edge_count() == 15

    def test_cycle_and_path_edge_counts(self):
        assert cycle(7).edge_count() == 7
        assert path(7).edge_count() == 6

    def test_star_and_wheel_edge_counts(self):
        assert star(5).edge_count() == 5
        assert wheel(5).edge_count() == 10

    def test_grid_edge_count_follows_the_formula(self):
        # 2 * side * (side - 1) edges
        assert grid(4).edge_count() == 24

    def test_complete_bipartite_edge_count_is_the_product(self):
        assert complete_bipartite(3, 4).edge_count() == 12

    def test_petersen_is_three_regular_on_ten_nodes(self):
        g = petersen()
        assert g.node_count() == 10
        assert g.edge_count() == 15
        assert all(g.degree(n) == 3 for n in g.nodes())


class TestProperties:
    def test_the_grid_is_bipartite_with_a_long_diameter(self):
        g = grid(4)
        assert Bipartite(g).is_bipartite
        assert Diameter(g).diameter() == 6

    def test_the_wheel_is_full_of_triangles(self):
        assert Triangles(wheel(6)).total == 6

    def test_complete_bipartite_is_triangle_free(self):
        assert Triangles(complete_bipartite(3, 3)).total == 0

    def test_petersen_has_girth_five_and_no_hamiltonian_cycle(self):
        g = petersen()
        assert Girth(g).girth == 5
        assert not Hamiltonian(g, cycle=True).exists

    def test_petersen_is_the_case_the_planarity_obstructions_miss(self):
        # first guess: the search would catch it. It cannot: Petersen holds
        # K3,3 only as a subdivision, never as a subgraph, and its 15 edges sit
        # under both Euler bounds, so the honest verdict is no obstruction
        # found, the exact limitation the planarity module documents
        assert PlanarityCheck(petersen()).verdict == "no obstruction found"

    def test_a_prefix_namespaces_the_nodes(self):
        g = cycle(3, prefix="x")
        assert sorted(g.nodes()) == ["x0", "x1", "x2"]


class TestRefusals:
    def test_shapes_refuse_sizes_too_small_to_exist(self):
        with pytest.raises(Invalid):
            cycle(2)
        with pytest.raises(Invalid):
            wheel(2)
        with pytest.raises(Invalid):
            grid(0)
        with pytest.raises(Invalid):
            star(0)
        with pytest.raises(Invalid):
            complete_bipartite(0, 3)


class TestReport:
    def test_the_note_states_counts(self):
        assert note(complete(4), "K4") == "K4: 4 node(s), 6 edge(s)"

    def test_cayley_style_sanity_on_the_star(self):
        # a star on n nodes is a tree: edges are nodes minus one
        g = star(9)
        assert g.edge_count() == g.node_count() - 1
        assert math.isclose(g.edge_count() / g.node_count(), 0.9)
