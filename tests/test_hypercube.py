from __future__ import annotations

import pytest

from mesh.errors import Invalid
from mesh.graph import Graph
from mesh.graphproduct import GraphProduct
from mesh.hypercube import Hypercube
from mesh.isomorphism import Isomorphism


def _edge() -> Graph:
    g = Graph()
    g.add_node("0")
    g.add_node("1")
    g.add_edge("0", "1")
    return g


class TestCounts:
    def test_nodes_edges_and_degrees_follow_the_dimension(self):
        for d in range(5):
            h = Hypercube(d)
            assert h.graph.node_count() == 2**d
            assert h.graph.edge_count() == d * 2 ** (d - 1) if d else h.graph.edge_count() == 0
            assert h.is_regular()

    def test_a_zero_dimensional_cube_is_one_node(self):
        h = Hypercube(0)
        assert h.graph.nodes() == [""]
        assert h.graph.edge_count() == 0


class TestStructure:
    def test_bit_parity_is_a_bipartition(self):
        assert Hypercube(4).parity_is_a_bipartition()

    def test_hops_equal_hamming_distance(self):
        for d in (1, 2, 3, 4):
            assert Hypercube(d).hops_equal_hamming()

    def test_the_three_cube_is_edge_times_edge_times_edge(self):
        square = GraphProduct(_edge(), _edge()).cartesian()
        cube = GraphProduct(square, _edge()).cartesian()
        assert Isomorphism(Hypercube(3).graph, cube).isomorphic


class TestGrayCode:
    def test_reflection_builds_the_standard_two_bit_code(self):
        assert Hypercube(2).gray_code() == ["00", "01", "11", "10"]

    def test_the_gray_code_is_a_hamiltonian_cycle_from_two_dimensions_up(self):
        for d in (2, 3, 4, 5):
            assert Hypercube(d).gray_code_is_a_hamiltonian_cycle()

    def test_one_and_zero_dimensions_have_no_cycle(self):
        assert not Hypercube(1).gray_code_is_a_hamiltonian_cycle()
        assert not Hypercube(0).gray_code_is_a_hamiltonian_cycle()


class TestRefusal:
    def test_a_negative_dimension_is_refused(self):
        with pytest.raises(Invalid):
            Hypercube(-1)


class TestReport:
    def test_the_note_states_the_counts_and_the_cycle(self):
        note = Hypercube(3).note()
        assert "Q3: 8 node(s), 12 edge(s), 3-regular, Gray cycle closes" in note
        assert "Gray cycle absent" in Hypercube(1).note()
