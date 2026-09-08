from __future__ import annotations

import random

import pytest

from mesh.errors import Invalid
from mesh.graph import Graph
from mesh.kirchhoff import SpanningTreeCount
from mesh.prufer import Prufer


def _star(leaves: int) -> Graph:
    g = Graph()
    g.add_node("0")
    for i in range(1, leaves + 1):
        g.add_node(str(i))
        g.add_edge("0", str(i))
    return g


def _edge_set(g: Graph) -> set[frozenset[str]]:
    return {frozenset((u, v)) for u, v, _w in g.edges()}


class TestEncodeDecode:
    def test_a_star_encodes_to_its_hub_repeated(self):
        assert Prufer.encode(_star(4)) == [0, 0, 0]

    def test_a_path_encodes_to_its_interior(self):
        g = Graph()
        for n in "0123":
            g.add_node(n)
        g.add_edge("0", "1")
        g.add_edge("1", "2")
        g.add_edge("2", "3")
        assert Prufer.encode(g) == [1, 2]

    def test_decoding_a_code_recovers_the_tree(self):
        tree = Prufer.decode([1, 2])
        assert _edge_set(tree) == {
            frozenset(("0", "1")), frozenset(("1", "2")), frozenset(("2", "3"))
        }

    def test_the_round_trip_is_exact_on_random_trees(self):
        rng = random.Random(383)
        for _ in range(30):
            n = rng.randint(2, 20)
            code = [rng.randrange(n) for _ in range(n - 2)]
            tree = Prufer.decode(code)
            assert Prufer.encode(tree) == code
            assert _edge_set(Prufer.decode(Prufer.encode(tree))) == _edge_set(tree)

    def test_a_decoded_sequence_is_always_a_tree(self):
        rng = random.Random(389)
        for _ in range(20):
            n = rng.randint(2, 15)
            tree = Prufer.decode([rng.randrange(n) for _ in range(n - 2)])
            assert tree.edge_count() == n - 1
            assert SpanningTreeCount(tree).count == 1


class TestDegrees:
    def test_a_labels_count_plus_one_is_its_degree(self):
        code = [0, 0, 0, 2]
        tree = Prufer.decode(code)
        for label, degree in Prufer.implied_degrees(code).items():
            assert tree.degree(str(label)) == degree

    def test_the_random_tree_is_reproducible(self):
        a = _edge_set(Prufer.random_tree(12, seed=5))
        b = _edge_set(Prufer.random_tree(12, seed=5))
        assert a == b


class TestRefusals:
    def test_a_graph_with_a_cycle_is_refused(self):
        g = Graph()
        for n in "012":
            g.add_node(n)
        g.add_edge("0", "1")
        g.add_edge("1", "2")
        g.add_edge("2", "0")
        with pytest.raises(Invalid):
            Prufer.encode(g)

    def test_a_disconnected_graph_with_the_right_edge_count_is_refused(self):
        # an edge 0-1 plus a triangle on 2,3,4: four edges over five nodes,
        # the count of a tree, but two pieces
        f = Graph()
        for n in "01234":
            f.add_node(n)
        f.add_edge("0", "1")
        f.add_edge("2", "3")
        f.add_edge("3", "4")
        f.add_edge("4", "2")
        with pytest.raises(Invalid):
            Prufer.encode(f)

    def test_non_integer_labels_are_refused(self):
        g = Graph()
        g.add_node("a")
        g.add_node("b")
        g.add_edge("a", "b")
        with pytest.raises(Invalid):
            Prufer.encode(g)

    def test_a_code_entry_out_of_range_is_refused(self):
        with pytest.raises(Invalid):
            Prufer.decode([0, 7])

    def test_a_single_node_is_refused(self):
        g = Graph()
        g.add_node("0")
        with pytest.raises(Invalid):
            Prufer.encode(g)


class TestReport:
    def test_the_note_names_the_hub_degree(self):
        assert "0 has degree 4" in Prufer.note([0, 0, 0])
