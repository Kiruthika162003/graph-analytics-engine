from __future__ import annotations

import random
from itertools import combinations

import pytest

from mesh.commonsubgraph import CommonSubgraph
from mesh.complement import Complement
from mesh.errors import Invalid
from mesh.factories import complete, cycle, path, star
from mesh.graph import Graph


def _relabel(g: Graph, prefix: str) -> Graph:
    h = Graph()
    for n in g.nodes():
        h.add_node(prefix + n)
    for u, v, _w in g.edges():
        h.add_edge(prefix + u, prefix + v)
    return h


class TestSizes:
    def test_a_graph_against_itself_shares_everything(self):
        cs = CommonSubgraph(cycle(5), _relabel(cycle(5), "r"))
        assert cs.size() == 5
        assert cs.shared_edges() == 5
        assert cs.is_consistent()
        assert cs.similarity() == 1.0

    def test_a_path_of_three_and_a_triangle_share_an_edge(self):
        cs = CommonSubgraph(path(3), cycle(3))
        assert cs.size() == 2
        assert cs.shared_edges() == 1

    def test_a_star_and_a_path_share_a_path_of_three(self):
        cs = CommonSubgraph(star(3), path(4))
        assert cs.size() == 3
        assert cs.shared_edges() == 2

    def test_a_clique_against_its_complement_shares_one_node(self):
        k = complete(4)
        cs = CommonSubgraph(k, Complement(k).complement)
        assert cs.size() == 1
        assert cs.shared_edges() == 0

    def test_a_square_inside_a_cube_face(self):
        cube = Graph()
        for i in range(8):
            cube.add_node(format(i, "03b"))
        for a in cube.nodes():
            for b in cube.nodes():
                if a < b and sum(x != y for x, y in zip(a, b, strict=True)) == 1:
                    cube.add_edge(a, b)
        cs = CommonSubgraph(cycle(4), cube)
        assert cs.size() == 4
        assert cs.shared_edges() == 4


class TestConsistency:
    def test_every_answer_is_a_consistent_correspondence_on_random_pairs(self):
        rng = random.Random(853)
        for _ in range(8):
            a = Graph()
            b = Graph()
            for i in range(5):
                a.add_node(f"a{i}")
                b.add_node(f"b{i}")
            for x, y in combinations(range(5), 2):
                if rng.random() < 0.5:
                    a.add_edge(f"a{x}", f"a{y}")
                if rng.random() < 0.5:
                    b.add_edge(f"b{x}", f"b{y}")
            cs = CommonSubgraph(a, b)
            assert cs.is_consistent()
            assert 1 <= cs.size() <= 5

    def test_the_answer_is_symmetric_in_size(self):
        one_way = CommonSubgraph(star(4), cycle(6)).size()
        assert one_way == CommonSubgraph(cycle(6), star(4)).size()


class TestRefusal:
    def test_directed_and_oversized_inputs_are_refused(self):
        with pytest.raises(Invalid):
            CommonSubgraph(Graph(directed=True), path(2))
        with pytest.raises(Invalid):
            CommonSubgraph(path(9), path(8))

    def test_an_empty_graph_shares_nothing(self):
        cs = CommonSubgraph(Graph(), path(3))
        assert cs.size() == 0
        assert cs.similarity() == 0.0


class TestReport:
    def test_the_note_states_size_edges_and_similarity(self):
        note = CommonSubgraph(path(3), cycle(3)).note()
        assert "common subgraph of 2 node(s) and 1 edge(s)" in note
        assert "similarity 0.67 over a product of 9" in note
