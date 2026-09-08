from __future__ import annotations

from itertools import combinations

import pytest

from mesh.errors import Invalid
from mesh.graph import Graph
from mesh.subgraphmatch import SubgraphMatch
from mesh.triangles import Triangles


def _complete(k: int, prefix: str = "") -> Graph:
    g = Graph()
    nodes = [f"{prefix}{i}" for i in range(k)]
    for n in nodes:
        g.add_node(n)
    for a, b in combinations(nodes, 2):
        g.add_edge(a, b)
    return g


def _path(k: int, prefix: str = "p") -> Graph:
    g = Graph()
    nodes = [f"{prefix}{i}" for i in range(k)]
    for n in nodes:
        g.add_node(n)
    for i in range(k - 1):
        g.add_edge(nodes[i], nodes[i + 1])
    return g


class TestCounting:
    def test_a_triangle_pattern_counts_the_graphs_triangles(self):
        g = _complete(5)
        m = SubgraphMatch(g, _complete(3, "t"))
        assert m.copies == Triangles(g).total == 10
        assert m.automorphisms == 6

    def test_raw_mappings_are_copies_times_automorphisms(self):
        m = SubgraphMatch(_complete(4), _complete(3, "t"))
        assert m.raw == m.copies * m.automorphisms

    def test_an_edge_pattern_counts_edges(self):
        g = _path(5)
        m = SubgraphMatch(g, _path(2, "e"))
        assert m.copies == 4

    def test_induced_matching_excludes_chorded_copies(self):
        # a path of three inside K4 is never induced: every pair is adjacent
        loose = SubgraphMatch(_complete(4), _path(3, "q"), induced=False)
        strict = SubgraphMatch(_complete(4), _path(3, "q"), induced=True)
        assert loose.copies == 12
        assert strict.copies == 0

    def test_a_missing_pattern_does_not_occur(self):
        m = SubgraphMatch(_path(4), _complete(3, "t"))
        assert not m.occurs()
        assert m.witness is None

    def test_the_witness_is_a_valid_embedding(self):
        g = _complete(4)
        m = SubgraphMatch(g, _path(3, "q"))
        assert m.witness is not None
        for u, v, _w in _path(3, "q").edges():
            assert g.has_edge(m.witness[u], m.witness[v])


class TestDirected:
    def test_direction_is_respected(self):
        g = Graph(directed=True)
        for n in "abc":
            g.add_node(n)
        g.add_edge("a", "b")
        g.add_edge("b", "c")
        g.add_edge("a", "c")  # a feed-forward loop
        ffl = Graph(directed=True)
        for n in "xyz":
            ffl.add_node(n)
        ffl.add_edge("x", "y")
        ffl.add_edge("y", "z")
        ffl.add_edge("x", "z")
        assert SubgraphMatch(g, ffl).copies == 1
        cycle = Graph(directed=True)
        for n in "xyz":
            cycle.add_node(n)
        cycle.add_edge("x", "y")
        cycle.add_edge("y", "z")
        cycle.add_edge("z", "x")
        assert not SubgraphMatch(g, cycle).occurs()


class TestRefusals:
    def test_mixed_directedness_is_refused(self):
        with pytest.raises(Invalid):
            SubgraphMatch(Graph(directed=True), Graph())

    def test_a_pattern_larger_than_the_graph_is_refused(self):
        with pytest.raises(Invalid):
            SubgraphMatch(_path(2), _path(3, "q"))


class TestReport:
    def test_the_note_states_copies_raw_and_automorphisms(self):
        note = SubgraphMatch(_complete(4), _complete(3, "t")).note()
        assert "4 distinct cop(ies) from 24 raw mapping(s) over 6 automorphism(s)" in note
