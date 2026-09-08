from __future__ import annotations

import random
from itertools import combinations

import pytest

from mesh.errors import Invalid
from mesh.factories import complete, cycle, star
from mesh.graph import Graph
from mesh.kcore import KCore
from mesh.ktruss import KTruss


def _truss_by_definition(g: Graph, k: int) -> set[frozenset[str]]:
    # repeatedly drop edges in fewer than k-2 surviving triangles
    alive = {frozenset((u, v)) for u, v, _w in g.edges()}
    changed = True
    while changed:
        changed = False
        for edge in list(alive):
            u, v = tuple(edge)
            common = sum(
                1
                for w in g.nodes()
                if frozenset((u, w)) in alive and frozenset((v, w)) in alive
            )
            if common < k - 2:
                alive.discard(edge)
                changed = True
    return alive


class TestTruss:
    def test_a_complete_graph_is_one_truss_at_n(self):
        kt = KTruss(complete(5))
        assert kt.truss_number() == 5
        assert kt.k_truss(5).edge_count() == 10

    def test_a_star_has_no_triangles_and_truss_number_two(self):
        kt = KTruss(star(5))
        assert kt.truss_number() == 2
        assert kt.k_truss(3).edge_count() == 0

    def test_a_cycle_survives_only_the_two_truss(self):
        kt = KTruss(cycle(6))
        assert all(t == 2 for t in kt.truss.values())

    def test_a_hub_sits_in_a_high_core_but_a_low_truss(self):
        # a star of eight: degeneracy 1 and truss number 2 both low here, so
        # build a hub joined to two disjoint triangles: core 2, truss 3 on
        # the triangle edges but the spokes fall out of the 3-truss
        g = Graph()
        for n in ["hub", "a", "b", "c", "d", "e", "f"]:
            g.add_node(n)
        for x, y in combinations("abc", 2):
            g.add_edge(x, y)
        for x, y in combinations("def", 2):
            g.add_edge(x, y)
        for leaf in "abcdef":
            g.add_edge("hub", leaf)
        kt = KTruss(g)
        assert kt.k_truss(4).edge_count() == 12  # hub plus each triangle is a K4
        assert KCore(g).degeneracy() == 3

    def test_the_result_is_inside_the_core(self):
        g = complete(4)
        g.add_node("tail")
        g.add_edge("0", "tail")
        assert KTruss(g).k_truss(4).edge_count() == 6


class TestRefusals:
    def test_a_directed_graph_is_refused(self):
        with pytest.raises(Invalid):
            KTruss(Graph(directed=True))

    def test_k_below_two_is_refused(self):
        with pytest.raises(Invalid):
            KTruss(complete(3)).k_truss(1)


class TestAgainstDefinition:
    def test_every_k_truss_matches_iterative_stripping(self):
        rng = random.Random(401)
        for _ in range(25):
            g = Graph()
            nodes = [str(i) for i in range(9)]
            for n in nodes:
                g.add_node(n)
            for a, b in combinations(nodes, 2):
                if rng.random() < 0.45:
                    g.add_edge(a, b)
            kt = KTruss(g)
            for k in range(2, kt.truss_number() + 2):
                got = {frozenset((u, v)) for u, v, _w in kt.k_truss(k).edges()}
                assert got == _truss_by_definition(g, k)


class TestReport:
    def test_the_note_sets_truss_beside_core(self):
        note = KTruss(complete(4)).note()
        assert "truss number 4 against core number 3" in note
