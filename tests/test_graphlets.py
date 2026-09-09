from __future__ import annotations

import random
from itertools import combinations
from math import comb

import pytest

from mesh.errors import Invalid
from mesh.factories import complete, cycle, path, star
from mesh.graph import Graph
from mesh.graphlets import Graphlets


def _random_graph(seed: int, n: int, p: float) -> Graph:
    rng = random.Random(seed)
    g = Graph()
    nodes = [str(i) for i in range(n)]
    for node in nodes:
        g.add_node(node)
    for a, b in combinations(nodes, 2):
        if rng.random() < p:
            g.add_edge(a, b)
    return g


class TestOrbits:
    def test_a_path_middle_is_orbit_two_and_its_ends_orbit_one(self):
        gl = Graphlets(path(3))
        assert gl.signature["1"] == [2, 0, 1, 0]
        assert gl.signature["0"] == [1, 1, 0, 0]
        assert gl.paths == 1
        assert gl.triangles == 0

    def test_a_triangle_corner_is_orbit_three_only(self):
        gl = Graphlets(cycle(3))
        assert all(sig == [2, 0, 0, 1] for sig in gl.signature.values())
        assert gl.triangles == 1

    def test_a_star_hub_is_the_middle_of_every_leaf_pair(self):
        gl = Graphlets(star(4))
        assert gl.signature["0"] == [4, 0, comb(4, 2), 0]
        assert gl.signature["1"] == [1, 3, 0, 0]

    def test_a_complete_graph_has_only_triangle_corners(self):
        gl = Graphlets(complete(5))
        assert all(sig == [4, 0, 0, comb(4, 2)] for sig in gl.signature.values())
        assert gl.triangles == comb(5, 3)


class TestIdentities:
    def test_the_orbit_identities_hold_on_random_graphs(self):
        for seed in range(857, 869):
            assert Graphlets(_random_graph(seed, 9, 0.4)).identities_hold()

    def test_a_cycle_has_n_paths_and_no_triangles(self):
        gl = Graphlets(cycle(7))
        assert gl.paths == 7
        assert gl.triangles == 0
        assert gl.identities_hold()


class TestPeers:
    def test_leaves_of_a_star_are_each_others_peers_and_the_hub_is_far(self):
        gl = Graphlets(star(4))
        ranked = gl.peers("1")
        assert [n for n, _d in ranked[:3]] == ["2", "3", "4"]
        assert ranked[-1][0] == "0"
        assert ranked[0][1] == 0.0

    def test_same_degree_different_role(self):
        # the middle of a path and a triangle corner both have degree two
        g = Graph()
        for n in "abcxyz":
            g.add_node(n)
        g.add_edge("a", "b")
        g.add_edge("b", "c")
        for u, v in combinations("xyz", 2):
            g.add_edge(u, v)
        gl = Graphlets(g)
        assert gl.signature["b"][0] == gl.signature["x"][0] == 2
        assert Graphlets.distance(gl.signature["b"], gl.signature["x"]) > 0
        assert Graphlets.distance(gl.signature["x"], gl.signature["y"]) == 0.0

    def test_an_unknown_node_is_refused(self):
        with pytest.raises(Invalid):
            Graphlets(path(3)).peers("zz")


class TestRefusal:
    def test_a_directed_graph_is_refused(self):
        with pytest.raises(Invalid):
            Graphlets(Graph(directed=True))


class TestReport:
    def test_the_note_reads_the_four_orbits(self):
        note = Graphlets(star(3)).note("0")
        assert "0: degree 3, path end 0 time(s), path middle 3, triangle corner 0" in note
        assert "3 induced path(s) and 0 triangle(s) overall" in note
