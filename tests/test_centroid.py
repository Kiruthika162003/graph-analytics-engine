from __future__ import annotations

import math
import random

import pytest

from mesh.centroid import CentroidDecomposition
from mesh.errors import Invalid, Missing
from mesh.graph import Graph


def _path(k: int) -> Graph:
    g = Graph()
    nodes = [str(i) for i in range(k)]
    for n in nodes:
        g.add_node(n)
    for i in range(k - 1):
        g.add_edge(nodes[i], nodes[i + 1])
    return g


def _random_tree(rng: random.Random, n: int) -> Graph:
    g = Graph()
    g.add_node("0")
    for i in range(1, n):
        g.add_node(str(i))
        g.add_edge(str(rng.randrange(i)), str(i))
    return g


class TestDecomposition:
    def test_a_long_path_gets_a_logarithmic_centroid_tree(self):
        cd = CentroidDecomposition(_path(1000))
        assert cd.original_depth() == 999
        assert cd.depth() <= math.ceil(math.log2(1001))

    def test_the_first_centroid_of_a_path_is_its_middle(self):
        cd = CentroidDecomposition(_path(7))
        root = next(n for n, p in cd.centroid_parent.items() if p is None)
        assert root == "3"

    def test_every_node_appears_exactly_once(self):
        g = _path(10)
        cd = CentroidDecomposition(g)
        assert sorted(cd.centroid_parent) == sorted(g.nodes())

    def test_the_halving_property_holds_at_every_centroid(self):
        rng = random.Random(233)
        for _ in range(20):
            cd = CentroidDecomposition(_random_tree(rng, rng.randint(1, 60)))
            assert cd.halving_holds()

    def test_a_single_node_is_its_own_centroid(self):
        g = Graph()
        g.add_node("solo")
        cd = CentroidDecomposition(g)
        assert cd.depth() == 0
        assert cd.ancestor_of("solo") == ["solo"]

    def test_the_ancestor_chain_climbs_to_the_root_centroid(self):
        cd = CentroidDecomposition(_path(7))
        chain = cd.ancestor_of("0")
        assert chain[-1] == "3"
        assert chain[0] == "0"


class TestRefusals:
    def test_a_directed_graph_is_refused(self):
        with pytest.raises(Invalid):
            CentroidDecomposition(Graph(directed=True))

    def test_a_cycle_is_refused(self):
        g = _path(3)
        g.add_edge("0", "2")
        with pytest.raises(Invalid):
            CentroidDecomposition(g)

    def test_a_missing_node_is_refused(self):
        with pytest.raises(Missing):
            CentroidDecomposition(_path(3)).ancestor_of("ghost")


class TestDepthBound:
    def test_depth_never_exceeds_the_log_bound_on_random_trees(self):
        rng = random.Random(239)
        for _ in range(20):
            n = rng.randint(2, 200)
            cd = CentroidDecomposition(_random_tree(rng, n))
            assert cd.depth() <= math.ceil(math.log2(n + 1))


class TestReport:
    def test_the_note_contrasts_the_two_depths(self):
        note = CentroidDecomposition(_path(64)).note()
        assert "against original depth 63" in note
