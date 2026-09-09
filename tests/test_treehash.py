from __future__ import annotations

import random
from itertools import permutations

import pytest

from mesh.errors import Invalid
from mesh.factories import cycle, path, star
from mesh.graph import Graph
from mesh.prufer import Prufer
from mesh.treehash import TreeHash


def _relabel(tree: Graph, rng: random.Random) -> Graph:
    names = tree.nodes()
    shuffled = list(names)
    rng.shuffle(shuffled)
    mapping = dict(zip(names, shuffled, strict=True))
    g = Graph()
    for n in shuffled:
        g.add_node(n)
    for u, v, _w in tree.edges():
        g.add_edge(mapping[u], mapping[v])
    return g


class TestCanonicalNames:
    def test_a_leaf_pair_and_a_path_of_three(self):
        assert TreeHash(path(2)).canonical == "(())"
        assert TreeHash(path(3)).canonical == "(()())"

    def test_a_star_names_its_leaves_under_the_hub(self):
        assert TreeHash(star(3)).canonical == "(()()())"

    def test_relabeling_never_changes_the_name(self):
        rng = random.Random(757)
        for _ in range(20):
            tree = Prufer.decode([rng.randrange(9) for _ in range(7)])
            assert TreeHash(tree).same_shape(TreeHash(_relabel(tree, rng)))

    def test_different_shapes_get_different_names(self):
        assert not TreeHash(path(4)).same_shape(TreeHash(star(3)))


class TestCounting:
    def test_labeled_trees_on_four_and_five_nodes_collapse_to_two_and_three_shapes(self):
        four = [Prufer.decode(list(code)) for code in permutations(range(4), 2)]
        four += [Prufer.decode([i, i]) for i in range(4)]
        assert TreeHash.distinct_shapes(four) == 2
        five = []
        for a in range(5):
            for b in range(5):
                for c in range(5):
                    five.append(Prufer.decode([a, b, c]))
        assert TreeHash.distinct_shapes(five) == 3

    def test_repeated_subtrees_are_found(self):
        # a spider: hub with three legs of two nodes each
        g = Graph()
        g.add_node("hub")
        for leg in "abc":
            g.add_node(leg)
            g.add_node(leg + "2")
            g.add_edge("hub", leg)
            g.add_edge(leg, leg + "2")
        th = TreeHash(g)
        assert th.repeated_subtrees() == {"(())": 3}


class TestCenters:
    def test_an_even_path_has_two_centers_and_both_give_one_name(self):
        th = TreeHash(path(6))
        assert th.centers == ["2", "3"]
        assert th.canonical == th.names_by_root["2"]["2"]

    def test_a_single_node_and_an_empty_graph(self):
        g = Graph()
        g.add_node("solo")
        assert TreeHash(g).canonical == "()"
        assert TreeHash(Graph()).canonical == ""


class TestRefusal:
    def test_a_cycle_and_a_forest_are_refused(self):
        with pytest.raises(Invalid):
            TreeHash(cycle(4))
        g = Graph()
        for n in "abcd":
            g.add_node(n)
        g.add_edge("a", "b")
        g.add_edge("c", "d")
        with pytest.raises(Invalid):
            TreeHash(g)


class TestReport:
    def test_the_note_names_the_centers_and_the_repeats(self):
        note = TreeHash(star(3)).note()
        assert "tree of 4 rooted at ['0']" in note
        assert "0 repeated sub-shape(s)" in note
