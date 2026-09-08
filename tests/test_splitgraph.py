from __future__ import annotations

import random
from itertools import combinations

import pytest

from mesh.complement import Complement
from mesh.errors import Invalid
from mesh.factories import complete, cycle, path, star
from mesh.graph import Graph
from mesh.splitgraph import SplitGraph


def _brute_split(g: Graph) -> bool:
    nodes = g.nodes()
    for mask in range(1 << len(nodes)):
        clique = [nodes[i] for i in range(len(nodes)) if mask >> i & 1]
        fringe = [n for n in nodes if n not in clique]
        if all(g.has_edge(a, b) for a, b in combinations(clique, 2)) and not any(
            g.has_edge(a, b) for a, b in combinations(fringe, 2)
        ):
            return True
    return False


class TestVerdict:
    def test_a_star_is_split_with_the_hub_and_one_leaf_as_the_clique(self):
        # the guess was a clique of the hub alone; the degree rule takes m = 2, hub plus
        # the first leaf, and that pair is a clique too, so both partitions are valid
        sg = SplitGraph(star(4))
        assert sg.is_split
        assert sg.clique == {"0", "1"}
        assert sg.partition_holds()

    def test_a_complete_graph_and_an_edgeless_graph_are_split(self):
        assert SplitGraph(complete(4)).is_split
        g = Graph()
        for n in "abc":
            g.add_node(n)
        assert SplitGraph(g).is_split

    def test_a_square_is_not_split(self):
        sg = SplitGraph(cycle(4))
        assert not sg.is_split
        assert "not a split graph" in sg.note()

    def test_paths_are_split_but_two_separate_edges_are_not(self):
        # the guess was that a path of four is not split; it is, with the two inner
        # nodes as the clique and the two ends as the fringe. the forbidden shapes
        # are a square, a pentagon, and two edges with nothing between them
        assert SplitGraph(path(4)).is_split
        assert SplitGraph(path(3)).is_split
        two_edges = Graph()
        for n in "abcd":
            two_edges.add_node(n)
        two_edges.add_edge("a", "b")
        two_edges.add_edge("c", "d")
        assert not SplitGraph(two_edges).is_split
        assert not SplitGraph(cycle(5)).is_split

    def test_a_clique_with_pendants_is_split(self):
        g = complete(4)
        for i in range(3):
            g.add_node(f"p{i}")
            g.add_edge(str(i), f"p{i}")
        sg = SplitGraph(g)
        assert sg.is_split
        assert sg.clique == {"0", "1", "2", "3"}
        assert sg.partition_holds()


class TestInvariants:
    def test_split_graphs_are_chordal_and_closed_under_complement(self):
        rng = random.Random(457)
        checked = 0
        for _ in range(40):
            g = Graph()
            nodes = [str(i) for i in range(7)]
            for n in nodes:
                g.add_node(n)
            for a, b in combinations(nodes, 2):
                if rng.random() < 0.5:
                    g.add_edge(a, b)
            sg = SplitGraph(g)
            if sg.is_split:
                checked += 1
                assert sg.is_chordal_too()
                assert SplitGraph(Complement(g).complement).is_split
        assert checked > 0


class TestAgainstBruteForce:
    def test_the_degree_test_matches_trying_every_partition(self):
        rng = random.Random(461)
        for _ in range(40):
            g = Graph()
            nodes = [str(i) for i in range(7)]
            for n in nodes:
                g.add_node(n)
            for a, b in combinations(nodes, 2):
                if rng.random() < 0.5:
                    g.add_edge(a, b)
            sg = SplitGraph(g)
            assert sg.is_split == _brute_split(g)
            if sg.is_split:
                assert sg.partition_holds()


class TestRefusal:
    def test_a_directed_graph_is_refused(self):
        with pytest.raises(Invalid):
            SplitGraph(Graph(directed=True))


class TestReport:
    def test_the_note_states_the_part_sizes(self):
        note = SplitGraph(star(4)).note()
        assert "a clique of 2 and an independent fringe of 3" in note
