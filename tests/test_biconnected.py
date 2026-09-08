from __future__ import annotations

import random
from itertools import combinations

import pytest

from mesh.articulation import Articulation
from mesh.biconnected import Biconnected
from mesh.errors import Invalid, Missing
from mesh.graph import Graph


def _bowtie() -> Graph:
    # two triangles sharing the node c: two blocks, c is the cut node
    g = Graph()
    for n in "abcde":
        g.add_node(n)
    for u, v in [("a", "b"), ("b", "c"), ("c", "a"), ("c", "d"), ("d", "e"), ("e", "c")]:
        g.add_edge(u, v)
    return g


class TestBlocks:
    def test_a_bowtie_has_two_blocks_at_one_cut_node(self):
        b = Biconnected(_bowtie())
        assert len(b.blocks) == 2
        assert b.cut_nodes == {"c"}

    def test_every_edge_lies_in_exactly_one_block(self):
        g = _bowtie()
        b = Biconnected(g)
        seen: list[frozenset[str]] = []
        for block in b.blocks:
            seen.extend(block)
        assert sorted(map(sorted, seen)) == sorted(sorted((u, v)) for u, v, _w in g.edges())

    def test_the_cut_node_belongs_to_both_blocks(self):
        b = Biconnected(_bowtie())
        assert len(b.blocks_of("c")) == 2
        assert len(b.blocks_of("a")) == 1

    def test_a_cycle_is_a_single_block(self):
        g = Graph()
        for n in "abcd":
            g.add_node(n)
        for u, v in [("a", "b"), ("b", "c"), ("c", "d"), ("d", "a")]:
            g.add_edge(u, v)
        b = Biconnected(g)
        assert b.is_biconnected()

    def test_a_path_is_one_block_per_edge(self):
        g = Graph()
        for n in "abc":
            g.add_node(n)
        g.add_edge("a", "b")
        g.add_edge("b", "c")
        b = Biconnected(g)
        assert len(b.blocks) == 2
        assert b.cut_nodes == {"b"}


class TestBlockCutTree:
    def test_the_tree_alternates_blocks_and_cuts(self):
        tree = Biconnected(_bowtie()).block_cut_tree()
        assert tree.node_count() == 3  # two blocks and one cut node
        assert tree.edge_count() == 2
        assert tree.degree("C:c") == 2


class TestRefusals:
    def test_a_directed_graph_is_refused(self):
        with pytest.raises(Invalid):
            Biconnected(Graph(directed=True))

    def test_a_missing_node_is_refused(self):
        with pytest.raises(Missing):
            Biconnected(_bowtie()).blocks_of("ghost")


class TestAgainstArticulation:
    def test_cut_nodes_match_the_articulation_finder_on_random_graphs(self):
        rng = random.Random(139)
        for _ in range(30):
            g = Graph()
            nodes = [str(i) for i in range(8)]
            for n in nodes:
                g.add_node(n)
            for a, c in combinations(nodes, 2):
                if rng.random() < 0.3:
                    g.add_edge(a, c)
            b = Biconnected(g)
            assert b.cut_nodes == Articulation(g).points
            # and the blocks partition the edge set
            total = sum(len(block) for block in b.blocks)
            assert total == g.edge_count()


class TestReport:
    def test_the_note_counts_blocks_and_cuts(self):
        assert "2 block(s) glued at 1 cut node(s)" in Biconnected(_bowtie()).note()
