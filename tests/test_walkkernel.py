from __future__ import annotations

import pytest

from mesh.errors import Invalid
from mesh.factories import complete, cycle, path, star
from mesh.graph import Graph
from mesh.walkkernel import WalkKernel


def _relabeled_cycle() -> Graph:
    g = Graph()
    names = ["p", "q", "r", "s", "t"]
    for n in names:
        g.add_node(n)
    # the same pentagon in a different insertion order and with different names
    for a, b in [("r", "p"), ("p", "t"), ("t", "q"), ("q", "s"), ("s", "r")]:
        g.add_edge(a, b)
    return g


class TestSelfSimilarity:
    def test_a_graph_against_itself_scores_one(self):
        k = WalkKernel()
        for g in (cycle(5), star(4), path(6), complete(4)):
            assert k.similarity(g, g) == pytest.approx(1.0)

    def test_isomorphic_graphs_score_one_and_share_a_hash(self):
        k = WalkKernel()
        a, b = cycle(5), _relabeled_cycle()
        assert k.similarity(a, b) == pytest.approx(1.0)
        assert WalkKernel.same_hash(a, b)


class TestDiscrimination:
    def test_a_cycle_is_closer_to_a_cycle_than_to_a_star(self):
        k = WalkKernel()
        assert k.similarity(cycle(5), cycle(6)) > k.similarity(cycle(5), star(5))

    def test_a_path_is_closer_to_a_path_than_to_a_clique(self):
        k = WalkKernel()
        assert k.similarity(path(5), path(6)) > k.similarity(path(5), complete(5))

    def test_the_score_is_symmetric(self):
        k = WalkKernel()
        assert k.similarity(star(4), path(5)) == pytest.approx(k.similarity(path(5), star(4)))

    def test_the_score_stays_within_the_unit_interval(self):
        k = WalkKernel()
        graphs = [cycle(4), cycle(6), star(3), star(6), path(4), complete(4)]
        for a in graphs:
            for b in graphs:
                s = k.similarity(a, b)
                assert 0.0 <= s <= 1.0 + 1e-9


class TestRawCount:
    def test_the_raw_kernel_starts_at_the_product_size(self):
        # two isolated nodes against one isolated node: two product nodes, no walks
        a = Graph()
        a.add_node("x")
        a.add_node("y")
        b = Graph()
        b.add_node("z")
        assert WalkKernel().raw(a, b) == 2.0

    def test_a_smaller_decay_leaves_a_smaller_last_term(self):
        quick = WalkKernel(decay=0.05)
        slow = WalkKernel(decay=0.2)
        quick.raw(cycle(5), cycle(5))
        slow.raw(cycle(5), cycle(5))
        assert quick.last_term < slow.last_term

    def test_an_empty_graph_scores_zero_against_anything(self):
        assert WalkKernel().similarity(Graph(), cycle(4)) == 0.0


class TestRefusal:
    def test_decay_and_depth_are_checked(self):
        with pytest.raises(Invalid):
            WalkKernel(decay=1.0)
        with pytest.raises(Invalid):
            WalkKernel(decay=0.0)
        with pytest.raises(Invalid):
            WalkKernel(depth=0)

    def test_directed_graphs_are_refused(self):
        with pytest.raises(Invalid):
            WalkKernel().raw(Graph(directed=True), cycle(3))


class TestReport:
    def test_the_note_names_the_decay_and_the_last_term(self):
        note = WalkKernel(decay=0.1, depth=6).note(cycle(4), cycle(4))
        assert "similarity 1.0000 at decay 0.1 over 6 terms" in note
        assert "last term" in note
