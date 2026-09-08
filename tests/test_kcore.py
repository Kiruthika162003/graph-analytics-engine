from __future__ import annotations

import random

import pytest

from mesh.errors import Invalid
from mesh.graph import Graph
from mesh.kcore import KCore


def _triangle_with_tail() -> Graph:
    # a-b-c triangle, plus a chain c-d-e hanging off
    g = Graph()
    for n in "abcde":
        g.add_node(n)
    for u, v in [("a", "b"), ("b", "c"), ("c", "a"), ("c", "d"), ("d", "e")]:
        g.add_edge(u, v)
    return g


def _core_by_definition(g: Graph, k: int) -> set[str]:
    # repeatedly drop nodes with fewer than k neighbors inside the survivor set
    alive = set(g.nodes())
    changed = True
    while changed:
        changed = False
        for n in list(alive):
            inside = sum(1 for m in g.neighbors(n) if m in alive)
            if inside < k:
                alive.discard(n)
                changed = True
    return alive


class TestCoreNumbers:
    def test_the_triangle_is_the_two_core(self):
        kc = KCore(_triangle_with_tail())
        assert kc.k_core(2) == {"a", "b", "c"}

    def test_the_tail_has_core_number_one(self):
        kc = KCore(_triangle_with_tail())
        assert kc.core["d"] == 1
        assert kc.core["e"] == 1

    def test_the_degeneracy_is_the_innermost_core(self):
        assert KCore(_triangle_with_tail()).degeneracy() == 2

    def test_a_high_degree_hub_with_leaves_has_core_one(self):
        # degree does not equal depth: a star's hub is only in the 1-core
        g = Graph()
        g.add_node("hub")
        for leaf in "abcdef":
            g.add_node(leaf)
            g.add_edge("hub", leaf)
        kc = KCore(g)
        assert kc.core["hub"] == 1
        assert kc.degeneracy() == 1

    def test_the_zero_core_is_everything(self):
        g = _triangle_with_tail()
        assert KCore(g).k_core(0) == set(g.nodes())


class TestRefusals:
    def test_a_directed_graph_is_refused(self):
        with pytest.raises(Invalid):
            KCore(Graph(directed=True))

    def test_a_negative_k_is_refused(self):
        with pytest.raises(Invalid):
            KCore(_triangle_with_tail()).k_core(-1)


class TestAgainstDefinition:
    def test_every_k_core_matches_iterative_stripping(self):
        rng = random.Random(19)
        for _ in range(30):
            g = Graph()
            nodes = [str(i) for i in range(9)]
            for n in nodes:
                g.add_node(n)
            for u in nodes:
                for v in nodes:
                    if u < v and rng.random() < 0.35:
                        g.add_edge(u, v)
            kc = KCore(g)
            for k in range(0, kc.degeneracy() + 2):
                assert kc.k_core(k) == _core_by_definition(g, k)


class TestReport:
    def test_the_note_contrasts_degeneracy_with_max_degree(self):
        note = KCore(_triangle_with_tail()).note()
        assert "degeneracy 2 against max degree 3" in note
