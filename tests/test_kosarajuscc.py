from __future__ import annotations

import random

import pytest

from mesh.errors import Invalid, Missing
from mesh.graph import Graph
from mesh.kosarajuscc import KosarajuSCC
from mesh.tarjanscc import TarjanSCC


class TestComponents:
    def test_a_directed_cycle_is_one_component(self):
        g = Graph(directed=True)
        for n in "abc":
            g.add_node(n)
        g.add_edge("a", "b")
        g.add_edge("b", "c")
        g.add_edge("c", "a")
        assert KosarajuSCC(g).count() == 1

    def test_a_dag_is_all_singletons(self):
        g = Graph(directed=True)
        for n in "abc":
            g.add_node(n)
        g.add_edge("a", "b")
        g.add_edge("b", "c")
        assert KosarajuSCC(g).count() == 3

    def test_it_groups_a_two_node_cycle(self):
        g = Graph(directed=True)
        g.add_node("a")
        g.add_node("b")
        g.add_edge("a", "b")
        g.add_edge("b", "a")
        assert KosarajuSCC(g).component_of("a") == {"a", "b"}


class TestRefusals:
    def test_an_undirected_graph_is_refused(self):
        with pytest.raises(Invalid):
            KosarajuSCC(Graph(directed=False))

    def test_a_missing_node_is_refused(self):
        g = Graph(directed=True)
        g.add_node("a")
        with pytest.raises(Missing):
            KosarajuSCC(g).component_of("ghost")


class TestAgreesWithTarjan:
    def test_the_two_algorithms_find_the_same_partition(self):
        rng = random.Random(88)
        for _ in range(40):
            g = Graph(directed=True)
            nodes = [str(i) for i in range(9)]
            for n in nodes:
                g.add_node(n)
            for u in nodes:
                for v in nodes:
                    if u != v and rng.random() < 0.22:
                        g.add_edge(u, v)
            k = KosarajuSCC(g)
            t = TarjanSCC(g)
            assert k.count() == t.count()
            for node in nodes:
                assert k.component_of(node) == t.component_of(node)


class TestReport:
    def test_the_note_counts_components(self):
        g = Graph(directed=True)
        g.add_node("a")
        assert "1 strongly connected component(s)" in KosarajuSCC(g).note()
