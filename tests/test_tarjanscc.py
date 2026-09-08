from __future__ import annotations

import random

import pytest

from mesh.errors import Invalid, Missing
from mesh.graph import Graph
from mesh.tarjanscc import TarjanSCC


def _reaches(g: Graph, src: str, dst: str) -> bool:
    seen = {src}
    stack = [src]
    while stack:
        node = stack.pop()
        if node == dst:
            return True
        for nbr in g.neighbors(node):
            if nbr not in seen:
                seen.add(nbr)
                stack.append(nbr)
    return src == dst or dst in seen


class TestComponents:
    def test_a_directed_cycle_is_one_component(self):
        g = Graph(directed=True)
        for n in "abc":
            g.add_node(n)
        g.add_edge("a", "b")
        g.add_edge("b", "c")
        g.add_edge("c", "a")
        scc = TarjanSCC(g)
        assert scc.count() == 1
        assert scc.is_strongly_connected()

    def test_a_dag_is_all_singletons(self):
        g = Graph(directed=True)
        for n in "abc":
            g.add_node(n)
        g.add_edge("a", "b")
        g.add_edge("b", "c")
        scc = TarjanSCC(g)
        assert scc.count() == 3

    def test_two_cycles_joined_one_way_stay_separate(self):
        # {a,b} cycle, {c,d} cycle, a single edge b->c does not merge them
        g = Graph(directed=True)
        for n in "abcd":
            g.add_node(n)
        g.add_edge("a", "b")
        g.add_edge("b", "a")
        g.add_edge("c", "d")
        g.add_edge("d", "c")
        g.add_edge("b", "c")
        scc = TarjanSCC(g)
        assert scc.count() == 2
        assert scc.component_of("a") == {"a", "b"}


class TestRefusals:
    def test_an_undirected_graph_is_refused(self):
        with pytest.raises(Invalid):
            TarjanSCC(Graph(directed=False))

    def test_a_missing_node_is_refused(self):
        g = Graph(directed=True)
        g.add_node("a")
        with pytest.raises(Missing):
            TarjanSCC(g).component_of("ghost")


class TestAgainstMutualReachability:
    def test_membership_matches_mutual_reachability(self):
        rng = random.Random(44)
        for _ in range(30):
            g = Graph(directed=True)
            nodes = [str(i) for i in range(8)]
            for n in nodes:
                g.add_node(n)
            for u in nodes:
                for v in nodes:
                    if u != v and rng.random() < 0.25:
                        g.add_edge(u, v)
            scc = TarjanSCC(g)
            comp = {n: scc.component_of(n) for n in nodes}
            for a in nodes:
                for b in nodes:
                    mutual = _reaches(g, a, b) and _reaches(g, b, a)
                    same = comp[a] == comp[b]
                    assert same == mutual

    def test_the_condensation_is_acyclic(self):
        # no edge should point from a later component back into an earlier one
        rng = random.Random(45)
        g = Graph(directed=True)
        nodes = [str(i) for i in range(10)]
        for n in nodes:
            g.add_node(n)
        for u in nodes:
            for v in nodes:
                if u != v and rng.random() < 0.2:
                    g.add_edge(u, v)
        scc = TarjanSCC(g)
        label: dict[str, int] = {}
        for i, comp in enumerate(scc.components()):
            for node in comp:
                label[node] = i
        # within-SCC edges are fine; cross-SCC edges must not form a cycle,
        # which holds if the component DAG has a consistent topological order
        cross: set[tuple[int, int]] = set()
        for u, v, _w in g.edges():
            if label[u] != label[v]:
                cross.add((label[u], label[v]))
        for a, b in cross:
            assert (b, a) not in cross


class TestReport:
    def test_the_note_counts_components(self):
        g = Graph(directed=True)
        g.add_node("a")
        assert "1 strongly connected component(s)" in TarjanSCC(g).note()
