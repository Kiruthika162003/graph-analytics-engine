from __future__ import annotations

import random
from itertools import combinations

import pytest

from mesh.dominatingset import DominatingSet
from mesh.errors import Invalid
from mesh.graph import Graph


def _star(leaves: int) -> Graph:
    g = Graph()
    g.add_node("hub")
    for i in range(leaves):
        g.add_node(f"l{i}")
        g.add_edge("hub", f"l{i}")
    return g


def _path(k: int) -> Graph:
    g = Graph()
    nodes = [str(i) for i in range(k)]
    for n in nodes:
        g.add_node(n)
    for i in range(k - 1):
        g.add_edge(nodes[i], nodes[i + 1])
    return g


class TestGreedy:
    def test_a_star_is_dominated_by_its_hub_alone(self):
        ds = DominatingSet(_star(5))
        assert ds.chosen == ["hub"]
        assert ds.dominates(ds.chosen)

    def test_a_path_of_six_needs_two(self):
        ds = DominatingSet(_path(6))
        assert len(ds.chosen) == 2
        assert ds.optimum() == 2

    def test_the_result_always_dominates(self):
        rng = random.Random(373)
        for _ in range(20):
            g = Graph()
            nodes = [str(i) for i in range(10)]
            for n in nodes:
                g.add_node(n)
            for a, b in combinations(nodes, 2):
                if rng.random() < 0.25:
                    g.add_edge(a, b)
            ds = DominatingSet(g)
            assert ds.dominates(ds.chosen)

    def test_isolated_nodes_must_all_be_chosen(self):
        g = Graph()
        for n in "abc":
            g.add_node(n)
        assert sorted(DominatingSet(g).chosen) == ["a", "b", "c"]

    def test_the_middle_of_a_short_path_dominates_alone(self):
        ds = DominatingSet(_path(3))
        assert ds.chosen == ["1"]


class TestBounds:
    def test_greedy_stays_within_the_logarithmic_bound(self):
        rng = random.Random(379)
        for _ in range(20):
            g = Graph()
            nodes = [str(i) for i in range(9)]
            for n in nodes:
                g.add_node(n)
            for a, b in combinations(nodes, 2):
                if rng.random() < 0.3:
                    g.add_edge(a, b)
            ds = DominatingSet(g)
            best = ds.optimum()
            assert best <= len(ds.chosen) <= ds.log_bound(best) + 1e-9


class TestRefusals:
    def test_a_directed_graph_is_refused(self):
        with pytest.raises(Invalid):
            DominatingSet(Graph(directed=True))

    def test_an_empty_graph_is_refused(self):
        with pytest.raises(Invalid):
            DominatingSet(Graph())

    def test_the_exact_optimum_is_capped(self):
        with pytest.raises(Invalid):
            DominatingSet(_path(15)).optimum()


class TestReport:
    def test_the_note_states_greedy_against_optimum(self):
        note = DominatingSet(_star(4)).note()
        assert "greedy dominating set of 1 against optimum 1" in note
