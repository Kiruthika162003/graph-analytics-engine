from __future__ import annotations

import random
from itertools import combinations

import pytest

from mesh.complement import Complement
from mesh.errors import Invalid
from mesh.factories import complete, cycle, path, star
from mesh.graph import Graph
from mesh.isomorphism import Isomorphism


def _brute_independence(g: Graph) -> int:
    nodes = g.nodes()
    for size in range(len(nodes), 0, -1):
        for subset in combinations(nodes, size):
            if not any(g.has_edge(a, b) for a, b in combinations(subset, 2)):
                return size
    return 0


class TestConstruction:
    def test_the_complement_of_a_complete_graph_is_edgeless(self):
        assert Complement(complete(5)).complement.edge_count() == 0

    def test_the_identities_hold_on_random_graphs(self):
        rng = random.Random(421)
        for _ in range(20):
            g = Graph()
            nodes = [str(i) for i in range(8)]
            for n in nodes:
                g.add_node(n)
            for a, b in combinations(nodes, 2):
                if rng.random() < 0.4:
                    g.add_edge(a, b)
            assert Complement(g).identities_hold()

    def test_a_five_cycle_is_self_complementary(self):
        c = Complement(cycle(5))
        assert Isomorphism(c.graph, c.complement).isomorphic
        assert c.graph.edge_count() == 5


class TestIndependentSet:
    def test_a_star_has_its_leaves_as_the_independent_set(self):
        c = Complement(star(4))
        assert c.maximum_independent_set() == {"1", "2", "3", "4"}

    def test_a_path_alternates(self):
        c = Complement(path(5))
        assert c.independence_number() == 3
        assert c.is_independent(c.maximum_independent_set())

    def test_the_set_is_always_independent_and_maximum(self):
        rng = random.Random(431)
        for _ in range(20):
            g = Graph()
            nodes = [str(i) for i in range(8)]
            for n in nodes:
                g.add_node(n)
            for a, b in combinations(nodes, 2):
                if rng.random() < 0.4:
                    g.add_edge(a, b)
            c = Complement(g)
            best = c.maximum_independent_set()
            assert c.is_independent(best)
            assert len(best) == _brute_independence(g)

    def test_independence_plus_clique_is_bounded_by_the_node_count_plus_one(self):
        # a clique and an independent set share at most one node
        c = Complement(cycle(6))
        assert c.independence_number() + c.clique_number() <= c.graph.node_count() + 1


class TestRefusal:
    def test_a_directed_graph_is_refused(self):
        with pytest.raises(Invalid):
            Complement(Graph(directed=True))


class TestReport:
    def test_the_note_states_both_numbers_and_the_density_flip(self):
        note = Complement(star(4)).note()
        assert "independence number 4 beside clique number 2" in note
        assert "flips to" in note
