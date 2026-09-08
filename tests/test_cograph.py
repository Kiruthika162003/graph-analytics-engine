from __future__ import annotations

import random
from itertools import combinations, permutations

import pytest

from mesh.bronkerbosch import BronKerbosch
from mesh.chromatic import ChromaticNumber
from mesh.cograph import Cograph
from mesh.complement import Complement
from mesh.errors import Invalid
from mesh.factories import complete, cycle, path, star
from mesh.graph import Graph


def _has_induced_p4(g: Graph) -> bool:
    for four in combinations(g.nodes(), 4):
        for a, b, c, d in permutations(four):
            if (
                g.has_edge(a, b)
                and g.has_edge(b, c)
                and g.has_edge(c, d)
                and not (g.has_edge(a, c) or g.has_edge(a, d) or g.has_edge(b, d))
            ):
                return True
    return False


def _random_cograph(rng: random.Random, names: list[str]) -> Graph:
    # build by random unions and joins so the answer is known to be a cograph
    if len(names) == 1:
        g = Graph()
        g.add_node(names[0])
        return g
    cut = rng.randrange(1, len(names))
    left = _random_cograph(rng, names[:cut])
    right = _random_cograph(rng, names[cut:])
    g = Graph()
    for part in (left, right):
        for n in part.nodes():
            g.add_node(n)
        for u, v, _w in part.edges():
            g.add_edge(u, v)
    if rng.random() < 0.5:
        for u in left.nodes():
            for v in right.nodes():
                g.add_edge(u, v)
    return g


class TestVerdict:
    def test_a_path_of_four_is_the_smallest_non_cograph(self):
        c = Cograph(path(4))
        assert not c.is_cograph
        assert c.witness == ("0", "1", "2", "3")
        assert "induced path 0-1-2-3" in c.note()

    def test_shapes_built_by_union_and_join_are_cographs(self):
        for g in (complete(5), star(4), cycle(4), path(3)):
            assert Cograph(g).is_cograph

    def test_a_pentagon_hides_a_p4_and_is_not_a_cograph(self):
        assert not Cograph(cycle(5)).is_cograph

    def test_an_empty_graph_is_a_cograph_of_depth_zero(self):
        c = Cograph(Graph())
        assert c.is_cograph
        assert c.depth == 0


class TestNumbers:
    def test_the_cotree_numbers_match_the_other_modules_on_random_cographs(self):
        rng = random.Random(467)
        for _ in range(25):
            g = _random_cograph(rng, [str(i) for i in range(8)])
            c = Cograph(g)
            assert c.is_cograph
            assert c.clique_number == BronKerbosch(g).clique_number()
            assert c.independence_number == Complement(g).independence_number()
            assert c.chromatic_number() == ChromaticNumber(g).value

    def test_a_complete_bipartite_graph_is_one_join_of_two_unions(self):
        g = Graph()
        for n in "abcxyz":
            g.add_node(n)
        for u in "abc":
            for v in "xyz":
                g.add_edge(u, v)
        c = Cograph(g)
        assert c.joins == 1
        assert c.unions == 2
        assert c.clique_number == 2
        assert c.independence_number == 3

    def test_the_numbers_are_refused_off_a_cograph(self):
        with pytest.raises(Invalid):
            Cograph(path(5)).chromatic_number()


class TestAgainstBruteForce:
    def test_the_verdict_matches_searching_for_an_induced_p4(self):
        rng = random.Random(469)
        for _ in range(30):
            g = Graph()
            nodes = [str(i) for i in range(7)]
            for n in nodes:
                g.add_node(n)
            for a, b in combinations(nodes, 2):
                if rng.random() < 0.5:
                    g.add_edge(a, b)
            assert Cograph(g).is_cograph == (not _has_induced_p4(g))


class TestRefusal:
    def test_a_directed_graph_is_refused(self):
        with pytest.raises(Invalid):
            Cograph(Graph(directed=True))


class TestReport:
    def test_the_note_counts_operations_and_states_the_numbers(self):
        note = Cograph(star(3)).note()
        assert "1 union(s) and 1 join(s)" in note
        assert "clique number 2, independence number 3, chromatic number 2" in note
