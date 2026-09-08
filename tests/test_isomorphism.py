from __future__ import annotations

import random
from itertools import combinations, permutations

import pytest

from mesh.errors import Invalid
from mesh.graph import Graph
from mesh.isomorphism import Isomorphism


def _relabel(g: Graph, names: dict[str, str]) -> Graph:
    h = Graph(directed=g.directed)
    for n in g.nodes():
        h.add_node(names[n])
    for u, v, w in g.edges():
        h.add_edge(names[u], names[v], w)
    return h


def _brute_isomorphic(a: Graph, b: Graph) -> bool:
    if a.node_count() != b.node_count():
        return False
    na, nb = a.nodes(), b.nodes()
    for perm in permutations(nb):
        m = dict(zip(na, perm, strict=True))
        if all(b.has_edge(m[u], m[v]) for u, v, _w in a.edges()) and all(
            a.has_edge(u, v) for u, v, _w in _relabel(b, {v: k for k, v in m.items()}).edges()
        ):
            return True
    return False


def _cycle(k: int, prefix: str) -> Graph:
    g = Graph()
    nodes = [f"{prefix}{i}" for i in range(k)]
    for n in nodes:
        g.add_node(n)
    for i in range(k):
        g.add_edge(nodes[i], nodes[(i + 1) % k])
    return g


class TestVerdict:
    def test_a_relabelled_copy_is_isomorphic_with_a_valid_mapping(self):
        g = _cycle(5, "a")
        h = _relabel(g, {f"a{i}": f"z{(i * 3) % 5}" for i in range(5)})
        iso = Isomorphism(g, h)
        assert iso.isomorphic
        assert iso.mapping is not None
        for u, v, _w in g.edges():
            assert h.has_edge(iso.mapping[u], iso.mapping[v])

    def test_different_edge_counts_are_decided_by_counts(self):
        g = _cycle(4, "a")
        h = _cycle(4, "b")
        h.add_edge("b0", "b2")
        iso = Isomorphism(g, h)
        assert not iso.isomorphic
        assert iso.decided_by == "counts"

    def test_refinement_separates_a_path_from_a_star(self):
        path = Graph()
        for n in "abcd":
            path.add_node(n)
        for u, v in [("a", "b"), ("b", "c"), ("c", "d")]:
            path.add_edge(u, v)
        star = Graph()
        star.add_node("h")
        for leaf in "xyz":
            star.add_node(leaf)
            star.add_edge("h", leaf)
        iso = Isomorphism(path, star)
        assert not iso.isomorphic
        assert iso.decided_by == "refinement"

    def test_two_triangles_versus_a_hexagon_needs_the_search(self):
        # both are 2-regular on six nodes, so refinement cannot tell them apart
        two = Graph()
        for n in "abcdef":
            two.add_node(n)
        for u, v in combinations("abc", 2):
            two.add_edge(u, v)
        for u, v in combinations("def", 2):
            two.add_edge(u, v)
        hexagon = _cycle(6, "h")
        iso = Isomorphism(two, hexagon)
        assert not iso.isomorphic
        assert iso.decided_by == "search"

    def test_a_graph_is_isomorphic_to_itself(self):
        g = _cycle(4, "a")
        assert Isomorphism(g, g).isomorphic


class TestRefusal:
    def test_mixed_directedness_is_refused(self):
        with pytest.raises(Invalid):
            Isomorphism(Graph(), Graph(directed=True))


class TestAgainstBruteForce:
    def test_the_verdict_matches_trying_every_permutation(self):
        rng = random.Random(167)
        for _ in range(30):
            a = Graph()
            b = Graph()
            for n in "abcde":
                a.add_node(n)
                b.add_node(n)
            for u, v in combinations("abcde", 2):
                if rng.random() < 0.5:
                    a.add_edge(u, v)
                if rng.random() < 0.5:
                    b.add_edge(u, v)
            assert Isomorphism(a, b).isomorphic == _brute_isomorphic(a, b)


class TestReport:
    def test_the_note_names_the_deciding_stage(self):
        note = Isomorphism(_cycle(3, "a"), _cycle(3, "b")).note()
        assert "isomorphic, decided by" in note
