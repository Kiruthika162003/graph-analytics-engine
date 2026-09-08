from __future__ import annotations

from itertools import combinations

import pytest

from mesh.assortativity import Assortativity
from mesh.errors import Invalid
from mesh.graph import Graph


def _star(leaves: int) -> Graph:
    g = Graph()
    g.add_node("hub")
    for i in range(leaves):
        g.add_node(f"l{i}")
        g.add_edge("hub", f"l{i}")
    return g


def _pearson(pairs: list[tuple[int, int]]) -> float:
    n = len(pairs)
    mx = sum(a for a, _ in pairs) / n
    my = sum(b for _, b in pairs) / n
    cov = sum((a - mx) * (b - my) for a, b in pairs) / n
    vx = sum((a - mx) ** 2 for a, _ in pairs) / n
    vy = sum((b - my) ** 2 for _, b in pairs) / n
    return cov / (vx * vy) ** 0.5


class TestCoefficient:
    def test_a_star_is_perfectly_disassortative(self):
        a = Assortativity(_star(5))
        assert a.coefficient == pytest.approx(-1.0)
        assert "disassortative" in a.verdict()

    def test_a_regular_graph_has_no_coefficient(self):
        g = Graph()
        for n in "abcd":
            g.add_node(n)
        for u, v in [("a", "b"), ("b", "c"), ("c", "d"), ("d", "a")]:
            g.add_edge(u, v)
        a = Assortativity(g)
        assert a.coefficient is None
        assert "regular" in a.verdict()

    def test_two_cliques_of_different_size_joined_by_matching_degrees(self):
        # a K4 and a K3 with an edge between a K4 node and a K3 node
        g = Graph()
        big = [f"b{i}" for i in range(4)]
        small = [f"s{i}" for i in range(3)]
        for n in big + small:
            g.add_node(n)
        for x, y in combinations(big, 2):
            g.add_edge(x, y)
        for x, y in combinations(small, 2):
            g.add_edge(x, y)
        g.add_edge("b0", "s0")
        a = Assortativity(g)
        # most edges join equal degrees inside cliques: positive
        assert a.coefficient is not None
        assert a.coefficient > 0

    def test_the_coefficient_matches_a_direct_pearson_over_both_orientations(self):
        g = _star(3)
        g.add_node("x")
        g.add_edge("l0", "x")
        pairs: list[tuple[int, int]] = []
        for u, v, _w in g.edges():
            du, dv = g.degree(u), g.degree(v)
            pairs.append((du, dv))
            pairs.append((dv, du))
        assert Assortativity(g).coefficient == pytest.approx(_pearson(pairs))

    def test_a_directed_graph_counts_each_edge_once(self):
        g = Graph(directed=True)
        for n in "abc":
            g.add_node(n)
        g.add_edge("a", "b")
        g.add_edge("a", "c")
        # out-degrees a=2, b=0, c=0: both edges pair 2 with 0, zero variance on y
        assert Assortativity(g).coefficient is None


class TestRefusal:
    def test_an_edgeless_graph_is_refused(self):
        g = Graph()
        g.add_node("a")
        with pytest.raises(Invalid):
            Assortativity(g)


class TestReport:
    def test_the_note_carries_the_signed_coefficient(self):
        note = Assortativity(_star(4)).note()
        assert "assortativity -1.000" in note
        assert "the gap is structure" in note
