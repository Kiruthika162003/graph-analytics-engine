from __future__ import annotations

import random
from itertools import combinations

import pytest

from mesh.cheeger import Cheeger
from mesh.errors import Invalid
from mesh.factories import complete, cycle, path, star
from mesh.graph import Graph


def _two_cliques() -> Graph:
    g = Graph()
    for group in ("abcd", "wxyz"):
        for n in group:
            g.add_node(n)
        for a, b in combinations(group, 2):
            g.add_edge(a, b)
    g.add_edge("d", "w")
    return g


def _random_connected(seed: int, n: int, p: float) -> Graph:
    rng = random.Random(seed)
    g = path(n)
    for a, b in combinations(g.nodes(), 2):
        if not g.has_edge(a, b) and rng.random() < p:
            g.add_edge(a, b)
    return g


class TestExact:
    def test_two_cliques_cut_at_the_bridge(self):
        ch = Cheeger(_two_cliques())
        side, h = ch.exact()
        assert h == pytest.approx(1 / 13)
        assert side in ({*"abcd"}, {*"wxyz"})

    def test_an_even_cycle_halves(self):
        _side, h = Cheeger(cycle(8)).exact()
        assert h == pytest.approx(2 / 8)

    def test_a_complete_graph_splits_as_evenly_as_it_can(self):
        # K4: two and two, four crossing edges over a volume of six
        _side, h = Cheeger(complete(4)).exact()
        assert h == pytest.approx(4 / 6)

    def test_a_star_cuts_off_one_leaf(self):
        side, h = Cheeger(star(4)).exact()
        assert h == pytest.approx(1.0)
        assert len(side) in (1, 4)


class TestSweep:
    def test_the_sweep_finds_the_bridge_between_two_cliques(self):
        ch = Cheeger(_two_cliques())
        side, phi = ch.sweep_cut()
        assert phi == pytest.approx(1 / 13)
        assert side in ({*"abcd"}, {*"wxyz"})

    def test_the_sweep_never_beats_the_exact_constant(self):
        for seed in range(719, 729):
            ch = Cheeger(_random_connected(seed, 8, 0.3))
            assert ch.sweep_cut()[1] >= ch.exact()[1] - 1e-9


class TestBounds:
    def test_cheeger_inequality_holds_on_random_connected_graphs(self):
        for seed in range(733, 745):
            assert Cheeger(_random_connected(seed, 8, 0.3)).bounds_hold()

    def test_a_disconnected_graph_has_zero_everywhere(self):
        g = Graph()
        for n in "abcd":
            g.add_node(n)
        g.add_edge("a", "b")
        g.add_edge("c", "d")
        ch = Cheeger(g)
        assert ch.lambda2 == pytest.approx(0.0, abs=1e-9)
        assert ch.exact()[1] == 0.0


class TestRefusal:
    def test_isolated_nodes_directed_graphs_and_improper_sides_are_refused(self):
        g = path(3)
        g.add_node("alone")
        with pytest.raises(Invalid):
            Cheeger(g)
        with pytest.raises(Invalid):
            Cheeger(Graph(directed=True))
        with pytest.raises(Invalid):
            Cheeger(cycle(4)).conductance(set())
        with pytest.raises(Invalid):
            Cheeger(path(16)).exact()


class TestReport:
    def test_the_note_gives_the_constant_the_sweep_and_the_bracket(self):
        note = Cheeger(_two_cliques()).note()
        assert "Cheeger constant 0.0769 on a side of 4, sweep cut 0.0769" in note
        assert "bracketed by" in note
