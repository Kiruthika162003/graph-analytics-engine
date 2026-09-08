from __future__ import annotations

import random
from itertools import pairwise

import pytest

from mesh.errors import Invalid, Missing, Unreachable
from mesh.graph import Graph
from mesh.yen import YenKShortest


def _diamond() -> Graph:
    # s->a->t (1+1), s->b->t (2+2), s->c->t (3+3): three distinct routes
    g = Graph(directed=True)
    for n in "sabct":
        g.add_node(n)
    g.add_edge("s", "a", 1)
    g.add_edge("a", "t", 1)
    g.add_edge("s", "b", 2)
    g.add_edge("b", "t", 2)
    g.add_edge("s", "c", 3)
    g.add_edge("c", "t", 3)
    return g


def _all_simple_paths(g: Graph, u: str, t: str, seen: set[str]) -> list[list[str]]:
    if u == t:
        return [[t]]
    out: list[list[str]] = []
    for v in g.neighbors(u):
        if v not in seen:
            for rest in _all_simple_paths(g, v, t, seen | {v}):
                out.append([u, *rest])
    return out


class TestOrder:
    def test_paths_come_out_cheapest_first(self):
        y = YenKShortest(_diamond(), "s", "t", 3)
        assert y.costs() == [2, 4, 6]

    def test_the_first_path_is_the_shortest(self):
        y = YenKShortest(_diamond(), "s", "t", 1)
        assert y.paths[0][1] == ["s", "a", "t"]

    def test_every_path_is_a_real_loopless_walk(self):
        g = _diamond()
        for _c, path in YenKShortest(g, "s", "t", 3).paths:
            assert len(path) == len(set(path))
            for u, v in pairwise(path):
                assert g.has_edge(u, v)

    def test_it_stops_when_routes_run_out(self):
        y = YenKShortest(_diamond(), "s", "t", 10)
        assert y.found() == 3


class TestSpur:
    def test_a_spur_off_a_shared_prefix_is_found(self):
        # s->a->b->t is best; s->a->c->t shares the prefix s->a
        g = Graph(directed=True)
        for n in "sabct":
            g.add_node(n)
        g.add_edge("s", "a", 1)
        g.add_edge("a", "b", 1)
        g.add_edge("b", "t", 1)
        g.add_edge("a", "c", 2)
        g.add_edge("c", "t", 2)
        y = YenKShortest(g, "s", "t", 2)
        assert y.paths[1][1] == ["s", "a", "c", "t"]
        assert y.costs() == [3, 5]


class TestRefusals:
    def test_a_missing_endpoint_is_refused(self):
        with pytest.raises(Missing):
            YenKShortest(_diamond(), "s", "ghost", 1)

    def test_k_below_one_is_refused(self):
        with pytest.raises(Invalid):
            YenKShortest(_diamond(), "s", "t", 0)

    def test_an_unreachable_target_yields_no_paths(self):
        g = _diamond()
        g.add_node("island")
        y = YenKShortest(g, "s", "island", 2)
        assert y.found() == 0
        with pytest.raises(Unreachable):
            y.note()


class TestAgainstEnumeration:
    def test_the_k_cheapest_match_sorting_all_simple_paths(self):
        rng = random.Random(73)
        for _ in range(25):
            g = Graph(directed=True)
            nodes = [str(i) for i in range(6)]
            for n in nodes:
                g.add_node(n)
            for u in nodes:
                for v in nodes:
                    if u != v and rng.random() < 0.4:
                        g.add_edge(u, v, rng.randint(1, 9))
            every = _all_simple_paths(g, "0", "5", {"0"})
            expected = sorted(
                sum(g.weight(a, b) for a, b in pairwise(p)) for p in every
            )[:4]
            assert YenKShortest(g, "0", "5", 4).costs() == expected


class TestReport:
    def test_the_note_states_found_and_spread(self):
        note = YenKShortest(_diamond(), "s", "t", 3).note()
        assert "found 3 of 3" in note
        assert "4 worse" in note
