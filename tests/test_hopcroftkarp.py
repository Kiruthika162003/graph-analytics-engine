from __future__ import annotations

import random
from itertools import combinations

import pytest

from mesh.errors import Invalid
from mesh.hopcroftkarp import HopcroftKarp


def _brute_max_matching(edges: list[tuple[str, str]]) -> int:
    best = 0
    for k in range(len(edges), 0, -1):
        for subset in combinations(edges, k):
            lefts = [u for u, _v in subset]
            rights = [v for _u, v in subset]
            if len(set(lefts)) == k and len(set(rights)) == k:
                return k
    return best


class TestMatching:
    def test_a_perfect_pairing_matches_every_node(self):
        hk = HopcroftKarp(["w1", "w2", "w3"], ["s1", "s2", "s3"])
        for u, v in [("w1", "s1"), ("w2", "s2"), ("w3", "s3"), ("w1", "s2")]:
            hk.add_edge(u, v)
        assert hk.solve() == 3

    def test_a_contested_partner_limits_the_matching(self):
        # both workers want only s1, so one is left out
        hk = HopcroftKarp(["w1", "w2"], ["s1", "s2"])
        hk.add_edge("w1", "s1")
        hk.add_edge("w2", "s1")
        assert hk.solve() == 1

    def test_matched_pairs_use_real_edges_and_share_no_endpoint(self):
        hk = HopcroftKarp(["a", "b", "c"], ["x", "y", "z"])
        edges = [("a", "x"), ("a", "y"), ("b", "y"), ("c", "z"), ("c", "y")]
        for u, v in edges:
            hk.add_edge(u, v)
        hk.solve()
        m = hk.matching()
        for u, v in m.items():
            assert (u, v) in edges
        assert len(set(m.values())) == len(m)

    def test_an_augmenting_path_reroutes_an_early_choice(self):
        # a greedy a->x would block b, whose only option is x; the path fixes it
        hk = HopcroftKarp(["a", "b"], ["x", "y"])
        hk.add_edge("a", "x")
        hk.add_edge("a", "y")
        hk.add_edge("b", "x")
        assert hk.solve() == 2


class TestRefusals:
    def test_an_edge_not_crossing_the_sides_is_refused(self):
        hk = HopcroftKarp(["a"], ["x"])
        with pytest.raises(Invalid):
            hk.add_edge("a", "a")

    def test_overlapping_sides_are_refused(self):
        with pytest.raises(Invalid):
            HopcroftKarp(["a", "b"], ["b", "c"])


class TestAgainstBruteForce:
    def test_the_size_matches_exhaustive_search(self):
        rng = random.Random(77)
        for _ in range(30):
            left = [f"l{i}" for i in range(4)]
            right = [f"r{i}" for i in range(4)]
            edges = [(u, v) for u in left for v in right if rng.random() < 0.4]
            hk = HopcroftKarp(left, right)
            for u, v in edges:
                hk.add_edge(u, v)
            assert hk.solve() == _brute_max_matching(edges)


class TestReport:
    def test_the_note_reports_size_against_the_smaller_side(self):
        hk = HopcroftKarp(["a", "b"], ["x", "y", "z"])
        hk.add_edge("a", "x")
        hk.add_edge("b", "y")
        hk.solve()
        assert "matched 2 of the smaller side's 2" in hk.note()
