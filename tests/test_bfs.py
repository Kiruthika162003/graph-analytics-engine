from __future__ import annotations

from itertools import pairwise

import pytest

from mesh.bfs import BFS
from mesh.errors import Missing, Unreachable
from mesh.graph import Graph


def _line() -> Graph:
    # a - b - c - d
    g = Graph()
    for n in "abcd":
        g.add_node(n)
    g.add_edge("a", "b")
    g.add_edge("b", "c")
    g.add_edge("c", "d")
    return g


class TestDistance:
    def test_the_source_is_at_distance_zero(self):
        assert BFS(_line(), "a").distance_to("a") == 0

    def test_distance_counts_edges_on_a_line(self):
        b = BFS(_line(), "a")
        assert b.distance_to("d") == 3

    def test_bfs_finds_the_fewest_edge_path_not_just_a_path(self):
        # a diamond: a-b, a-c, b-d, c-d, plus a long detour a-e-f-d
        g = Graph()
        for n in "abcdef":
            g.add_node(n)
        for u, v in [("a", "b"), ("a", "c"), ("b", "d"), ("c", "d"),
                     ("a", "e"), ("e", "f"), ("f", "d")]:
            g.add_edge(u, v)
        assert BFS(g, "a").distance_to("d") == 2  # not 3 via the detour


class TestPath:
    def test_the_path_starts_at_the_source_and_ends_at_the_target(self):
        path = BFS(_line(), "a").path_to("d")
        assert path[0] == "a"
        assert path[-1] == "d"

    def test_the_path_length_matches_the_distance(self):
        b = BFS(_line(), "a")
        assert len(b.path_to("d")) == b.distance_to("d") + 1

    def test_every_step_on_the_path_is_a_real_edge(self):
        g = _line()
        path = BFS(g, "a").path_to("d")
        for u, v in pairwise(path):
            assert g.has_edge(u, v)


class TestReachability:
    def test_a_disconnected_node_is_not_reached(self):
        g = _line()
        g.add_node("island")
        b = BFS(g, "a")
        assert not b.reached("island")

    def test_distance_to_an_unreached_node_is_refused(self):
        g = _line()
        g.add_node("island")
        with pytest.raises(Unreachable):
            BFS(g, "a").distance_to("island")

    def test_the_reachable_set_is_the_source_component(self):
        assert BFS(_line(), "a").reachable_nodes() == {"a", "b", "c", "d"}


class TestRefusalAndReport:
    def test_a_missing_source_is_refused(self):
        with pytest.raises(Missing):
            BFS(_line(), "ghost")

    def test_the_eccentricity_is_the_farthest_distance(self):
        assert BFS(_line(), "a").eccentricity() == 3

    def test_the_note_states_the_reached_count(self):
        assert "reached 4 node(s)" in BFS(_line(), "a").note()
