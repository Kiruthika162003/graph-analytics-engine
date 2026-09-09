from __future__ import annotations

from math import inf

import pytest

from mesh.errors import Invalid, Unreachable
from mesh.factories import cycle, path
from mesh.graph import Graph
from mesh.pathexplain import PathExplanation


def _roads() -> Graph:
    g = Graph()
    for n in ("home", "mill", "bridge", "town", "ford"):
        g.add_node(n)
    for u, v, w in [
        ("home", "mill", 2), ("mill", "bridge", 3), ("bridge", "town", 1),
        ("home", "ford", 4), ("ford", "town", 4),
    ]:
        g.add_edge(u, v, w)
    return g


class TestRoute:
    def test_the_steps_sum_to_the_distance_and_run_in_order(self):
        pe = PathExplanation(_roads(), "home", "town")
        assert pe.route == ["home", "mill", "bridge", "town"]
        assert pe.distance == 6
        assert pe.steps_sum_to_distance()
        assert [r for _a, _b, _w, r in pe.steps()] == [2, 5, 6]

    def test_slack_reads_the_detour_cost_of_each_edge(self):
        pe = PathExplanation(_roads(), "home", "town")
        slack = pe.slack()
        # avoiding any edge of the mill road forces the ford road at 8, two longer
        assert all(s == 2 for s in slack.values())

    def test_a_bridge_on_the_route_has_infinite_slack(self):
        pe = PathExplanation(path(4), "0", "3")
        assert all(s == inf for s in pe.slack().values())
        assert pe.runner_up() is None
        assert "3 edge(s) with no alternative" in pe.note()

    def test_an_equal_parallel_route_gives_zero_slack(self):
        pe = PathExplanation(cycle(4), "0", "2")
        assert all(s == 0 for s in pe.slack().values())
        runner = pe.runner_up()
        assert runner is not None
        assert runner[1] == 2


class TestRunnerUp:
    def test_the_runner_up_is_never_shorter_than_the_route_and_differs_from_it(self):
        pe = PathExplanation(_roads(), "home", "town")
        route, length = pe.runner_up()
        assert length >= pe.distance
        assert route != pe.route
        assert route == ["home", "ford", "town"]
        assert "runner-up 8" in pe.note()


class TestDirected:
    def test_arcs_are_followed_one_way(self):
        g = Graph(directed=True)
        for n in "abc":
            g.add_node(n)
        g.add_edge("a", "b", 1)
        g.add_edge("b", "c", 1)
        assert PathExplanation(g, "a", "c").distance == 2
        with pytest.raises(Unreachable):
            PathExplanation(g, "c", "a")


class TestRefusal:
    def test_unknown_nodes_unreachable_sinks_and_negative_weights_are_refused(self):
        g = path(3)
        with pytest.raises(Invalid):
            PathExplanation(g, "0", "zz")
        g.add_node("alone")
        with pytest.raises(Unreachable):
            PathExplanation(g, "0", "alone")
        g.add_edge("0", "1", -1)
        with pytest.raises(Invalid):
            PathExplanation(g, "0", "2")
