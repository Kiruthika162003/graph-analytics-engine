from __future__ import annotations

import random

import pytest

from mesh.bellmanford import BellmanFord
from mesh.dagpaths import DagPaths
from mesh.errors import Cyclic, Missing, Unreachable
from mesh.graph import Graph


def _project() -> Graph:
    # start -> design(3) -> build(5) -> test(2) -> ship; start -> docs(1) -> ship
    g = Graph(directed=True)
    for n in ["start", "design", "build", "test", "docs", "ship"]:
        g.add_node(n)
    for u, v, w in [
        ("start", "design", 3), ("design", "build", 5), ("build", "test", 2),
        ("test", "ship", 0), ("start", "docs", 1), ("docs", "ship", 0),
    ]:
        g.add_edge(u, v, w)
    return g


class TestShortest:
    def test_shortest_takes_the_cheap_branch(self):
        assert DagPaths(_project(), "start").shortest_to("ship") == 1

    def test_shortest_handles_a_negative_edge(self):
        g = Graph(directed=True)
        for n in "abc":
            g.add_node(n)
        g.add_edge("a", "b", 5)
        g.add_edge("b", "c", -3)
        g.add_edge("a", "c", 4)
        assert DagPaths(g, "a").shortest_to("c") == 2

    def test_the_shortest_path_is_reconstructed(self):
        assert DagPaths(_project(), "start").shortest_path_to("ship") == [
            "start", "docs", "ship"
        ]


class TestLongest:
    def test_longest_takes_the_expensive_branch(self):
        assert DagPaths(_project(), "start").longest_to("ship") == 10

    def test_the_critical_path_lists_the_slackless_tasks(self):
        path = DagPaths(_project(), "start").critical_path()
        assert path == ["start", "design", "build", "test", "ship"]


class TestRefusals:
    def test_a_cyclic_graph_is_refused(self):
        g = Graph(directed=True)
        g.add_node("a")
        g.add_node("b")
        g.add_edge("a", "b")
        g.add_edge("b", "a")
        with pytest.raises(Cyclic):
            DagPaths(g, "a")

    def test_a_missing_source_is_refused(self):
        with pytest.raises(Missing):
            DagPaths(_project(), "ghost")

    def test_an_unreachable_node_is_refused(self):
        g = _project()
        g.add_node("island")
        with pytest.raises(Unreachable):
            DagPaths(g, "start").shortest_to("island")


class TestAgainstBellmanFord:
    def test_shortest_agrees_with_bellman_ford_on_random_dags(self):
        rng = random.Random(31)
        for _ in range(30):
            g = Graph(directed=True)
            nodes = [str(i) for i in range(8)]
            for n in nodes:
                g.add_node(n)
            # edges only from lower to higher index guarantee acyclicity
            for i, u in enumerate(nodes):
                for v in nodes[i + 1 :]:
                    if rng.random() < 0.4:
                        g.add_edge(u, v, rng.randint(-3, 9))
            dag = DagPaths(g, "0")
            bf = BellmanFord(g, "0")
            for n in nodes:
                reachable = bf.distance[n] != float("inf")
                if reachable:
                    assert dag.shortest_to(n) == bf.distance_to(n)
                else:
                    with pytest.raises(Unreachable):
                        dag.shortest_to(n)


class TestReport:
    def test_the_note_states_the_critical_length(self):
        assert "critical path of length 10" in DagPaths(_project(), "start").note()
