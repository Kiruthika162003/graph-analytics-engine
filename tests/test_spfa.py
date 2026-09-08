from __future__ import annotations

import random

import pytest

from mesh.bellmanford import BellmanFord
from mesh.errors import Missing, Negative, Unreachable
from mesh.graph import Graph
from mesh.spfa import SPFA


class TestDistances:
    def test_it_handles_a_negative_edge(self):
        g = Graph(directed=True)
        for n in "sab":
            g.add_node(n)
        g.add_edge("s", "a", 4)
        g.add_edge("s", "b", 5)
        g.add_edge("b", "a", -3)
        assert SPFA(g, "s").distance_to("a") == 2

    def test_a_negative_cycle_is_refused(self):
        g = Graph(directed=True)
        for n in "abc":
            g.add_node(n)
        g.add_edge("a", "b", 1)
        g.add_edge("b", "c", -3)
        g.add_edge("c", "a", 1)
        with pytest.raises(Negative) as caught:
            SPFA(g, "a")
        assert "dequeued more than" in str(caught.value)

    def test_the_path_is_reconstructed(self):
        g = Graph(directed=True)
        for n in "sab":
            g.add_node(n)
        g.add_edge("s", "a", 4)
        g.add_edge("s", "b", 5)
        g.add_edge("b", "a", -3)
        assert SPFA(g, "s").path_to("a") == ["s", "b", "a"]


class TestRefusals:
    def test_a_missing_source_is_refused(self):
        g = Graph(directed=True)
        g.add_node("a")
        with pytest.raises(Missing):
            SPFA(g, "ghost")

    def test_an_unreachable_node_is_refused(self):
        g = Graph(directed=True)
        g.add_node("a")
        g.add_node("island")
        with pytest.raises(Unreachable):
            SPFA(g, "a").distance_to("island")


class TestAgainstBellmanFord:
    def test_it_agrees_with_bellman_ford_including_refusals(self):
        rng = random.Random(223)
        for _ in range(40):
            g = Graph(directed=True)
            nodes = [str(i) for i in range(8)]
            for n in nodes:
                g.add_node(n)
            for u in nodes:
                for v in nodes:
                    if u != v and rng.random() < 0.3:
                        g.add_edge(u, v, rng.randint(-2, 9))
            try:
                bf = BellmanFord(g, "0")
            except Negative:
                with pytest.raises(Negative):
                    SPFA(g, "0")
                continue
            s = SPFA(g, "0")
            for n in nodes:
                if bf.distance[n] == float("inf"):
                    with pytest.raises(Unreachable):
                        s.distance_to(n)
                else:
                    assert s.distance_to(n) == bf.distance_to(n)


class TestWork:
    def test_the_queue_does_far_less_than_the_worst_case_on_an_easy_graph(self):
        g = Graph(directed=True)
        nodes = [str(i) for i in range(20)]
        for n in nodes:
            g.add_node(n)
        for i in range(19):
            g.add_edge(nodes[i], nodes[i + 1], 1)
        s = SPFA(g, "0")
        assert s.work_ratio() < 0.2

    def test_the_note_states_the_work_ratio(self):
        g = Graph(directed=True)
        g.add_node("a")
        g.add_node("b")
        g.add_edge("a", "b", 1)
        assert "of the nodes-times-edges bound" in SPFA(g, "a").note()
