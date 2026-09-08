from __future__ import annotations

import random
from collections import deque

import pytest

from mesh.betweenness import Betweenness
from mesh.errors import Invalid
from mesh.graph import Graph


def _star() -> Graph:
    # a hub with four leaves: every leaf pair routes through the hub
    g = Graph()
    g.add_node("hub")
    for leaf in "abcd":
        g.add_node(leaf)
        g.add_edge("hub", leaf)
    return g


def _brute_betweenness(g: Graph) -> dict[str, float]:
    # enumerate all shortest paths between all pairs by BFS layers
    nodes = g.nodes()
    score = dict.fromkeys(nodes, 0.0)
    for s in nodes:
        for t in nodes:
            if s == t:
                continue
            paths = _all_shortest_paths(g, s, t)
            if not paths:
                continue
            for node in nodes:
                if node in (s, t):
                    continue
                through = sum(1 for p in paths if node in p)
                score[node] += through / len(paths)
    if not g.directed:
        for node in score:
            score[node] /= 2
    return score


def _all_shortest_paths(g: Graph, s: str, t: str) -> list[list[str]]:
    dist = {s: 0}
    queue = deque([s])
    while queue:
        v = queue.popleft()
        for w in g.neighbors(v):
            if w not in dist:
                dist[w] = dist[v] + 1
                queue.append(w)
    if t not in dist:
        return []
    paths: list[list[str]] = []

    def walk(node: str, path: list[str]) -> None:
        if node == t:
            paths.append(path)
            return
        for w in g.neighbors(node):
            if dist.get(w) == dist[node] + 1:
                walk(w, [*path, w])

    walk(s, [s])
    return paths


class TestScores:
    def test_the_hub_of_a_star_carries_every_leaf_pair(self):
        b = Betweenness(_star())
        # four leaves make six pairs, all through the hub
        assert b.score["hub"] == 6.0

    def test_leaves_have_zero_betweenness(self):
        b = Betweenness(_star())
        for leaf in "abcd":
            assert b.score[leaf] == 0.0

    def test_the_middle_of_a_path_is_the_broker(self):
        g = Graph()
        for n in "abc":
            g.add_node(n)
        g.add_edge("a", "b")
        g.add_edge("b", "c")
        assert Betweenness(g).top(1)[0][0] == "b"

    def test_normalized_hub_score_is_the_maximum(self):
        b = Betweenness(_star())
        assert b.normalized()["hub"] == pytest.approx(1.0)


class TestRefusal:
    def test_fewer_than_three_nodes_is_refused(self):
        g = Graph()
        g.add_node("a")
        g.add_node("b")
        with pytest.raises(Invalid):
            Betweenness(g)


class TestAgainstBruteForce:
    def test_brandes_matches_path_enumeration_on_random_graphs(self):
        rng = random.Random(13)
        for _ in range(25):
            g = Graph()
            nodes = [str(i) for i in range(7)]
            for n in nodes:
                g.add_node(n)
            for u in nodes:
                for v in nodes:
                    if u < v and rng.random() < 0.35:
                        g.add_edge(u, v)
            fast = Betweenness(g).score
            slow = _brute_betweenness(g)
            for node in nodes:
                assert fast[node] == pytest.approx(slow[node])


class TestReport:
    def test_the_note_names_the_top_broker(self):
        assert "'hub' at 100%" in Betweenness(_star()).note()
