from __future__ import annotations

import random

import pytest

from mesh.errors import Invalid, Missing
from mesh.graph import Graph
from mesh.hld import HeavyLight


def _random_tree(rng: random.Random, n: int) -> tuple[Graph, dict[str, float]]:
    g = Graph()
    g.add_node("0")
    for i in range(1, n):
        g.add_node(str(i))
        g.add_edge(str(rng.randrange(i)), str(i))
    values = {n: float(rng.randint(0, 100)) for n in g.nodes()}
    return g, values


def _walk_max(
    parent: dict[str, str | None],
    depth: dict[str, int],
    values: dict[str, float],
    a: str,
    b: str,
) -> float:
    best = float("-inf")
    while a != b:
        if depth[a] < depth[b]:
            a, b = b, a
        best = max(best, values[a])
        a = parent[a]  # type: ignore[assignment]
    return max(best, values[a])


def _path(k: int) -> tuple[Graph, dict[str, float]]:
    g = Graph()
    nodes = [str(i) for i in range(k)]
    for n in nodes:
        g.add_node(n)
    for i in range(k - 1):
        g.add_edge(nodes[i], nodes[i + 1])
    return g, {n: float(i) for i, n in enumerate(nodes)}


class TestQueries:
    def test_a_path_is_a_single_chain(self):
        g, values = _path(6)
        hld = HeavyLight(g, "0", values)
        assert hld.chain_count() == 1
        assert hld.path_max("1", "4") == 4.0

    def test_a_star_has_one_chain_per_leaf_but_one(self):
        g = Graph()
        g.add_node("h")
        for leaf in "abc":
            g.add_node(leaf)
            g.add_edge("h", leaf)
        hld = HeavyLight(g, "h", {"h": 1.0, "a": 5.0, "b": 2.0, "c": 9.0})
        assert hld.chain_count() == 3
        assert hld.path_max("a", "c") == 9.0

    def test_the_path_to_itself_is_the_node_value(self):
        g, values = _path(4)
        assert HeavyLight(g, "0", values).path_max("2", "2") == 2.0

    def test_an_update_changes_later_queries(self):
        g, values = _path(5)
        hld = HeavyLight(g, "0", values)
        hld.update("2", 50.0)
        assert hld.path_max("0", "4") == 50.0


class TestRefusals:
    def test_a_directed_graph_is_refused(self):
        with pytest.raises(Invalid):
            HeavyLight(Graph(directed=True), "r", {})

    def test_a_graph_with_a_cycle_is_refused(self):
        g, values = _path(3)
        g.add_edge("0", "2")
        with pytest.raises(Invalid):
            HeavyLight(g, "0", values)

    def test_a_missing_root_is_refused(self):
        g, values = _path(3)
        with pytest.raises(Missing):
            HeavyLight(g, "ghost", values)

    def test_a_missing_query_node_is_refused(self):
        g, values = _path(3)
        with pytest.raises(Missing):
            HeavyLight(g, "0", values).path_max("0", "ghost")


class TestAgainstWalking:
    def test_path_max_matches_walking_the_path_on_random_trees(self):
        rng = random.Random(227)
        for _ in range(25):
            n = rng.randint(2, 40)
            g, values = _random_tree(rng, n)
            hld = HeavyLight(g, "0", values)
            for _ in range(30):
                a, b = rng.choice(g.nodes()), rng.choice(g.nodes())
                expected = _walk_max(hld.parent, hld.depth, values, a, b)
                assert hld.path_max(a, b) == expected

    def test_chains_crossed_stay_logarithmic(self):
        rng = random.Random(229)
        g, values = _random_tree(rng, 200)
        hld = HeavyLight(g, "0", values)
        for _ in range(200):
            a, b = rng.choice(g.nodes()), rng.choice(g.nodes())
            hld.path_max(a, b)
        # at most two log2(200) chains, one climb per endpoint
        assert hld.max_chains_crossed <= 2 * 8


class TestReport:
    def test_the_note_states_chains_and_crossings(self):
        g, values = _path(5)
        hld = HeavyLight(g, "0", values)
        hld.path_max("0", "4")
        assert "1 chain(s)" in hld.note()
