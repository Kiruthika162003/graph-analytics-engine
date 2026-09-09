from __future__ import annotations

import random
from collections import deque
from itertools import combinations
from math import comb

import pytest

from mesh.factories import complete, cycle, path, star
from mesh.graph import Graph
from mesh.prufer import Prufer
from mesh.stresscentrality import StressCentrality


def _counts(g: Graph, source: str) -> tuple[dict[str, int], dict[str, int]]:
    dist = {source: 0}
    sigma = {source: 1}
    queue = deque([source])
    while queue:
        v = queue.popleft()
        for w in g.neighbors(v):
            if w not in dist:
                dist[w] = dist[v] + 1
                sigma[w] = 0
                queue.append(w)
            if dist[w] == dist[v] + 1:
                sigma[w] += sigma[v]
    return dist, sigma


def _brute_stress(g: Graph) -> dict[str, float]:
    tables = {n: _counts(g, n) for n in g.nodes()}
    out = dict.fromkeys(g.nodes(), 0.0)
    for s, t in combinations(g.nodes(), 2):
        dist_s, sigma_s = tables[s]
        if t not in dist_s:
            continue
        dist_t, sigma_t = tables[t]
        for v in g.nodes():
            if v in (s, t) or v not in dist_s:
                continue
            if dist_s[v] + dist_t[v] == dist_s[t]:
                out[v] += sigma_s[v] * sigma_t[v]
    return out


def _random_graph(seed: int, n: int, p: float) -> Graph:
    rng = random.Random(seed)
    g = Graph()
    nodes = [str(i) for i in range(n)]
    for node in nodes:
        g.add_node(node)
    for a, b in combinations(nodes, 2):
        if rng.random() < p:
            g.add_edge(a, b)
    return g


def _uneven_predecessors() -> Graph:
    # t has predecessors c (two paths from s) and e2 (one path), so load splits half
    # and half where betweenness splits two thirds to one third
    g = Graph()
    for n in ("s", "a", "b", "c", "e1", "e2", "t"):
        g.add_node(n)
    for u, v in [("s", "a"), ("s", "b"), ("a", "c"), ("b", "c"), ("s", "e1"), ("e1", "e2")]:
        g.add_edge(u, v)
    g.add_edge("c", "t")
    g.add_edge("e2", "t")
    return g


class TestStress:
    def test_a_path_scores_the_product_of_the_two_sides(self):
        sc = StressCentrality(path(5))
        assert sc.stress["2"] == 2 * 2
        assert sc.stress["1"] == 1 * 3
        assert sc.stress["0"] == 0

    def test_a_star_hub_scores_the_leaf_pairs(self):
        sc = StressCentrality(star(5))
        assert sc.stress["0"] == comb(5, 2)
        assert sc.stress["1"] == 0

    def test_every_node_of_a_cycle_scores_the_same_and_a_clique_scores_nothing(self):
        cyc = StressCentrality(cycle(7)).stress
        assert len(set(cyc.values())) == 1
        assert all(s == 0 for s in StressCentrality(complete(5)).stress.values())

    def test_a_square_node_counts_one_of_the_two_paths_across(self):
        sc = StressCentrality(cycle(4))
        assert sc.stress["0"] == 1
        assert sc.betweenness["0"] == 0.5

    def test_stress_matches_the_direct_count_on_random_graphs(self):
        for seed in range(823, 833):
            g = _random_graph(seed, 8, 0.4)
            sc = StressCentrality(g)
            brute = _brute_stress(g)
            for n in g.nodes():
                assert sc.stress[n] == pytest.approx(brute[n])


class TestLoad:
    def test_load_equals_betweenness_on_random_trees(self):
        rng = random.Random(839)
        for _ in range(10):
            tree = Prufer.decode([rng.randrange(9) for _ in range(7)])
            assert StressCentrality(tree).load_matches_betweenness()

    def test_load_diverges_where_predecessors_carry_different_counts(self):
        sc = StressCentrality(_uneven_predecessors())
        assert not sc.load_matches_betweenness()
        assert sc.load["c"] < sc.betweenness["c"]
        assert sc.load["e2"] > sc.betweenness["e2"]


class TestDirected:
    def test_arcs_count_each_ordered_pair_once(self):
        g = Graph(directed=True)
        for n in "abc":
            g.add_node(n)
        g.add_edge("a", "b")
        g.add_edge("b", "c")
        sc = StressCentrality(g)
        assert sc.stress["b"] == 1
        assert sc.betweenness["b"] == 1


class TestReport:
    def test_ranking_and_note(self):
        sc = StressCentrality(path(5))
        assert sc.ranking()[0] == "2"
        assert sc.ranking("load")[0] == "2"
        with pytest.raises(KeyError):
            sc.ranking("fame")
        assert "most stressed 2 with 4 shortest path(s)" in sc.note()
        assert StressCentrality(Graph()).note() == "no nodes to score"
