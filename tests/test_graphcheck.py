from __future__ import annotations

import random
from itertools import combinations

from mesh.factories import complete, cycle, path, star
from mesh.graph import Graph
from mesh.graphcheck import SelfCheck


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


class TestShapes:
    def test_every_identity_holds_on_the_standard_shapes(self):
        for g in (path(6), star(5), cycle(7), complete(5)):
            sc = SelfCheck(g)
            assert sc.all_held()
            assert sc.counts()["failed"] == 0

    def test_a_tree_runs_the_wiener_check_and_a_cycle_skips_it(self):
        assert ("tree wiener", "held", "index 35") in SelfCheck(path(6)).results
        skipped = [v for n, v, _r in SelfCheck(cycle(5)).results if n == "tree wiener"]
        assert skipped == ["skipped"]

    def test_random_graphs_hold_every_applicable_identity(self):
        for seed in range(1051, 1057):
            sc = SelfCheck(_random_graph(seed, 9, 0.4))
            assert sc.all_held()
            assert sc.counts()["held"] >= 4


class TestSkips:
    def test_a_directed_graph_skips_the_undirected_identities_and_fails_nothing(self):
        g = Graph(directed=True)
        for n in "abc":
            g.add_node(n)
        g.add_edge("a", "b")
        g.add_edge("b", "c")
        sc = SelfCheck(g)
        assert sc.all_held()
        assert sc.counts()["skipped"] == 4
        assert ("handshake", "held", "degree sum 2 against 2") in sc.results

    def test_an_empty_graph_runs_what_makes_sense(self):
        sc = SelfCheck(Graph())
        assert sc.all_held()
        assert sc.counts()["skipped"] == 3
        assert sc.note() == "3 held, 3 skipped, 0 failed"


class TestReport:
    def test_lines_end_with_the_tally(self):
        lines = SelfCheck(star(3)).lines()
        assert lines[0].startswith("handshake: held (degree sum 6 against 6)")
        assert lines[-1] == "6 held, 0 skipped, 0 failed"
        assert "summary counts: held" in lines[-2]
