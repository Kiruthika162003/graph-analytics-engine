from __future__ import annotations

import pytest

from mesh.errors import Invalid
from mesh.graph import Graph
from mesh.pagerank import PageRank


def _cycle3() -> Graph:
    g = Graph(directed=True)
    for n in "abc":
        g.add_node(n)
    g.add_edge("a", "b")
    g.add_edge("b", "c")
    g.add_edge("c", "a")
    return g


class TestDistribution:
    def test_ranks_sum_to_one(self):
        assert PageRank(_cycle3()).total() == pytest.approx(1.0)

    def test_a_symmetric_cycle_ranks_every_node_equally(self):
        pr = PageRank(_cycle3())
        for node in "abc":
            assert pr.rank[node] == pytest.approx(1 / 3)

    def test_a_node_everyone_points_to_ranks_highest(self):
        g = Graph(directed=True)
        for n in ["hub", "x", "y", "z"]:
            g.add_node(n)
        for leaf in "xyz":
            g.add_edge(leaf, "hub")
        g.add_edge("hub", "x")
        assert PageRank(g).top(1)[0][0] == "hub"

    def test_a_dangling_node_does_not_swallow_the_rank(self):
        # b has no outgoing edge; without dangling redistribution mass leaks
        g = Graph(directed=True)
        g.add_node("a")
        g.add_node("b")
        g.add_edge("a", "b")
        pr = PageRank(g)
        assert pr.total() == pytest.approx(1.0)
        assert pr.rank["b"] > pr.rank["a"]


class TestFixedPoint:
    def test_the_result_is_a_fixed_point_of_the_update(self):
        g = Graph(directed=True)
        for n in "abcd":
            g.add_node(n)
        for u, v in [("a", "b"), ("a", "c"), ("b", "c"), ("c", "a"), ("d", "c")]:
            g.add_edge(u, v)
        pr = PageRank(g, tolerance=1e-12)
        n = g.node_count()
        d = pr.damping
        out = {u: len(g.neighbors(u)) for u in g.nodes()}
        dangling = sum(pr.rank[u] for u in g.nodes() if out[u] == 0)
        for v in g.nodes():
            inflow = sum(
                d * pr.rank[u] / out[u] for u in g.nodes() if out[u] and g.has_edge(u, v)
            )
            expected = (1 - d) / n + d * dangling / n + inflow
            assert pr.rank[v] == pytest.approx(expected, abs=1e-9)


class TestRefusals:
    def test_a_damping_outside_the_unit_interval_is_refused(self):
        with pytest.raises(Invalid):
            PageRank(_cycle3(), damping=1.0)

    def test_an_empty_graph_is_refused(self):
        with pytest.raises(Invalid):
            PageRank(Graph(directed=True))


class TestReport:
    def test_the_note_reports_iterations_and_the_top_node(self):
        note = PageRank(_cycle3()).note()
        assert "converged in" in note
        assert "top '" in note
