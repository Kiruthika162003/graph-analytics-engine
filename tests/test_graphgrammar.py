from __future__ import annotations

import pytest

from mesh.errors import Invalid
from mesh.factories import cycle, path, star
from mesh.graph import Graph
from mesh.graphgrammar import Rewriter, Rule


def _graph(nodes: str, edges: list[tuple[str, str]], directed: bool = False) -> Graph:
    g = Graph(directed=directed)
    for n in nodes.split():
        g.add_node(n)
    for u, v in edges:
        g.add_edge(u, v)
    return g


def contract_rule() -> Rule:
    # a path a-b-c becomes the edge a-c; b is dropped
    return Rule(_graph("a b c", [("a", "b"), ("b", "c")]), _graph("a c", [("a", "c")]))


def subdivide_rule() -> Rule:
    # an edge a-b becomes a-m-b through a fresh node m
    return Rule(_graph("a b", [("a", "b")]), _graph("a m b", [("a", "m"), ("m", "b")]))


class TestApplication:
    def test_contracting_shrinks_a_path_to_an_edge_in_n_minus_two_steps(self):
        rw = Rewriter(path(6), contract_rule())
        steps = rw.apply_until_fixed()
        assert steps == 4
        assert rw.host.node_count() == 2
        assert rw.host.edge_count() == 1

    def test_subdividing_an_edge_creates_a_fresh_node(self):
        rw = Rewriter(path(2), subdivide_rule())
        assert rw.apply_once()
        assert rw.host.node_count() == 3
        assert "m#1" in rw.host.nodes()
        assert rw.host.has_edge("0", "m#1")
        assert rw.host.has_edge("m#1", "1")
        assert not rw.host.has_edge("0", "1")

    def test_a_rule_that_never_matches_leaves_the_host_alone(self):
        rw = Rewriter(star(3), Rule(cycle(3), _graph("0", [])))
        assert not rw.apply_once()
        assert rw.applications == 0
        assert rw.host.edge_count() == 3

    def test_edges_to_the_rest_of_the_host_survive_at_kept_nodes(self):
        g = path(3)
        g.add_node("tail")
        g.add_edge("2", "tail")
        rw = Rewriter(g, contract_rule())
        assert rw.apply_once()
        assert rw.host.node_count() == 3
        assert rw.host.edge_count() == 2
        assert rw.host.degree("tail") == 1

    def test_fresh_names_never_collide_across_applications(self):
        rw = Rewriter(path(3), subdivide_rule())
        rw.apply_once()
        rw.apply_once()
        names = rw.host.nodes()
        assert "m#1" in names and "m#2" in names
        assert len(set(names)) == len(names)


class TestDirected:
    def test_a_directed_rule_reverses_an_arc(self):
        rule = Rule(
            _graph("a b", [("a", "b")], directed=True),
            _graph("a b", [("b", "a")], directed=True),
        )
        host = _graph("x y", [("x", "y")], directed=True)
        rw = Rewriter(host, rule)
        assert rw.apply_once()
        assert rw.host.has_edge("y", "x")
        assert not rw.host.has_edge("x", "y")


class TestRefusal:
    def test_mismatched_directions_and_empty_left_sides_are_refused(self):
        with pytest.raises(Invalid):
            Rule(Graph(directed=True), Graph())
        with pytest.raises(Invalid):
            Rule(Graph(), Graph())
        with pytest.raises(Invalid):
            Rewriter(Graph(directed=True), contract_rule())


class TestReport:
    def test_the_note_counts_applications_and_the_host_size(self):
        rw = Rewriter(path(4), contract_rule())
        rw.apply_until_fixed()
        assert rw.note() == "2 application(s); host now 2 node(s) and 1 edge(s)"
