from __future__ import annotations

import pytest

from mesh.errors import Invalid
from mesh.factories import cycle, path
from mesh.graph import Graph
from mesh.graphvalidate import Contract


class TestRequirements:
    def test_a_graph_that_meets_everything_has_no_breaches(self):
        c = (
            Contract()
            .directed(False)
            .connected()
            .weights_between(0, 10)
            .no_self_loops()
            .no_isolated()
            .at_most(nodes=10, edges=20)
            .whole_weights()
            .requires(["0", "1"])
        )
        assert c.check(cycle(5)) == []
        assert c.note(cycle(5)) == "meets all 8 requirement(s)"

    def test_each_breach_is_named_precisely(self):
        g = path(3)
        g.add_node("alone")
        g.add_edge("0", "1", -2.5)
        c = (
            Contract()
            .directed(True)
            .connected()
            .weights_between(0, 1)
            .no_isolated()
            .whole_weights()
            .requires(["0", "zed"])
        )
        breaches = c.check(g)
        assert "the graph is undirected" in breaches
        assert "the graph is not connected" in breaches
        assert "edge 0-1 has weight -2.5 outside 0..1" in breaches
        assert "node alone has no edge" in breaches
        assert "edge 0-1 has a fractional weight -2.5" in breaches
        assert "required node zed is missing" in breaches
        assert len(breaches) == 6

    def test_size_limits_state_the_count_and_the_limit(self):
        breaches = Contract().at_most(nodes=3, edges=2).check(cycle(5))
        assert breaches == [
            "5 node(s) exceeds the limit of 3",
            "5 edge(s) exceeds the limit of 2",
        ]

    def test_an_empty_contract_checks_nothing(self):
        assert Contract().check(Graph(directed=True)) == []
        assert Contract().note(Graph()) == "meets all 0 requirement(s)"


class TestEnforce:
    def test_enforce_raises_with_every_breach_joined(self):
        c = Contract().connected().no_isolated()
        g = Graph()
        g.add_node("a")
        g.add_node("b")
        with pytest.raises(Invalid, match="not connected; node a has no edge"):
            c.enforce(g)
        c.enforce(cycle(3))

    def test_reversed_weight_bounds_are_refused_when_stated(self):
        with pytest.raises(Invalid):
            Contract().weights_between(5, 1)


class TestReport:
    def test_the_note_counts_breaches(self):
        note = Contract().directed().connected().note(path(2))
        assert note.startswith("1 breach(es) of 2 requirement(s): the graph is undirected")
