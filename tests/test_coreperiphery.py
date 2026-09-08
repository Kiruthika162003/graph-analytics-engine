from __future__ import annotations

from itertools import combinations

import pytest

from mesh.coreperiphery import CorePeriphery
from mesh.errors import Invalid
from mesh.graph import Graph


def _hub_and_spokes() -> Graph:
    # a K3 core, each core node with three private spokes
    g = Graph()
    core = ["c0", "c1", "c2"]
    for c in core:
        g.add_node(c)
    for a, b in combinations(core, 2):
        g.add_edge(a, b)
    for c in core:
        for i in range(3):
            leaf = f"{c}s{i}"
            g.add_node(leaf)
            g.add_edge(c, leaf)
    return g


def _three_cliques() -> Graph:
    g = Graph()
    for c in "abc":
        members = [f"{c}{i}" for i in range(4)]
        for m in members:
            g.add_node(m)
        for x, y in combinations(members, 2):
            g.add_edge(x, y)
    return g


class TestFit:
    def test_the_core_is_recovered_on_a_hub_and_spoke_graph(self):
        cp = CorePeriphery(_hub_and_spokes())
        assert cp.core == {"c0", "c1", "c2"}
        assert cp.fits()

    def test_the_periphery_is_everything_else(self):
        cp = CorePeriphery(_hub_and_spokes())
        assert cp.periphery() == set(_hub_and_spokes().nodes()) - cp.core

    def test_a_perfect_ideal_pattern_correlates_at_one(self):
        # core K2 with spokes on both, no spoke-spoke edges, and the two
        # core nodes each linked to every spoke: exactly the ideal block form
        g = Graph()
        for n in ["c0", "c1", "p0", "p1", "p2"]:
            g.add_node(n)
        g.add_edge("c0", "c1")
        for p in ["p0", "p1", "p2"]:
            g.add_edge("c0", p)
            g.add_edge("c1", p)
        cp = CorePeriphery(g)
        assert cp.core == {"c0", "c1"}
        assert cp.correlation == pytest.approx(1.0)

    def test_community_structure_does_not_fit_the_model(self):
        cp = CorePeriphery(_three_cliques())
        assert not cp.fits()
        assert "does not describe" in cp.note()

    def test_the_correlation_never_exceeds_one(self):
        cp = CorePeriphery(_hub_and_spokes())
        assert -1.0 <= cp.correlation <= 1.0


class TestRefusals:
    def test_a_directed_graph_is_refused(self):
        with pytest.raises(Invalid):
            CorePeriphery(Graph(directed=True))

    def test_fewer_than_three_nodes_is_refused(self):
        g = Graph()
        g.add_node("a")
        g.add_node("b")
        with pytest.raises(Invalid):
            CorePeriphery(g)


class TestReport:
    def test_the_note_states_core_size_and_correlation(self):
        note = CorePeriphery(_hub_and_spokes()).note()
        assert "core of 3" in note
        assert "hub-and-spoke" in note
