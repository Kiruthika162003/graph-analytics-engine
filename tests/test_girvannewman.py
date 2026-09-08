from __future__ import annotations

from itertools import combinations

import pytest

from mesh.errors import Invalid
from mesh.girvannewman import GirvanNewman
from mesh.graph import Graph


def _two_cliques_bridged() -> Graph:
    g = Graph()
    left = [f"l{i}" for i in range(4)]
    right = [f"r{i}" for i in range(4)]
    for n in left + right:
        g.add_node(n)
    for a, b in combinations(left, 2):
        g.add_edge(a, b)
    for a, b in combinations(right, 2):
        g.add_edge(a, b)
    g.add_edge("l0", "r0")
    return g


class TestRemoval:
    def test_the_bridge_is_removed_first(self):
        gn = GirvanNewman(_two_cliques_bridged(), max_removals=1)
        assert gn.removed == [("l0", "r0")]

    def test_the_best_partition_is_the_two_cliques(self):
        gn = GirvanNewman(_two_cliques_bridged())
        assert gn.best_at == 1
        assert {f"l{i}" for i in range(4)} in gn.best_partition
        assert len(gn.best_partition) == 2

    def test_modularity_peaks_then_falls_as_cliques_are_shredded(self):
        gn = GirvanNewman(_two_cliques_bridged())
        peak = max(q for _r, q in gn.history)
        assert gn.history[-1][1] < peak
        assert gn.best_modularity == pytest.approx(peak)

    def test_a_cap_limits_the_removals(self):
        gn = GirvanNewman(_two_cliques_bridged(), max_removals=3)
        assert len(gn.removed) == 3

    def test_a_three_clique_chain_splits_into_three(self):
        g = Graph()
        groups = [[f"{c}{i}" for i in range(4)] for c in "abc"]
        for group in groups:
            for n in group:
                g.add_node(n)
            for x, y in combinations(group, 2):
                g.add_edge(x, y)
        g.add_edge("a0", "b0")
        g.add_edge("b1", "c0")
        gn = GirvanNewman(g)
        assert len(gn.best_partition) == 3
        assert gn.best_at == 2


class TestEdgeBetweenness:
    def test_the_bridge_carries_every_cross_pair(self):
        g = _two_cliques_bridged()
        scores = GirvanNewman(g, max_removals=0)._edge_betweenness(g)
        bridge = scores[frozenset(("l0", "r0"))]
        # 16 cross pairs, each counted from both ends
        assert bridge == pytest.approx(32.0)
        inside = scores[frozenset(("l1", "l2"))]
        assert inside < bridge


class TestRefusals:
    def test_a_directed_graph_is_refused(self):
        with pytest.raises(Invalid):
            GirvanNewman(Graph(directed=True))

    def test_an_edgeless_graph_is_refused(self):
        g = Graph()
        g.add_node("a")
        with pytest.raises(Invalid):
            GirvanNewman(g)


class TestReport:
    def test_the_note_states_the_peak_and_when(self):
        note = GirvanNewman(_two_cliques_bridged()).note()
        assert "2 communit(ies) after 1 removal(s)" in note
