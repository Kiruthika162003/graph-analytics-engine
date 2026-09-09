from __future__ import annotations

from itertools import combinations

from mesh.factories import cycle, star
from mesh.graph import Graph
from mesh.graphrecipes import Recipes


def _three_cliques() -> Graph:
    g = Graph()
    for group in ("abcd", "efgh", "ijkl"):
        for n in group:
            g.add_node(n)
        for a, b in combinations(group, 2):
            g.add_edge(a, b)
    g.add_edge("d", "e")
    g.add_edge("h", "i")
    return g


class TestCommunities:
    def test_three_cliques_come_back_as_three_groups_with_a_clear_verdict(self):
        lines = Recipes(_three_cliques(), seed=1).communities()
        assert "suggests 3 group(s)" in lines[1]
        assert lines[-1] == "verdict: clear communities"
        assert any(line.strip() == "a, b, c, d" for line in lines)

    def test_a_cycle_has_no_strong_structure(self):
        lines = Recipes(cycle(9), seed=1).communities()
        assert lines[-1] == "verdict: no strong community structure"

    def test_too_few_nodes(self):
        assert Recipes(Graph()).communities() == ["too few nodes to look for communities"]


class TestBottleneck:
    def test_the_thinnest_cut_of_three_cliques_is_a_bridge(self):
        lines = Recipes(_three_cliques()).bottleneck()
        assert "at 1" in lines[0]
        assert lines[-1] == "verdict: 1 unit(s) of capacity is all that joins them"

    def test_a_graph_in_pieces_reports_zero(self):
        g = Graph()
        for n in "abcd":
            g.add_node(n)
        g.add_edge("a", "b")
        g.add_edge("c", "d")
        lines = Recipes(g).bottleneck()
        assert lines[-1] == "verdict: the graph is already in pieces"


class TestRobustness:
    def test_a_star_halves_after_one_targeted_removal(self):
        lines = Recipes(star(10), seed=2).robustness()
        assert lines[0].startswith("halving the largest piece takes 1 targeted removal(s)")
        assert "fragile to a targeted attack" in lines[-1]

    def test_a_cycle_is_no_easier_to_break_by_targeting(self):
        lines = Recipes(cycle(8), seed=3).robustness()
        assert "no easier to break" in lines[-1]


class TestInfluence:
    def test_a_hub_tops_every_ranking(self):
        lines = Recipes(star(5)).influence()
        assert lines[-1] == "verdict: 0 tops 3 of 3 rankings"
        assert lines[0].startswith("degree: 0")

    def test_the_empty_graph(self):
        assert Recipes(Graph()).influence() == ["no nodes to rank"]


class TestDispatch:
    def test_run_names_unknown_recipes_and_refusals_in_words(self):
        r = Recipes(Graph(directed=True))
        assert r.run("magic")[0].startswith("no recipe called 'magic'")
        assert r.run("bottleneck") == [
            "the bottleneck recipe stopped: the bottleneck recipe runs on an undirected graph"
        ]
        assert Recipes(star(3)).run("influence")[-1].endswith("3 of 3 rankings")
