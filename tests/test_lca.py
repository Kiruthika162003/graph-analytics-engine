from __future__ import annotations

import random

import pytest

from mesh.errors import Invalid, Missing
from mesh.graph import Graph
from mesh.lca import LowestCommonAncestor


def _tree() -> Graph:
    #        r
    #      /   \
    #     a     b
    #    / \     \
    #   c   d     e
    #  /
    # f
    g = Graph()
    for n in "rabcdef":
        g.add_node(n)
    for u, v in [("r", "a"), ("r", "b"), ("a", "c"), ("a", "d"), ("b", "e"), ("c", "f")]:
        g.add_edge(u, v)
    return g


def _naive_lca(parent: dict[str, str | None], a: str, b: str) -> str:
    ancestors = set()
    cur: str | None = a
    while cur is not None:
        ancestors.add(cur)
        cur = parent[cur]
    cur = b
    while cur not in ancestors:
        cur = parent[cur]  # type: ignore[assignment]
    return cur  # type: ignore[return-value]


class TestLca:
    def test_siblings_meet_at_their_parent(self):
        assert LowestCommonAncestor(_tree(), "r").lca("c", "d") == "a"

    def test_cousins_meet_at_the_root(self):
        assert LowestCommonAncestor(_tree(), "r").lca("f", "e") == "r"

    def test_an_ancestor_is_its_own_lca_with_a_descendant(self):
        assert LowestCommonAncestor(_tree(), "r").lca("a", "f") == "a"

    def test_a_node_with_itself(self):
        assert LowestCommonAncestor(_tree(), "r").lca("d", "d") == "d"


class TestDistance:
    def test_distance_uses_depths_and_the_ancestor(self):
        # f is at depth 3, e at depth 2, lca r at depth 0: 3 + 2 - 0 = 5
        assert LowestCommonAncestor(_tree(), "r").distance("f", "e") == 5

    def test_distance_to_self_is_zero(self):
        assert LowestCommonAncestor(_tree(), "r").distance("c", "c") == 0


class TestRefusals:
    def test_a_node_outside_the_tree_is_refused(self):
        with pytest.raises(Missing):
            LowestCommonAncestor(_tree(), "r").lca("a", "ghost")

    def test_a_graph_with_a_cycle_is_not_a_tree(self):
        g = _tree()
        g.add_edge("d", "e")  # one edge too many
        with pytest.raises(Invalid):
            LowestCommonAncestor(g, "r")

    def test_a_disconnected_graph_is_not_a_tree(self):
        # a path a-b-c plus a stranded d-e: right edge count, wrong shape
        g = Graph()
        for n in "abcde":
            g.add_node(n)
        g.add_edge("a", "b")
        g.add_edge("b", "c")
        g.add_edge("d", "e")
        g.add_edge("a", "c")  # four edges over five nodes, but two pieces
        with pytest.raises(Invalid):
            LowestCommonAncestor(g, "a")

    def test_a_directed_graph_is_refused(self):
        with pytest.raises(Invalid):
            LowestCommonAncestor(Graph(directed=True), "r")


class TestAgainstNaiveWalk:
    def test_binary_lifting_matches_walking_up_on_random_trees(self):
        rng = random.Random(71)
        for _ in range(25):
            n = rng.randint(2, 30)
            g = Graph()
            nodes = [str(i) for i in range(n)]
            parent: dict[str, str | None] = {"0": None}
            g.add_node("0")
            for i in range(1, n):
                p = str(rng.randrange(i))  # attach to an earlier node
                g.add_node(nodes[i])
                g.add_edge(p, nodes[i])
                parent[nodes[i]] = p
            lca = LowestCommonAncestor(g, "0")
            for _ in range(20):
                a, b = rng.choice(nodes), rng.choice(nodes)
                assert lca.lca(a, b) == _naive_lca(parent, a, b)


class TestReport:
    def test_the_note_states_the_level_count(self):
        assert "level(s)" in LowestCommonAncestor(_tree(), "r").note()
