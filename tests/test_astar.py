from __future__ import annotations

import random
from itertools import pairwise

import pytest

from mesh.astar import AStar
from mesh.dijkstra import Dijkstra
from mesh.errors import Invalid, Missing, Unreachable
from mesh.graph import Graph


def _grid(n: int) -> tuple[Graph, dict[str, tuple[int, int]]]:
    # an n by n grid graph with unit edges; coords for a Manhattan heuristic
    g = Graph(directed=False)
    coords: dict[str, tuple[int, int]] = {}
    for r in range(n):
        for c in range(n):
            node = f"{r},{c}"
            g.add_node(node)
            coords[node] = (r, c)
    for r in range(n):
        for c in range(n):
            if c + 1 < n:
                g.add_edge(f"{r},{c}", f"{r},{c + 1}")
            if r + 1 < n:
                g.add_edge(f"{r},{c}", f"{r + 1},{c}")
    return g, coords


class TestOptimality:
    def test_a_star_matches_dijkstra_cost_with_a_manhattan_heuristic(self):
        g, coords = _grid(5)
        goal = "4,4"
        gr, gc = coords[goal]

        def h(node: str) -> float:
            r, c = coords[node]
            return abs(r - gr) + abs(c - gc)

        a = AStar(g, "0,0", goal, heuristic=h)
        d = Dijkstra(g, "0,0")
        assert a.cost() == d.distance_to(goal)

    def test_a_zero_heuristic_reduces_to_dijkstra_cost(self):
        g, _ = _grid(4)
        a = AStar(g, "0,0", "3,3")
        d = Dijkstra(g, "0,0")
        assert a.cost() == d.distance_to("3,3")

    def test_the_path_steps_are_real_edges_summing_to_the_cost(self):
        g, _coords = _grid(4)
        a = AStar(g, "0,0", "3,3")
        path = a.path()
        total = sum(g.weight(u, v) for u, v in pairwise(path))
        assert total == a.cost()


class TestSteering:
    def test_an_informed_heuristic_expands_no_more_than_zero(self):
        g, coords = _grid(6)
        goal = "5,5"
        gr, gc = coords[goal]

        def h(node: str) -> float:
            r, c = coords[node]
            return abs(r - gr) + abs(c - gc)

        informed = AStar(g, "0,0", goal, heuristic=h)
        blind = AStar(g, "0,0", goal)
        # steering can only reduce or match the nodes expanded
        assert informed.expanded <= blind.expanded


class TestRefusals:
    def test_a_missing_goal_is_refused(self):
        g, _ = _grid(3)
        with pytest.raises(Missing):
            AStar(g, "0,0", "ghost")

    def test_an_unreachable_goal_is_refused(self):
        g, _ = _grid(3)
        g.add_node("island")
        with pytest.raises(Unreachable):
            AStar(g, "0,0", "island").cost()

    def test_a_negative_edge_is_refused(self):
        g = Graph(directed=True)
        g.add_node("a")
        g.add_node("b")
        g.add_edge("a", "b", -1)
        with pytest.raises(Invalid):
            AStar(g, "a", "b")


class TestAgainstDijkstraRandom:
    def test_costs_agree_on_random_grids(self):
        rng = random.Random(5)
        for _ in range(20):
            n = rng.randint(3, 6)
            g, coords = _grid(n)
            goal = f"{n - 1},{n - 1}"
            gr, gc = coords[goal]

            def h(node: str, gr: int = gr, gc: int = gc, coords: dict = coords) -> float:
                r, c = coords[node]
                return abs(r - gr) + abs(c - gc)

            assert AStar(g, "0,0", goal, heuristic=h).cost() == \
                Dijkstra(g, "0,0").distance_to(goal)
