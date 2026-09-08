from __future__ import annotations

import random
from itertools import permutations

import pytest

from mesh.errors import Invalid, Missing
from mesh.mincostflow import MinCostFlow


def _two_routes() -> MinCostFlow:
    # s->a->t is cheap (cost 1+1) but narrow (cap 2); s->b->t costs 3+3, cap 5
    f = MinCostFlow()
    f.add_edge("s", "a", 2, 1)
    f.add_edge("a", "t", 2, 1)
    f.add_edge("s", "b", 5, 3)
    f.add_edge("b", "t", 5, 3)
    return f


class TestCost:
    def test_it_fills_the_cheap_route_first(self):
        f = _two_routes()
        f.send("s", "t", 2)
        assert f.total_cost == 4  # 2 units at 2 each

    def test_extra_flow_spills_onto_the_dearer_route(self):
        f = _two_routes()
        f.send("s", "t", 4)
        # 2 units at cost 2 plus 2 units at cost 6
        assert f.total_cost == 16
        assert f.sent == 4

    def test_a_request_beyond_capacity_is_refused(self):
        f = _two_routes()
        with pytest.raises(Invalid) as caught:
            f.send("s", "t", 100)
        assert "carry only 7" in str(caught.value)

    def test_reverse_edges_let_a_later_path_reroute(self):
        # the classic reroute: without cancelling, greedy would get stuck
        f = MinCostFlow()
        f.add_edge("s", "a", 1, 1)
        f.add_edge("s", "b", 1, 5)
        f.add_edge("a", "b", 1, 1)
        f.add_edge("a", "t", 1, 5)
        f.add_edge("b", "t", 1, 1)
        f.send("s", "t", 2)
        # optimum: s-a-t (6) and s-b-t (6) = 12, not s-a-b-t (3) then s-b... blocked
        assert f.total_cost == 12


class TestRefusals:
    def test_a_negative_capacity_is_refused(self):
        with pytest.raises(Invalid):
            MinCostFlow().add_edge("a", "b", -1, 1)

    def test_a_missing_endpoint_is_refused(self):
        f = _two_routes()
        with pytest.raises(Missing):
            f.send("s", "ghost", 1)


class TestAgainstAssignmentBruteForce:
    def test_unit_capacity_min_cost_flow_solves_assignment(self):
        rng = random.Random(67)
        for _ in range(25):
            n = rng.randint(1, 5)
            cost = [[rng.randint(0, 20) for _ in range(n)] for _ in range(n)]
            f = MinCostFlow()
            for w in range(n):
                f.add_edge("s", f"w{w}", 1, 0)
                f.add_edge(f"j{w}", "t", 1, 0)
                for j in range(n):
                    f.add_edge(f"w{w}", f"j{j}", 1, cost[w][j])
            f.send("s", "t", n)
            best = min(
                sum(cost[w][p[w]] for w in range(n)) for p in permutations(range(n))
            )
            assert f.total_cost == best


class TestReport:
    def test_the_note_states_the_per_unit_cost(self):
        f = _two_routes()
        f.send("s", "t", 2)
        assert "2.00 per unit" in f.note()
