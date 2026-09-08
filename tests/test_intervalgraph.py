from __future__ import annotations

import random

import pytest

from mesh.errors import Invalid
from mesh.intervalgraph import IntervalGraph


def _meetings() -> dict[str, tuple[float, float]]:
    return {
        "standup": (9.0, 9.5),
        "design": (9.25, 10.5),
        "review": (10.0, 11.0),
        "lunch": (12.0, 13.0),
        "planning": (10.25, 10.75),
    }


class TestBuild:
    def test_overlapping_intervals_become_edges(self):
        g = IntervalGraph(_meetings()).graph
        assert g.has_edge("standup", "design")
        assert g.has_edge("design", "review")
        assert g.has_edge("design", "planning")
        assert g.has_edge("review", "planning")
        assert not g.has_edge("standup", "review")
        assert g.degree("lunch") == 0

    def test_touching_at_a_single_point_counts_as_overlap(self):
        ig = IntervalGraph({"a": (0, 1), "b": (1, 2)})
        assert ig.graph.has_edge("a", "b")
        assert ig.depth() == 2


class TestDepth:
    def test_the_deepest_overlap_is_the_room_count(self):
        ig = IntervalGraph(_meetings())
        # design, review and planning all cover 10.25 to 10.5
        assert ig.depth() == 3
        assert ig.colors_used() == 3
        assert ig.coloring_is_proper()

    def test_disjoint_intervals_need_one_color(self):
        ig = IntervalGraph({"a": (0, 1), "b": (2, 3), "c": (4, 5)})
        assert ig.depth() == 1
        assert ig.colors_used() == 1

    def test_no_intervals_means_zero_depth(self):
        ig = IntervalGraph({})
        assert ig.depth() == 0
        assert ig.colors_used() == 0
        assert ig.chordal_agrees()


class TestPerfection:
    def test_greedy_by_left_end_uses_exactly_the_depth_on_random_intervals(self):
        rng = random.Random(463)
        for _ in range(30):
            intervals = {}
            for i in range(9):
                lo = rng.randrange(20)
                intervals[f"i{i}"] = (lo, lo + rng.randrange(1, 8))
            ig = IntervalGraph(intervals)
            assert ig.coloring_is_proper()
            assert ig.colors_used() == ig.depth()
            assert ig.chordal_agrees()


class TestRefusal:
    def test_a_backwards_interval_is_refused_by_name(self):
        with pytest.raises(Invalid, match="'bad' ends at 1 before it starts at 3"):
            IntervalGraph({"bad": (3, 1)})


class TestReport:
    def test_the_note_counts_intervals_overlaps_and_depth(self):
        note = IntervalGraph(_meetings()).note()
        # the guess was five overlaps; standup meets only design, so there are four
        assert "5 intervals, 4 overlaps, depth 3 so 3 colors suffice" in note
