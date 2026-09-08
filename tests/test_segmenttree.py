from __future__ import annotations

import random

import pytest

from mesh.errors import Invalid
from mesh.segmenttree import SegmentTree


class TestRangeMax:
    def test_a_single_element_range_returns_it(self):
        t = SegmentTree(size=5)
        t.update(2, 7.0)
        assert t.range_max(2, 2) == 7.0

    def test_a_range_returns_its_maximum(self):
        t = SegmentTree(size=6)
        for i, v in enumerate([3, 1, 4, 1, 5, 9]):
            t.update(i, float(v))
        assert t.range_max(0, 3) == 4.0
        assert t.range_max(2, 5) == 9.0

    def test_unset_leaves_read_as_zero(self):
        t = SegmentTree(size=4)
        t.update(0, -3.0)
        assert t.range_max(0, 3) == 0.0

    def test_a_decrease_of_the_maximum_is_handled(self):
        t = SegmentTree(size=4)
        for i, v in enumerate([2, 8, 1, 6]):
            t.update(i, float(v))
        t.update(1, 0.0)
        assert t.overall_max() == 6.0

    def test_sizes_that_are_not_powers_of_two_work(self):
        t = SegmentTree(size=7)
        for i in range(7):
            t.update(i, float(i))
        assert t.range_max(3, 6) == 6.0
        assert t.range_max(0, 6) == 6.0


class TestRefusals:
    def test_an_out_of_range_update_is_refused(self):
        with pytest.raises(Invalid):
            SegmentTree(size=3).update(5, 1.0)

    def test_an_inverted_range_is_refused(self):
        with pytest.raises(Invalid):
            SegmentTree(size=5).range_max(4, 1)

    def test_a_range_outside_the_array_is_refused(self):
        with pytest.raises(Invalid):
            SegmentTree(size=5).range_max(0, 9)

    def test_a_zero_size_is_refused(self):
        with pytest.raises(Invalid):
            SegmentTree(size=0)


class TestAgainstBruteForce:
    def test_it_matches_a_flat_array_under_random_updates(self):
        rng = random.Random(263)
        for n in (1, 2, 5, 13, 32):
            flat = [0.0] * n
            t = SegmentTree(size=n)
            for _ in range(200):
                i = rng.randrange(n)
                v = rng.uniform(-50, 50)
                flat[i] = v
                t.update(i, v)
                lo = rng.randrange(n)
                hi = rng.randrange(lo, n)
                assert t.range_max(lo, hi) == max(flat[lo : hi + 1])


class TestReport:
    def test_the_note_states_the_root(self):
        t = SegmentTree(size=2)
        t.update(0, 3.0)
        t.update(1, 9.0)
        assert "root max 9.0" in t.note()
