from __future__ import annotations

import random
from itertools import permutations

import pytest

from mesh.errors import Invalid
from mesh.stablematching import StableMatching


def _classic() -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    proposers = {
        "ann": ["xu", "yi", "zoe"],
        "bea": ["yi", "xu", "zoe"],
        "cat": ["xu", "yi", "zoe"],
    }
    acceptors = {
        "xu": ["bea", "ann", "cat"],
        "yi": ["ann", "bea", "cat"],
        "zoe": ["ann", "bea", "cat"],
    }
    return proposers, acceptors


def _all_stable(proposers: dict[str, list[str]], acceptors: dict[str, list[str]]) -> int:
    names = sorted(proposers)
    partners = sorted(acceptors)
    count = 0
    for perm in permutations(partners):
        match = dict(zip(names, perm, strict=True))
        sm = StableMatching.__new__(StableMatching)
        sm.proposers, sm.acceptors, sm.match = proposers, acceptors, match
        if sm.is_stable():
            count += 1
    return count


class TestStability:
    def test_the_result_has_no_blocking_pair(self):
        sm = StableMatching(*_classic())
        assert sm.is_stable()
        assert sm.blocking_pairs() == []

    def test_everyone_is_matched_exactly_once(self):
        sm = StableMatching(*_classic())
        assert sorted(sm.match) == ["ann", "bea", "cat"]
        assert sorted(sm.match.values()) == ["xu", "yi", "zoe"]

    def test_proposers_get_their_best_stable_partner(self):
        proposers, acceptors = _classic()
        sm = StableMatching(proposers, acceptors)
        # ann and bea both get their first choices; cat, ranked last by all, gets zoe
        assert sm.match == {"ann": "xu", "bea": "yi", "cat": "zoe"}
        assert sm.first_choices() == 2

    def test_swapping_sides_favors_the_other_side(self):
        proposers, acceptors = _classic()
        forward = StableMatching(proposers, acceptors)
        backward = StableMatching(acceptors, proposers)
        # xu prefers bea, and gets bea when xu's side proposes
        assert forward.match["ann"] == "xu"
        assert backward.match["xu"] == "bea"

    def test_random_instances_are_always_stable(self):
        rng = random.Random(433)
        for _ in range(30):
            n = rng.randint(1, 6)
            ps = [f"p{i}" for i in range(n)]
            qs = [f"q{i}" for i in range(n)]
            proposers = {p: rng.sample(qs, n) for p in ps}
            acceptors = {q: rng.sample(ps, n) for q in qs}
            assert StableMatching(proposers, acceptors).is_stable()

    def test_when_the_stable_matching_is_unique_both_sides_agree(self):
        rng = random.Random(439)
        for _ in range(20):
            n = 4
            ps = [f"p{i}" for i in range(n)]
            qs = [f"q{i}" for i in range(n)]
            proposers = {p: rng.sample(qs, n) for p in ps}
            acceptors = {q: rng.sample(ps, n) for q in qs}
            if _all_stable(proposers, acceptors) == 1:
                forward = StableMatching(proposers, acceptors).match
                backward = StableMatching(acceptors, proposers).match
                assert {a: p for p, a in backward.items()} == forward


class TestRefusals:
    def test_unequal_sides_are_refused(self):
        with pytest.raises(Invalid):
            StableMatching({"a": ["x"]}, {"x": ["a"], "y": ["a"]})

    def test_an_incomplete_ranking_is_refused(self):
        with pytest.raises(Invalid):
            StableMatching({"a": ["x"], "b": ["x"]}, {"x": ["a", "b"], "y": ["a", "b"]})


class TestReport:
    def test_the_note_counts_first_choices(self):
        note = StableMatching(*_classic()).note()
        assert "2 proposer(s) got their first choice" in note
