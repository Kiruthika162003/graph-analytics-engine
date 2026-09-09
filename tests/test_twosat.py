from __future__ import annotations

import random
from itertools import product

import pytest

from mesh.errors import Invalid
from mesh.twosat import Clause, TwoSat


def _brute(clauses: list[Clause]) -> bool:
    variables = sorted({var for clause in clauses for var, _v in clause})
    for values in product((False, True), repeat=len(variables)):
        assignment = dict(zip(variables, values, strict=True))
        if all(assignment[a[0]] == a[1] or assignment[b[0]] == b[1] for a, b in clauses):
            return True
    return False


def _random_instance(rng: random.Random, variables: int, clauses: int) -> list[Clause]:
    names = [f"x{i}" for i in range(variables)]
    out: list[Clause] = []
    for _ in range(clauses):
        a, b = rng.choice(names), rng.choice(names)
        out.append(((a, rng.random() < 0.5), (b, rng.random() < 0.5)))
    return out


class TestDecisions:
    def test_a_satisfiable_instance_yields_a_checked_assignment(self):
        clauses: list[Clause] = [
            (("a", True), ("b", True)),
            (("a", False), ("c", True)),
            (("b", False), ("c", False)),
        ]
        ts = TwoSat(clauses)
        assert ts.satisfiable
        assert ts.satisfies(ts.assignment())

    def test_a_contradiction_names_the_variable(self):
        clauses: list[Clause] = [
            (("x", True), ("y", True)),
            (("x", True), ("y", False)),
            (("x", False), ("y", True)),
            (("x", False), ("y", False)),
        ]
        ts = TwoSat(clauses)
        assert not ts.satisfiable
        assert ts.contradiction in ("x", "y")
        with pytest.raises(Invalid):
            ts.assignment()

    def test_a_unit_clause_forces_its_literal(self):
        ts = TwoSat([(("p", False), ("p", False)), (("p", True), ("q", True))])
        values = ts.assignment()
        assert values["p"] is False
        assert values["q"] is True

    def test_a_tautology_constrains_nothing(self):
        ts = TwoSat([(("t", True), ("t", False))])
        assert ts.satisfiable
        assert ts.satisfies(ts.assignment())


class TestAgainstBruteForce:
    def test_the_verdict_matches_trying_every_assignment(self):
        rng = random.Random(751)
        seen_both = {True: 0, False: 0}
        for _ in range(60):
            clauses = _random_instance(rng, 5, 9)
            ts = TwoSat(clauses)
            assert ts.satisfiable == _brute(clauses)
            seen_both[ts.satisfiable] += 1
            if ts.satisfiable:
                assert ts.satisfies(ts.assignment())
        assert seen_both[True] > 0 and seen_both[False] > 0


class TestImplicationGraph:
    def test_each_clause_becomes_two_arrows(self):
        ts = TwoSat([(("a", True), ("b", True))])
        assert ts.arrows[("a", False)] == [("b", True)]
        assert ts.arrows[("b", False)] == [("a", True)]

    def test_no_clauses_is_trivially_satisfiable(self):
        ts = TwoSat([])
        assert ts.satisfiable
        assert ts.assignment() == {}


class TestReport:
    def test_the_note_lists_the_assignment_or_the_contradiction(self):
        ts = TwoSat([(("p", False), ("p", False))])
        assert ts.note() == "satisfiable over 1 variable(s): p=F"
        bad = TwoSat([(("x", True), ("x", True)), (("x", False), ("x", False))])
        assert "unsatisfiable: 'x'" in bad.note()
