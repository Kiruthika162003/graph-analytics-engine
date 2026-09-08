"""Stable matching: pair two sides by their preferences so nobody wants to elope.

Maximum matching counts pairs; stable matching respects wishes. Each
side ranks the other, and a matching is stable when no two unmatched-to-
each-other participants both prefer each other to their assigned
partners, since such a pair, a blocking pair, would leave their
assignments and pair up. Residents to hospitals, students to schools,
and the original marriage framing all ask for a stable matching, and
Gale and Shapley showed one always exists and gave the procedure that
finds it. One side proposes, in order of preference, to the best
partner who has not yet rejected them; the other side holds the best
offer received so far and rejects the rest, releasing any previously
held proposer. When everyone is held the result is stable, because a
proposer who preferred someone else to their partner was rejected by
that someone, who was holding a better offer at the time and only
trades up afterward. The procedure has a bias the engine names rather
than hides: the proposing side gets its best possible stable partner
and the accepting side its worst, so swapping which side proposes gives
the other extreme, and the two runs agree exactly when the stable
matching is unique. The matcher takes complete preference lists for
both sides, runs proposals from either side, verifies stability by
checking every pair for blocking, refuses incomplete or unequal lists,
and reports how many participants got their first choice under each
proposing side, because that gap is the proposer advantage made visible.
"""

from __future__ import annotations

from mesh.errors import Invalid


class StableMatching:
    def __init__(
        self,
        proposers: dict[str, list[str]],
        acceptors: dict[str, list[str]],
    ) -> None:
        if not proposers or len(proposers) != len(acceptors):
            raise Invalid("both sides need the same non-zero number of participants")
        for name, ranking in proposers.items():
            if sorted(ranking) != sorted(acceptors):
                raise Invalid(f"'{name}' must rank every acceptor exactly once")
        for name, ranking in acceptors.items():
            if sorted(ranking) != sorted(proposers):
                raise Invalid(f"'{name}' must rank every proposer exactly once")
        self.proposers = proposers
        self.acceptors = acceptors
        self.rounds = 0
        self.match = self._gale_shapley()

    def _gale_shapley(self) -> dict[str, str]:
        rank = {a: {p: i for i, p in enumerate(prefs)} for a, prefs in self.acceptors.items()}
        next_choice = dict.fromkeys(self.proposers, 0)
        held: dict[str, str] = {}
        free = sorted(self.proposers)
        while free:
            self.rounds += 1
            proposer = free.pop(0)
            target = self.proposers[proposer][next_choice[proposer]]
            next_choice[proposer] += 1
            current = held.get(target)
            if current is None:
                held[target] = proposer
            elif rank[target][proposer] < rank[target][current]:
                held[target] = proposer  # the acceptor trades up
                free.append(current)
            else:
                free.append(proposer)  # rejected, try the next preference
        return {p: a for a, p in held.items()}

    def blocking_pairs(self) -> list[tuple[str, str]]:
        p_rank = {p: {a: i for i, a in enumerate(prefs)} for p, prefs in self.proposers.items()}
        a_rank = {a: {p: i for i, p in enumerate(prefs)} for a, prefs in self.acceptors.items()}
        partner_of_acceptor = {a: p for p, a in self.match.items()}
        blocking = []
        for p in self.proposers:
            for a in self.acceptors:
                if self.match[p] == a:
                    continue
                prefers_a = p_rank[p][a] < p_rank[p][self.match[p]]
                prefers_p = a_rank[a][p] < a_rank[a][partner_of_acceptor[a]]
                if prefers_a and prefers_p:
                    blocking.append((p, a))
        return blocking

    def is_stable(self) -> bool:
        return not self.blocking_pairs()

    def first_choices(self) -> int:
        return sum(1 for p, a in self.match.items() if self.proposers[p][0] == a)

    def note(self) -> str:
        flipped = StableMatching(self.acceptors, self.proposers)
        same = {a: p for p, a in flipped.match.items()} == self.match
        swap = "gives the same matching" if same else "changes it"
        return (
            f"stable after {self.rounds} proposal(s): {self.first_choices()} proposer(s) got "
            f"their first choice; swapping sides {swap}, the proposer advantage made visible"
        )
