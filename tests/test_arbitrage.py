from __future__ import annotations

import pytest

from mesh.arbitrage import Arbitrage
from mesh.errors import Invalid, Missing


def _market(profitable: bool) -> Arbitrage:
    a = Arbitrage()
    a.add_rate("usd", "eur", 0.9)
    a.add_rate("eur", "gbp", 0.85)
    # gbp back to usd at 1.35 closes a profitable loop, 1.25 does not
    a.add_rate("gbp", "usd", 1.35 if profitable else 1.25)
    a.add_rate("usd", "jpy", 150.0)
    a.add_rate("jpy", "usd", 1 / 160.0)
    return a


class TestDetection:
    def test_a_profitable_loop_is_found_and_multiplies_above_one(self):
        a = _market(profitable=True)
        cycle = a.find_cycle()
        assert cycle is not None
        assert set(cycle) == {"usd", "eur", "gbp"}
        assert a.multiplier(cycle) == pytest.approx(0.9 * 0.85 * 1.35)
        assert a.multiplier(cycle) > 1

    def test_a_fair_market_has_no_cycle(self):
        assert _market(profitable=False).find_cycle() is None

    def test_the_reported_cycle_uses_existing_rates_in_order(self):
        a = _market(profitable=True)
        cycle = a.find_cycle()
        assert cycle is not None
        for i in range(len(cycle)):
            assert (cycle[i], cycle[(i + 1) % len(cycle)]) in a.rates

    def test_a_two_currency_round_trip_above_one_is_an_arbitrage(self):
        a = Arbitrage()
        a.add_rate("x", "y", 2.0)
        a.add_rate("y", "x", 0.6)  # 1.2 round trip
        cycle = a.find_cycle()
        assert cycle is not None
        assert a.multiplier(cycle) == pytest.approx(1.2)

    def test_the_loop_sits_where_the_rates_say(self):
        # the jpy leg loses money, so the found loop never includes it
        cycle = _market(profitable=True).find_cycle()
        assert cycle is not None
        assert "jpy" not in cycle


class TestRefusals:
    def test_a_non_positive_rate_is_refused(self):
        with pytest.raises(Invalid):
            Arbitrage().add_rate("a", "b", 0.0)

    def test_an_empty_market_is_refused(self):
        with pytest.raises(Missing):
            Arbitrage().find_cycle()


class TestReport:
    def test_the_note_names_the_loop_and_its_gain(self):
        note = _market(profitable=True).note()
        assert "arbitrage" in note
        assert "above one" in note

    def test_the_note_says_no_arbitrage_on_a_fair_market(self):
        assert "no arbitrage" in _market(profitable=False).note()
