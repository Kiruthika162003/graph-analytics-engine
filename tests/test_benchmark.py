from __future__ import annotations

from itertools import combinations

import pytest

from mesh.benchmark import Benchmark
from mesh.errors import Invalid
from mesh.factories import cycle, path
from mesh.graph import Graph


def _edge_visits(g: Graph) -> int:
    return sum(1 for _u, _v, _w in g.edges())


def _pair_checks(g: Graph) -> int:
    return sum(1 for _a, _b in combinations(g.nodes(), 2))


def _refusing(g: Graph) -> None:
    if g.node_count() > 4:
        raise Invalid("too big for this reading")


class TestFrame:
    def test_sizes_ascend_and_seconds_are_never_negative(self):
        bench = Benchmark(cycle, [4, 8, 16]).add("edges", _edge_visits, _edge_visits)
        rows = bench.run()
        assert [size for _n, size, _s, _w, _f in rows] == [4, 8, 16]
        assert all(seconds >= 0 for _n, _s, seconds, _w, _f in rows)
        assert all(fail is None for _n, _s, _sec, _w, fail in rows)

    def test_growth_is_read_from_work_counts(self):
        bench = Benchmark(cycle, [8, 64])
        bench.add("edges", _edge_visits, _edge_visits)
        bench.add("pairs", _pair_checks, _pair_checks)
        bench.run()
        assert bench.growth("edges") == pytest.approx(1.0)
        assert 1.5 < bench.growth("pairs") < 2.5
        assert Benchmark.describe(bench.growth("edges")) == "linear"
        assert Benchmark.describe(bench.growth("pairs")) == "quadratic"
        assert Benchmark.describe(None) == "unknown"
        assert Benchmark.describe(0.1) == "constant"
        assert Benchmark.describe(3.0) == "steeper than quadratic"

    def test_a_reading_that_raises_is_reported_in_its_row(self):
        bench = Benchmark(path, [2, 8]).add("picky", _refusing)
        rows = bench.run()
        assert rows[0][4] is None
        assert rows[1][4] == "too big for this reading"
        assert "failed: too big for this reading" in bench.table()
        assert bench.growth("picky") is None


class TestRefusal:
    def test_bad_sizes_and_duplicate_names_are_refused(self):
        with pytest.raises(Invalid):
            Benchmark(cycle, [])
        with pytest.raises(Invalid):
            Benchmark(cycle, [8, 4])
        with pytest.raises(Invalid):
            Benchmark(cycle, [0, 4])
        bench = Benchmark(cycle, [3]).add("edges", _edge_visits)
        with pytest.raises(Invalid):
            bench.add("edges", _edge_visits)


class TestReport:
    def test_the_table_and_note_describe_the_run(self):
        bench = Benchmark(cycle, [4, 16]).add("edges", _edge_visits, _edge_visits)
        bench.run()
        table = bench.table().splitlines()
        assert table[0].startswith("reading")
        assert len(table) == 3
        assert bench.note() == "2 row(s) over sizes [4, 16]; growth: edges linear"
