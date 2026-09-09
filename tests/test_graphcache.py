from __future__ import annotations

import pytest

from mesh.errors import Invalid
from mesh.factories import cycle, path
from mesh.graph import Graph
from mesh.graphcache import ReadingCache, fingerprint


def _counting_reading():
    calls = {"n": 0}

    def reading(g: Graph) -> int:
        calls["n"] += 1
        return g.edge_count()

    return reading, calls


class TestFingerprint:
    def test_equal_content_built_in_different_orders_shares_a_fingerprint(self):
        a = Graph()
        b = Graph()
        for n in "xyz":
            a.add_node(n)
        for n in "zyx":
            b.add_node(n)
        a.add_edge("x", "y")
        a.add_edge("y", "z")
        b.add_edge("z", "y")
        b.add_edge("y", "x")
        assert fingerprint(a) == fingerprint(b)

    def test_direction_weights_and_edges_all_change_the_fingerprint(self):
        base = path(3)
        assert fingerprint(base) != fingerprint(cycle(3))
        weighted = path(3)
        weighted.add_edge("0", "1", 2.0)
        assert fingerprint(base) != fingerprint(weighted)
        assert fingerprint(Graph()) != fingerprint(Graph(directed=True))

    def test_the_fingerprint_is_stable_text(self):
        assert fingerprint(path(2)) == fingerprint(path(2))
        assert len(fingerprint(path(2))) == 16


class TestCache:
    def test_the_same_graph_asked_twice_computes_once(self):
        reading, calls = _counting_reading()
        cache = ReadingCache()
        cache.register("edges", reading)
        g = cycle(5)
        assert cache.get("edges", g) == 5
        assert cache.get("edges", g) == 5
        assert calls["n"] == 1
        assert cache.hits == 1 and cache.misses == 1

    def test_an_edited_graph_computes_again(self):
        reading, calls = _counting_reading()
        cache = ReadingCache()
        cache.register("edges", reading)
        g = cycle(4)
        cache.get("edges", g)
        g.add_edge("0", "2")
        assert cache.get("edges", g) == 5
        assert calls["n"] == 2

    def test_forget_drops_every_answer_for_that_graph(self):
        cache = ReadingCache()
        cache.register("edges", lambda g: g.edge_count())
        cache.register("nodes", lambda g: g.node_count())
        g = path(4)
        cache.get("edges", g)
        cache.get("nodes", g)
        assert cache.forget(g) == 2
        assert cache.forget(g) == 0

    def test_unknown_and_duplicate_readings_are_refused(self):
        cache = ReadingCache()
        cache.register("edges", lambda g: g.edge_count())
        with pytest.raises(Invalid, match="no reading called 'fame'"):
            cache.get("fame", path(2))
        with pytest.raises(Invalid, match="already registered"):
            cache.register("edges", lambda g: g.node_count())


class TestReport:
    def test_the_note_counts_hits_and_misses(self):
        cache = ReadingCache()
        cache.register("edges", lambda g: g.edge_count())
        g = path(3)
        cache.get("edges", g)
        cache.get("edges", g)
        cache.get("edges", g)
        note = cache.note()
        assert note.startswith("1 reading(s) registered, 1 answer(s) held")
        assert "2 hit(s) and 1 miss(es) (67% served from memory)" in note
