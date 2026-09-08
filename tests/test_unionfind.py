from __future__ import annotations

import random

import pytest

from mesh.errors import Missing
from mesh.unionfind import UnionFind


class TestBasics:
    def test_a_fresh_element_is_its_own_group(self):
        uf = UnionFind()
        uf.add("a")
        assert uf.find("a") == "a"
        assert uf.group_count() == 1

    def test_union_merges_two_groups(self):
        uf = UnionFind()
        uf.add("a")
        uf.add("b")
        assert uf.union("a", "b")
        assert uf.connected("a", "b")
        assert uf.group_count() == 1

    def test_union_of_already_connected_returns_false(self):
        uf = UnionFind()
        for e in "abc":
            uf.add(e)
        uf.union("a", "b")
        uf.union("b", "c")
        assert not uf.union("a", "c")  # already together via b

    def test_transitivity_holds(self):
        uf = UnionFind()
        for e in "abcd":
            uf.add(e)
        uf.union("a", "b")
        uf.union("c", "d")
        uf.union("b", "c")
        assert uf.connected("a", "d")


class TestRefusal:
    def test_find_on_an_unknown_element_is_refused(self):
        with pytest.raises(Missing):
            UnionFind().find("ghost")


class TestReport:
    def test_the_group_count_drops_with_each_real_merge(self):
        uf = UnionFind()
        for e in "abcde":
            uf.add(e)
        assert uf.group_count() == 5
        uf.union("a", "b")
        uf.union("c", "d")
        assert uf.group_count() == 3

    def test_the_largest_group_size_is_tracked(self):
        uf = UnionFind()
        for e in "abcd":
            uf.add(e)
        uf.union("a", "b")
        uf.union("b", "c")  # group {a,b,c}
        assert uf.largest_group_size() == 3

    def test_the_note_states_the_group_count(self):
        uf = UnionFind()
        uf.add("a")
        assert "1 group(s)" in uf.note()


class TestAgainstBruteForce:
    def test_connectivity_matches_a_reference_partition(self):
        rng = random.Random(11)
        for _ in range(30):
            elements = [str(i) for i in range(12)]
            uf = UnionFind()
            for e in elements:
                uf.add(e)
            # reference: dict element -> set label, merged naively
            label = {e: e for e in elements}
            for _ in range(15):
                a = rng.choice(elements)
                b = rng.choice(elements)
                uf.union(a, b)
                if label[a] != label[b]:
                    old = label[b]
                    for e in elements:
                        if label[e] == old:
                            label[e] = label[a]
            for a in elements:
                for b in elements:
                    assert uf.connected(a, b) == (label[a] == label[b])
