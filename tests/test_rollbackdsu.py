from __future__ import annotations

import random

import pytest

from mesh.errors import Invalid, Missing
from mesh.rollbackdsu import RollbackUnionFind
from mesh.unionfind import UnionFind


class TestUnionAndUndo:
    def test_a_union_joins_and_a_rollback_separates(self):
        uf = RollbackUnionFind()
        uf.add("a")
        uf.add("b")
        uf.union("a", "b")
        assert uf.connected("a", "b")
        uf.rollback()
        assert not uf.connected("a", "b")
        assert uf.group_count() == 2

    def test_rolling_back_to_a_checkpoint_undoes_everything_after_it(self):
        uf = RollbackUnionFind()
        for e in "abcd":
            uf.add(e)
        uf.union("a", "b")
        mark = uf.checkpoint()
        uf.union("c", "d")
        uf.union("b", "c")
        assert uf.group_count() == 1
        uf.rollback(mark)
        assert uf.connected("a", "b")
        assert not uf.connected("a", "c")
        assert uf.group_count() == 3

    def test_a_no_op_union_still_rolls_back_cleanly(self):
        uf = RollbackUnionFind()
        uf.add("a")
        uf.add("b")
        uf.union("a", "b")
        uf.union("a", "b")  # already together
        uf.rollback()  # undoes the no-op
        assert uf.connected("a", "b")
        uf.rollback()
        assert not uf.connected("a", "b")

    def test_size_is_restored_so_later_unions_stay_balanced(self):
        uf = RollbackUnionFind()
        for e in "abc":
            uf.add(e)
        uf.union("a", "b")
        uf.rollback()
        uf.union("b", "c")
        assert uf.group_count() == 2


class TestRefusals:
    def test_a_rollback_before_history_is_refused(self):
        uf = RollbackUnionFind()
        with pytest.raises(Invalid):
            uf.rollback(-1)

    def test_find_on_an_unknown_element_is_refused(self):
        with pytest.raises(Missing):
            RollbackUnionFind().find("ghost")


class TestOfflineConnectivity:
    def test_a_temporary_edge_joins_only_while_alive(self):
        # a-b permanent, b-c alive for one window; queries inside and outside
        uf = RollbackUnionFind()
        for e in "abc":
            uf.add(e)
        uf.union("a", "b")
        before = uf.connected("a", "c")
        mark = uf.checkpoint()
        uf.union("b", "c")
        during = uf.connected("a", "c")
        uf.rollback(mark)
        after = uf.connected("a", "c")
        assert (before, during, after) == (False, True, False)


class TestAgainstPlainUnionFind:
    def test_it_matches_the_compressed_version_on_random_merge_sequences(self):
        rng = random.Random(281)
        for _ in range(30):
            elements = [str(i) for i in range(10)]
            plain = UnionFind()
            roll = RollbackUnionFind()
            for e in elements:
                plain.add(e)
                roll.add(e)
            for _ in range(12):
                a, b = rng.choice(elements), rng.choice(elements)
                assert plain.union(a, b) == roll.union(a, b)
            for a in elements:
                for b in elements:
                    assert plain.connected(a, b) == roll.connected(a, b)
            assert plain.group_count() == roll.group_count()


class TestReport:
    def test_the_note_states_depth_and_groups(self):
        uf = RollbackUnionFind()
        uf.add("a")
        uf.add("b")
        uf.union("a", "b")
        assert "undo stack 1 deep after 1 union(s), 1 group(s)" in uf.note()
