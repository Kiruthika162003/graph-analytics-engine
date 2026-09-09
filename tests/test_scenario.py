from __future__ import annotations

import pytest

from mesh.errors import Invalid
from mesh.factories import cycle, path, star
from mesh.graph import Graph
from mesh.scenario import Scenario, readings


class TestReadings:
    def test_a_path_reads_its_diameter_and_mean_distance(self):
        r = readings(path(4))
        assert r["pieces"] == 1
        assert r["largest"] == 4
        assert r["diameter"] == 3
        assert r["mean distance"] == pytest.approx(10 / 6)
        assert r["edges"] == 3

    def test_an_empty_graph_reads_zero(self):
        assert readings(Graph()) == {
            "pieces": 0.0,
            "largest": 0.0,
            "diameter": 0.0,
            "mean distance": 0.0,
            "edges": 0.0,
        }


class TestChanges:
    def test_removing_a_bridge_adds_a_piece(self):
        table = Scenario(path(4)).remove_edge("1", "2")
        assert table["pieces"] == (1.0, 2.0)
        assert table["largest"] == (4.0, 2.0)

    def test_adding_a_chord_to_a_long_path_lowers_the_diameter(self):
        table = Scenario(path(7)).add_edge("0", "6")
        assert table["diameter"] == (6.0, 3.0)
        assert table["edges"] == (6.0, 7.0)

    def test_removing_a_leaf_changes_only_the_counts(self):
        table = Scenario(star(4)).remove_node("1")
        assert table["pieces"] == (1.0, 1.0)
        assert table["diameter"] == (2.0, 2.0)
        assert table["largest"] == (5.0, 4.0)
        assert table["edges"] == (4.0, 3.0)

    def test_the_original_is_untouched_after_every_kind_of_change(self):
        sc = Scenario(cycle(5))
        sc.remove_node("0")
        sc.remove_edge("1", "2")
        sc.add_edge("0", "2")
        assert sc.untouched()
        assert cycle(5).edge_count() == 5


class TestSweeps:
    def test_the_hub_of_a_star_is_the_most_damaging_removal(self):
        ranked = Scenario(star(5)).sweep_nodes(by="largest")
        assert ranked[0] == ("0", -5.0)

    def test_the_best_addition_to_a_path_closes_it_or_nearly(self):
        # the guess was that joining the ends wins alone; joining 0 to 4 also brings
        # the diameter from 5 to 3, since the pendant end stays within three of all,
        # and ties resolve by name, so 0-4 is listed first
        ranked = Scenario(path(6)).sweep_additions(by="diameter")
        assert ranked[0][1] == -2
        best = {pair for pair, delta in ranked if delta == -2}
        assert ("0", "5") in best
        assert ranked[0][0] in best

    def test_unknown_readings_are_refused(self):
        with pytest.raises(Invalid):
            Scenario(path(3)).sweep_nodes(by="fame")
        with pytest.raises(Invalid):
            Scenario(path(3)).sweep_additions(by="fame")


class TestRefusal:
    def test_absent_items_and_present_edges_are_refused_by_name(self):
        sc = Scenario(path(3))
        with pytest.raises(Invalid, match="no node 'zz'"):
            sc.remove_node("zz")
        with pytest.raises(Invalid, match="no edge 0-2"):
            sc.remove_edge("0", "2")
        with pytest.raises(Invalid, match="already present"):
            sc.add_edge("0", "1")
        with pytest.raises(Invalid, match="no node 'zz' to join"):
            sc.add_edge("0", "zz")


class TestReport:
    def test_the_note_lists_only_what_changed(self):
        sc = Scenario(path(4))
        assert sc.note(sc.remove_edge("1", "2")).startswith("pieces 1 to 2; largest 4 to 2")
        assert sc.note(sc.remove_node("3")) != "nothing changed"
        assert sc.note({"pieces": (1.0, 1.0)}) == "nothing changed"
