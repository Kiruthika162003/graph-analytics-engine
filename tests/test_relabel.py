from __future__ import annotations

import pytest

from mesh.errors import Invalid
from mesh.factories import cycle, path, star
from mesh.graph import Graph
from mesh.graphio import same_graph
from mesh.relabel import Relabel


def _weighted() -> Graph:
    g = Graph()
    for n in ("Ada", "Ben", "Cal"):
        g.add_node(n)
    g.add_edge("Ada", "Ben", 2.5)
    g.add_edge("Ben", "Cal", 4.0)
    return g


class TestMethods:
    def test_a_mapping_keeps_edges_and_weights_and_returns_the_inverse(self):
        out, inverse = Relabel(_weighted()).by_mapping({"Ada": "a", "Ben": "b", "Cal": "c"})
        assert out.weight("a", "b") == 2.5
        assert out.weight("b", "c") == 4.0
        assert inverse == {"a": "Ada", "b": "Ben", "c": "Cal"}

    def test_a_function_folds_case_and_a_prefix_keeps_names_apart(self):
        lower, _inv = Relabel(_weighted()).by_function(str.lower)
        assert sorted(lower.nodes()) == ["ada", "ben", "cal"]
        prefixed, _inv = Relabel(path(3)).with_prefix("p:")
        assert prefixed.has_edge("p:0", "p:1")
        suffixed, _inv = Relabel(path(3)).with_suffix("!")
        assert suffixed.has_edge("1!", "2!")

    def test_the_input_is_left_untouched(self):
        g = _weighted()
        Relabel(g).with_prefix("x")
        assert sorted(g.nodes()) == ["Ada", "Ben", "Cal"]

    def test_relabeling_by_a_mapping_and_its_inverse_returns_an_equal_graph(self):
        g = cycle(5)
        out, inverse = Relabel(g).with_prefix("r")
        back, _inv = Relabel(out).by_mapping(inverse)
        assert same_graph(back, g)


class TestCanonical:
    def test_equal_graphs_in_different_orders_number_alike(self):
        a = Graph()
        b = Graph()
        for n in ("hub", "x", "y", "z"):
            a.add_node(n)
        for n in ("z", "y", "x", "hub"):
            b.add_node(n)
        for leaf in "xyz":
            a.add_edge("hub", leaf)
            b.add_edge(leaf, "hub")
        ca, _ = Relabel(a).canonical()
        cb, _ = Relabel(b).canonical()
        assert same_graph(ca, cb)
        assert ca.degree("0") == 3

    def test_the_canonical_names_are_zero_through_n_minus_one(self):
        out, inverse = Relabel(star(3)).canonical()
        assert sorted(out.nodes()) == ["0", "1", "2", "3"]
        assert inverse["0"] == "0"


class TestRefusal:
    def test_merging_and_dropping_are_refused_by_name(self):
        with pytest.raises(Invalid, match="merges 0 and 1 into 'same'"):
            Relabel(path(3)).by_mapping({"0": "same", "1": "same", "2": "c"})
        with pytest.raises(Invalid, match="drops '2'"):
            Relabel(path(3)).by_mapping({"0": "a", "1": "b"})
        # the guess was that first letters clash; Ada, Ben, and Cal start differently,
        # so the merging function now sends every name to one
        first_letters, _inv = Relabel(_weighted()).by_function(lambda n: n[0])
        assert sorted(first_letters.nodes()) == ["A", "B", "C"]
        with pytest.raises(Invalid, match="merges"):
            Relabel(_weighted()).by_function(lambda _n: "one")


class TestReport:
    def test_the_note_counts_changed_names(self):
        rl = Relabel(path(3))
        _out, inverse = rl.by_mapping({"0": "0", "1": "one", "2": "two"})
        assert rl.note(inverse) == "3 node(s) relabeled, 2 name(s) actually changed"
