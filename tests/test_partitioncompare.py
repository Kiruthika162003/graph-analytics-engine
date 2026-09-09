from __future__ import annotations

from itertools import combinations

import pytest

from mesh.errors import Invalid
from mesh.factories import cycle
from mesh.graph import Graph
from mesh.partitioncompare import PartitionCompare


def _two_cliques() -> Graph:
    g = Graph()
    for group in ("abcd", "wxyz"):
        for n in group:
            g.add_node(n)
        for a, b in combinations(group, 2):
            g.add_edge(a, b)
    g.add_edge("d", "w")
    return g


NATURAL = [list("abcd"), list("wxyz")]
HALVED = [list("abwx"), list("cdyz")]


class TestAgreement:
    def test_identical_partitions_score_one_whatever_the_labels(self):
        pc = PartitionCompare(_two_cliques(), NATURAL, [list("wxyz"), list("abcd")])
        assert pc.rand() == 1.0
        assert pc.jaccard() == 1.0
        assert pc.nmi() == pytest.approx(1.0)

    def test_singletons_against_one_group_share_no_information(self):
        g = _two_cliques()
        pc = PartitionCompare(g, [[n] for n in g.nodes()], [g.nodes()])
        assert pc.nmi() == pytest.approx(0.0, abs=1e-12)
        assert pc.jaccard() == 0.0
        assert pc.rand() == 0.0

    def test_every_agreement_reading_is_symmetric(self):
        g = _two_cliques()
        ab = PartitionCompare(g, NATURAL, HALVED)
        ba = PartitionCompare(g, HALVED, NATURAL)
        assert ab.rand() == ba.rand()
        assert ab.jaccard() == ba.jaccard()
        assert ab.nmi() == pytest.approx(ba.nmi())

    def test_crossing_halves_are_independent_and_a_partial_overlap_sits_between(self):
        # the guess was that the natural split and the halving share some information;
        # every cell of their joint table holds two of eight nodes, exactly the product
        # of the marginals, so the mutual information is zero. moving one node gives
        # the strictly-between case
        g = _two_cliques()
        assert PartitionCompare(g, NATURAL, HALVED).nmi() == pytest.approx(0.0, abs=1e-12)
        partial = PartitionCompare(g, NATURAL, [list("abc"), list("dwxyz")])
        assert 0 < partial.nmi() < 1
        assert 0 < partial.jaccard() < 1


class TestQuality:
    def test_the_natural_split_has_positive_modularity_and_the_halving_negative(self):
        pc = PartitionCompare(_two_cliques(), NATURAL, HALVED)
        assert pc.modularity("first") > 0
        assert pc.modularity("second") < 0
        assert pc.coverage("first") == pytest.approx(12 / 13)

    def test_one_group_covers_everything_with_zero_modularity(self):
        g = cycle(6)
        pc = PartitionCompare(g, [g.nodes()], [g.nodes()])
        assert pc.coverage() == 1.0
        assert pc.modularity() == pytest.approx(0.0)

    def test_an_edgeless_graph_reads_zero_modularity_and_full_coverage(self):
        g = Graph()
        for n in "ab":
            g.add_node(n)
        pc = PartitionCompare(g, [["a"], ["b"]], [["a", "b"]])
        assert pc.modularity() == 0.0
        assert pc.coverage() == 1.0


class TestRefusal:
    def test_repeated_missing_and_foreign_nodes_are_refused_by_name(self):
        g = _two_cliques()
        with pytest.raises(Invalid, match="'a' appears in two"):
            PartitionCompare(g, [list("abcd"), list("awxyz")], NATURAL)
        with pytest.raises(Invalid, match="'z' is in no group"):
            PartitionCompare(g, [list("abcd"), list("wxy")], NATURAL)
        with pytest.raises(Invalid, match="'q' is not a node"):
            PartitionCompare(g, [list("abcdq"), list("wxyz")], NATURAL)


class TestReport:
    def test_the_note_lists_agreement_and_both_modularities(self):
        note = PartitionCompare(_two_cliques(), NATURAL, NATURAL).note()
        assert note.startswith("rand 1.000, jaccard 1.000, nmi 1.000; modularity ")
