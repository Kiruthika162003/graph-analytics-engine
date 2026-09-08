from __future__ import annotations

from mesh.probes import all_probes, broken, report


class TestFrame:
    def test_there_is_at_least_one_probe(self):
        assert len(all_probes()) >= 1

    def test_no_probe_is_broken(self):
        # every registered guarantee must hold; this is the build gate
        assert broken() == []

    def test_each_probe_line_states_its_status(self):
        for probe in all_probes():
            assert "holds" in probe.line() or "BROKEN" in probe.line()

    def test_the_report_counts_probes(self):
        text = report()
        assert "probe(s)" in text
