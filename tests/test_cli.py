from __future__ import annotations

from pathlib import Path

import pytest

from mesh.cli import main

OFFICE = """undirected
ada ben
ben cal
cal dee
eve
"""


@pytest.fixture
def office(tmp_path: Path) -> str:
    target = tmp_path / "office.txt"
    target.write_text(OFFICE, encoding="utf-8")
    return str(target)


class TestProbeCommands:
    def test_summary_reports_the_probe_count(self, capsys):
        assert main(["summary"]) == 0
        out = capsys.readouterr().out
        assert "probe(s), 0 broken" in out

    def test_check_passes_when_every_probe_holds(self, capsys):
        assert main(["check"]) == 0
        assert capsys.readouterr().out.strip() == "all probes hold"

    def test_probes_prints_a_detail_line_per_probe(self, capsys):
        assert main(["probes"]) == 0
        lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
        assert len(lines) >= 40


class TestGraphCommands:
    def test_describe_prints_the_first_look_summary(self, office, capsys):
        assert main(["describe", office]) == 0
        out = capsys.readouterr().out
        assert out.startswith("undirected graph with 5 node(s) and 3 edge(s)")
        assert "not connected" in out

    def test_ask_answers_a_verb_with_arguments(self, office, capsys):
        assert main(["ask", office, "path", "ada", "dee"]) == 0
        assert capsys.readouterr().out.strip() == "ada > ben > cal > dee"
        assert main(["ask", office, "degree", "ben"]) == 0
        assert capsys.readouterr().out.strip() == "2"

    def test_ask_with_no_words_lists_the_verbs(self, office, capsys):
        assert main(["ask", office]) == 0
        assert capsys.readouterr().out.startswith("verbs: ")

    def test_a_typo_is_answered_in_words_not_a_traceback(self, office, capsys):
        assert main(["ask", office, "degree", "zed"]) == 0
        assert capsys.readouterr().out.strip() == "no node called 'zed'"


class TestBadFiles:
    def test_a_missing_file_is_reported_and_fails(self, tmp_path, capsys):
        missing = str(tmp_path / "nowhere.txt")
        assert main(["describe", missing]) == 1
        assert "cannot read" in capsys.readouterr().out

    def test_a_file_without_a_header_is_reported_and_fails(self, tmp_path, capsys):
        bad = tmp_path / "bad.txt"
        bad.write_text("ada ben\n", encoding="utf-8")
        assert main(["ask", str(bad), "nodes"]) == 1
        assert "is not an edge list" in capsys.readouterr().out
