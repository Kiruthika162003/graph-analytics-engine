from __future__ import annotations

import importlib
import pkgutil

import pytest

import examples

_MODULES = sorted(m.name for m in pkgutil.iter_modules(examples.__path__))


class TestExamples:
    def test_there_is_at_least_one_example(self):
        assert _MODULES

    @pytest.mark.parametrize("name", _MODULES)
    def test_each_example_runs_clean_and_prints(self, name: str, capsys):
        module = importlib.import_module(f"examples.{name}")
        assert module.main() == 0
        out = capsys.readouterr().out
        assert out.strip(), f"{name} printed nothing"
        assert "—" not in out, f"{name} printed an em dash"
