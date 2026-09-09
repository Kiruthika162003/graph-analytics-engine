from __future__ import annotations

import importlib
import inspect
import pkgutil
from pathlib import Path

import pytest

import examples
import mesh
from mesh.probes import all_probes

MODULES = sorted(name for _f, name, _p in pkgutil.walk_packages(mesh.__path__, "mesh."))
EM_DASH = chr(0x2014)
EN_DASH = chr(0x2013)


def _source(module_name: str) -> str:
    return inspect.getsource(importlib.import_module(module_name))


class TestEveryModule:
    def test_the_package_is_large_and_every_module_imports(self):
        assert len(MODULES) > 150
        for name in MODULES:
            importlib.import_module(name)

    @pytest.mark.parametrize("name", MODULES)
    def test_each_module_opens_with_a_docstring_in_plain_prose(self, name: str):
        module = importlib.import_module(name)
        doc = module.__doc__ or ""
        assert doc.strip(), f"{name} has no docstring"
        assert EM_DASH not in doc, f"{name} uses an em dash"
        assert EN_DASH not in doc, f"{name} uses an en dash"
        assert not any(ord(ch) > 0xFFFF for ch in doc), f"{name} carries a symbol outside text"
        first = doc.strip().splitlines()[0]
        assert len(first) < 100, f"{name} opens with a line over 99 characters"

    @pytest.mark.parametrize("name", MODULES)
    def test_no_module_leaves_a_stub_behind(self, name: str):
        text = _source(name)
        assert "TODO" not in text, f"{name} carries a TODO"
        assert "NotImplementedError" not in text, f"{name} raises NotImplementedError"
        assert EM_DASH not in text, f"{name} uses an em dash in code"


class TestExampleSurface:
    def test_every_example_names_its_own_run_line(self):
        names = sorted(n for _f, n, _p in pkgutil.iter_modules(examples.__path__))
        assert len(names) >= 25
        for name in names:
            module = importlib.import_module(f"examples.{name}")
            doc = module.__doc__ or ""
            run_line = f"Run with: python -m examples.{name}"
            assert run_line in doc, f"{name} misstates its run line"
            assert EM_DASH not in doc
            assert len(doc.strip().splitlines()[0]) < 100
            assert callable(module.main)

    def test_the_examples_share_no_helper_names_with_the_engine(self):
        # an example is a script, not a module the engine imports; its helpers stay local
        engine = {name.rsplit(".", 1)[-1] for name in MODULES}
        for _f, name, _p in pkgutil.iter_modules(examples.__path__):
            assert name not in engine, f"examples.{name} shadows mesh.{name}"


class TestCounterSurface:
    def test_the_strict_counter_walks_the_four_trees(self):
        root = Path(__file__).resolve().parents[1]
        text = (root / "scripts" / "strictcount.py").read_text(encoding="utf-8")
        for tree in ("mesh", "tests", "examples", "scripts"):
            assert f'"{tree}/**/*.py"' in text or f"'{tree}/**/*.py'" in text, tree
        assert EM_DASH not in text


class TestProbeSurface:
    def test_the_probes_number_at_least_sixty_across_many_provers(self):
        probes = all_probes()
        assert len(probes) >= 60
        assert len({p.prober for p in probes}) >= 40

    def test_every_probe_holds_and_carries_readings(self):
        for probe in all_probes():
            assert probe.holds, probe.prober
            assert probe.readings, probe.prober

    def test_every_probe_guarantee_is_short_plain_prose(self):
        for probe in all_probes():
            assert probe.guarantee.strip()
            assert EM_DASH not in probe.guarantee
            assert len(probe.guarantee) < 100
