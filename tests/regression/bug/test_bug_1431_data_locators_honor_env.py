# SPDX-License-Identifier: BSD-3-Clause
"""Regression: every data-file locator must honor ``$TENGRI_DATA_DIR`` (#1431).

Component locators used to anchor on ``Path(__file__).resolve().parents[N] /
"data"``, which reaches the repo's own ``data/`` in a source checkout and
nothing at all anywhere else. ``$TENGRI_DATA_DIR`` — the documented way to keep
grids on a scratch filesystem, and the only way a wheel install finds them —
was not consulted, so three *published* notebooks failed on grids the user
already had:

* ``cue_weights.npz`` — nb03 and the ``star_forming_photometry`` recipe
* ``skirtor_templates_v3.h5`` — nb02
* ``nenkova08_torus_grid.h5`` — nb02, whose message told the user to *build* a
  grid that was sitting in their configured data directory

An earlier pass (#1431) converted six ``_find_*`` functions and stopped there,
leaving sixteen sites on the old anchoring. The last test here is the one that
matters long-term: it fails on *any* new package-anchored data locator, so the
next partial sweep cannot leave siblings behind.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

pytestmark = pytest.mark.regression_bug

SRC = pathlib.Path(__file__).resolve().parents[3] / "src" / "tengri"


# ── The mechanism ─────────────────────────────────────────────────


def test_package_or_env_data_path_prefers_the_env_dir(tmp_path, monkeypatch):
    """An import-time default must resolve into ``$TENGRI_DATA_DIR`` when set."""
    from tengri._data_setup import package_or_env_data_path

    (tmp_path / "cue_weights.npz").write_bytes(b"x")
    monkeypatch.setenv("TENGRI_DATA_DIR", str(tmp_path))
    assert package_or_env_data_path("cue_weights.npz") == tmp_path / "cue_weights.npz"


def test_package_or_env_data_path_never_raises_when_absent(tmp_path, monkeypatch):
    """Missing data must stay a load-time error, not an import-time one.

    These feed module-level constants and default arguments, so raising here
    would make ``import tengri`` fail on an incomplete data directory.
    """
    from tengri._data_setup import package_or_env_data_path

    monkeypatch.setenv("TENGRI_DATA_DIR", str(tmp_path))
    resolved = package_or_env_data_path("__definitely_absent__.h5")
    assert resolved.name == "__definitely_absent__.h5"
    assert not resolved.exists()


def test_find_data_prefers_the_env_dir(tmp_path, monkeypatch):
    from tengri._data_setup import find_data

    (tmp_path / "probe_grid.h5").write_bytes(b"x")
    monkeypatch.setenv("TENGRI_DATA_DIR", str(tmp_path))
    assert find_data("probe_grid.h5") == tmp_path / "probe_grid.h5"
    assert find_data("__absent__.h5") is None


def test_find_data_preference_is_name_major(tmp_path, monkeypatch):
    """The first *filename* wins, even if a later one sits in an earlier dir.

    The locators this replaced ranked candidates by scientific fidelity
    (SKIRTOR v3 over v2 over the npz). A directory-major search would silently
    hand back the older grid whenever it happened to sit earlier on the path.
    """
    from tengri._data_setup import find_data

    preferred, fallback = tmp_path / "a", tmp_path / "b"
    preferred.mkdir()
    fallback.mkdir()
    (fallback / "grid_v3.h5").write_bytes(b"x")
    (preferred / "grid_v2.h5").write_bytes(b"x")
    monkeypatch.setenv("TENGRI_DATA_DIR", str(preferred))
    monkeypatch.chdir(fallback)
    assert find_data("grid_v3.h5", "grid_v2.h5").name == "grid_v3.h5"


# ── The three that were confirmed broken ──────────────────────────


@pytest.mark.parametrize(
    ("filename", "locator"),
    [
        ("skirtor_templates_v3.h5", "tengri.components.agn.skirtor:_find_skirtor_grid"),
        ("nenkova08_torus_grid.h5", "tengri.components.agn.torus:_find_nenkova_grid"),
    ],
)
def test_component_locator_finds_the_grid_in_the_env_dir(filename, locator, tmp_path, monkeypatch):
    """Each locator must return the copy under ``$TENGRI_DATA_DIR``.

    Asserting the *returned path*, not merely that it did not raise: in a source
    checkout the repo's own ``data/`` would satisfy a weaker assertion whether or
    not the env var was consulted at all.
    """
    import importlib

    module_name, func_name = locator.split(":")
    func = getattr(importlib.import_module(module_name), func_name)

    (tmp_path / filename).write_bytes(b"x")
    monkeypatch.setenv("TENGRI_DATA_DIR", str(tmp_path))
    assert func() == str(tmp_path / filename)


def test_cue_weights_default_honors_the_env_dir(tmp_path):
    """The Cue default must resolve into ``$TENGRI_DATA_DIR``.

    It is bound at import, so a subprocess is the only way to exercise the env
    var honestly — ``monkeypatch.setenv`` in-process would run after the module
    was already imported and would pass no matter what the code did.

    This is the failure that broke nb03 and the ``star_forming_photometry``
    recipe: the weights were sitting in the configured data directory and the
    default pointed at a package-relative path that did not exist.
    """
    import os
    import subprocess
    import sys

    import tengri

    (tmp_path / "cue_weights.npz").write_bytes(b"x")
    src_root = str(pathlib.Path(tengri.__file__).resolve().parents[1])
    env = {**os.environ, "TENGRI_DATA_DIR": str(tmp_path), "PYTHONPATH": src_root}
    out = subprocess.run(
        [
            sys.executable,
            "-c",
            "from tengri.components.nebular import _DEFAULT_CUE_WEIGHTS_PATH as p; print(p)",
        ],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    assert out.stdout.strip() == str(tmp_path / "cue_weights.npz")


# ── The guard that catches the next straggler ─────────────────────


def _package_anchored_paths() -> list[str]:
    """Every ``...parents[N]`` subscript under ``src/tengri``.

    AST-based on purpose: a text scan would also match the prose in
    ``_data_setup``'s docstrings, which legitimately describes the old pattern.

    Deliberately flags the bare subscript rather than only the
    ``parents[N] / "data"`` division. Two of the three locators that actually
    broke — ``skirtor``, ``torus`` — assigned ``base = ...parents[4]`` on one
    line and joined the filename on another, so a division-shaped rule would
    have stayed green on exactly the bugs that prompted it.
    """
    offenders: set[str] = set()
    for path in sorted(SRC.rglob("*.py")):
        # _data_setup is where the one legitimate package anchor lives.
        if path.name == "_data_setup.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Subscript)
                and isinstance(node.value, ast.Attribute)
                and node.value.attr == "parents"
            ):
                offenders.add(f"{path.relative_to(SRC)}:{node.lineno}")
    return sorted(offenders)


def test_no_component_resolves_data_by_counting_parent_directories():
    """No module may rebuild a data path from ``parents[N]``.

    ``parents[N]`` is a depth count from the calling file: it silently resolves
    somewhere else the moment a module moves between directory levels, and it
    cannot see ``$TENGRI_DATA_DIR`` at all. ``data_path`` / ``find_data`` /
    ``package_or_env_data_path`` are anchored once, in ``_data_setup``.
    """
    offenders = _package_anchored_paths()
    assert offenders == [], (
        "package-anchored data locators found — route these through "
        "tengri._data_setup instead:\n  " + "\n  ".join(offenders)
    )
