# SPDX-License-Identifier: BSD-3-Clause
"""Pin the single source of truth for BMA model keys (``_bma_keys``).

Pinned invariants, all jax-free:
- All 110 models (100 factorial + 5 grid + 5 X-like) round-trip through
  ``model_key`` / ``parse_model_key`` and every key is filename safe.
- The grid mappings (registered SSP grid, sfh type, attenuation, isochrone,
  spectral library) match ``config_metadata`` for configurations I-V. That file is
  read with ``ast``, never imported, so the check cannot share a mistake with the
  module it checks.
- No module other than ``_bma_keys`` builds a model-key string: the runner and
  the combiner once each restated the format and none of the combiner's keys
  matched what the runner wrote.
- Importing the runner, the combiner and the key module loads no JAX backend.
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path

import pytest

PAPER1 = Path(__file__).resolve().parents[1]
ANALYSIS = PAPER1.parent
for entry in (str(ANALYSIS), str(PAPER1)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from paper1 import _bma_keys as bk

pytestmark = pytest.mark.unit

CONFIG_METADATA = PAPER1 / "config_metadata.py"

# Attenuation display strings in config_metadata -> the factorial axis slug.
_ATTENUATION_SLUG = {
    "Kriek+13, 2-comp": "kriek_conroy_2c",
    "Calzetti, 1-comp": "calzetti",
    "Charlot+2000, 2-comp": "cf00_2c",
    "SMC, 1-comp": "smc",
}
_ISOCHRONE_SLUG = {"MIST": "mist", "PARSEC": "prsc", "BPASS": "bpass"}
_LIBRARY_SLUG = {"C3K": "c3k", "MILES": "miles"}


def _all_models() -> list[dict[str, str]]:
    return bk.enumerate_factorial() + bk.enumerate_named_all()


def _metadata_assignment(name: str) -> ast.expr:
    tree = ast.parse(CONFIG_METADATA.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            return node.value
    raise AssertionError(f"{name} not assigned in {CONFIG_METADATA}")


def _configs_from_metadata() -> dict[str, dict[str, str]]:
    """String-valued fields of every ``CONFIGS`` entry, read without importing."""
    node = _metadata_assignment("CONFIGS")
    assert isinstance(node, ast.Dict)
    out: dict[str, dict[str, str]] = {}
    for key_node, val_node in zip(node.keys, node.values, strict=True):
        assert isinstance(val_node, ast.Dict)
        fields = {}
        for k, v in zip(val_node.keys, val_node.values, strict=True):
            if isinstance(v, ast.Constant) and isinstance(v.value, str):
                fields[k.value] = v.value
        out[key_node.value] = fields
    return out


def test_all_110_models_round_trip_and_are_filename_safe() -> None:
    models = _all_models()
    assert len(models) == 110
    keys = [bk.model_key(m) for m in models]
    assert len(set(keys)) == 110
    for model, key in zip(models, keys, strict=True):
        assert re.match(r"^[A-Za-z0-9_.-]+$", key), key
        parsed = bk.parse_model_key(key)
        if "config" in model:
            assert parsed == {"config": model["config"]}
        else:
            assert parsed == model


def test_key_formats_are_the_documented_ones() -> None:
    assert bk.model_key({"config": "III"}) == "config-III"
    assert bk.model_key({"config": "cigale_like"}) == "xlike-cigale_like"
    factorial = bk.enumerate_factorial()[0]
    assert bk.model_key(factorial) == (
        "sfh-continuity__ssp-fsps_mist_c3k_a_chabrier__att-calzetti__ir-dl14__nebular-cue"
    )


@pytest.mark.parametrize("bad", ["config-VI", "xlike-nope", "X-like-cigale_like", "model_A", ""])
def test_parse_rejects_keys_outside_the_space(bad: str) -> None:
    with pytest.raises(ValueError, match="Not a BMA model key"):
        bk.parse_model_key(bad)
    assert bk.set_membership(bad) == set()


def test_model_key_rejects_unknown_config_and_missing_axes() -> None:
    with pytest.raises(ValueError, match="Unknown config id"):
        bk.model_key({"config": "VI"})
    with pytest.raises(ValueError, match="missing axes"):
        bk.model_key({"sfh": "dpl"})


def test_set_membership_sizes_and_overlap() -> None:
    grid = {bk.model_key(m) for m in bk.enumerate_named_grid()}
    xlike = {bk.model_key({"config": k}) for k in bk.XLIKE_IDS}
    factorial = {bk.model_key(m) for m in bk.enumerate_factorial()}
    assert len(grid) == 5 and len(xlike) == 5 and len(factorial) == 100
    assert grid.isdisjoint(xlike) and grid.isdisjoint(factorial) and xlike.isdisjoint(factorial)
    for key in grid:
        assert bk.set_membership(key) == {"named_grid", "named_all"}
    for key in xlike:
        assert bk.set_membership(key) == {"named_all"}
    for key in factorial:
        assert bk.set_membership(key) == {"factorial"}
    assert [len(bk.expected_keys(s)) for s in bk.WEIGHT_SETS] == [5, 10, 100]


def test_factorial_prior_mass_matches_the_flat_prior() -> None:
    prior = bk.factorial_prior_mass()
    assert prior["isochrone"] == pytest.approx({"mist": 0.4, "prsc": 0.4, "bpass": 0.2})
    assert prior["spectral_library"] == pytest.approx({"c3k": 0.6, "miles": 0.4})
    for axis, masses in prior.items():
        assert sum(masses.values()) == pytest.approx(1.0), axis


def test_grid_mappings_match_config_metadata() -> None:
    """Grid ids I-V agree with config_metadata.py (parsed with ast, not imported)."""
    ssp_for_config = ast.literal_eval(_metadata_assignment("SSP_FOR_CONFIG"))
    configs = _configs_from_metadata()
    named = {m["config"]: m for m in bk.enumerate_named_grid()}
    assert tuple(named) == bk.GRID_IDS == ("I", "II", "III", "IV", "V")
    assert "VI" not in named
    for cfg in bk.GRID_IDS:
        meta = configs[cfg]
        assert bk.ssp_grid_for_config(cfg) == ssp_for_config[cfg]
        assert named[cfg]["ssp"] == ssp_for_config[cfg]
        assert named[cfg]["sfh"] == meta["sfh_type"]
        assert named[cfg]["attenuation"] == _ATTENUATION_SLUG[meta["attenuation"]]
        library = meta["library"].removeprefix("FSPS ")
        iso_text, spec_text = library.split("/") if "/" in library else library.split(" ")
        entry = bk.ssp_entry(ssp_for_config[cfg])
        assert entry.config == cfg
        assert entry.isochrone == _ISOCHRONE_SLUG[iso_text]
        assert entry.spectral_library == _LIBRARY_SLUG[spec_text]
    assert [e.grid for e in bk.SSP_AXIS] == [ssp_for_config[c] for c in bk.GRID_IDS]


def test_xlike_ids_match_config_metadata() -> None:
    assert list(bk.XLIKE_IDS) == ast.literal_eval(_metadata_assignment("XLIKE_KEYS"))


_KEY_PREFIX = re.compile(r"^(config|xlike|sfh|ssp|att|ir|nebular|X-like)-")
_KEY_JOINER = re.compile(r"__(sfh|ssp|att|ir|nebular)-")


def _docstring_nodes(tree: ast.AST) -> set[int]:
    ids = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                ids.add(id(body[0].value))
    return ids


def _key_format_literals(path: Path) -> list[tuple[int, str]]:
    tree = ast.parse(path.read_text())
    skip = _docstring_nodes(tree)
    hits = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in skip
            and (_KEY_PREFIX.match(node.value) or _KEY_JOINER.search(node.value))
        ):
            hits.append((node.lineno, node.value))
    return hits


def test_no_module_outside_bma_keys_builds_key_strings() -> None:
    """Sweep: a key-format literal anywhere but ``_bma_keys`` restates the format."""
    offenders = {}
    for path in sorted(PAPER1.glob("*.py")):
        if path.name == "_bma_keys.py":
            continue
        hits = _key_format_literals(path)
        if hits:
            offenders[path.name] = hits
    assert not offenders, f"key-format strings outside _bma_keys: {offenders}"


def test_sweep_detects_a_restated_key_format(tmp_path: Path) -> None:
    """The sweep itself flags each way of spelling a key (not vacuous)."""
    probe = tmp_path / "probe.py"
    probe.write_text(
        'a = f"xlike-{k}"\nb = "config-I"\nc = "__".join(["sfh-x"])\nd = "x__ssp-y"\ne = "X-like-k"\n'
    )
    assert len(_key_format_literals(probe)) == 5


def test_key_and_runner_modules_import_without_jax() -> None:
    code = (
        "import sys\n"
        f"sys.path[:0] = [{str(ANALYSIS)!r}, {str(PAPER1)!r}]\n"
        "import paper1._bma_keys, paper1.bma_combine, paper1.bma_evidence\n"
        "loaded = sorted(m for m in sys.modules if m in ('jax', 'jaxlib', 'tengri'))\n"
        "assert not loaded, loaded\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
