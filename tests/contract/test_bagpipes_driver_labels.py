# SPDX-License-Identifier: BSD-3-Clause
"""Contract: the BAGPIPES reproduction pins stars at the same absolute Z as BAGPIPES.

BAGPIPES labels its BC03 metallicity nodes as fractions of Z = 0.02; tengri's
``met_logzsol`` is relative to Z_sun = 0.0142. The driver therefore stores the
nodes as absolute log10(Z), and every stellar-metallicity request in the
notebook and the validator is built from that pin, so both codes read the same
BC03 node. These tests pin the node labels, the speed-of-light constant, the
two ``MET_LOGZSOL`` expressions and the notebook's stellar requests.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import h5py
import numpy as np
import pytest

from tengri._data_setup import find_data
from tengri.utils.physics_constants import C_AA, LOG10_ZSUN

pytestmark = pytest.mark.contract

_REPRO = Path(__file__).resolve().parents[2] / "reproduction" / "bagpipes"
_NOTEBOOK = _REPRO / "01_bagpipes.py"
_VALIDATOR = _REPRO / "validate_matched_physics.py"
_DRIVER_SRC = _REPRO / "_drivers" / "bagpipes_ssp_to_dsps.py"
_DRIVER_GRID = _REPRO / "_drivers" / "data" / "bc03_miles_from_bagpipes.h5"

_BC03_ABSOLUTE_Z = (1e-4, 4e-4, 4e-3, 8e-3, 0.02, 0.05, 0.1)
_PIN_EXPECTED = float(np.log10(0.02) - LOG10_ZSUN)
_NATIVE_SSP = "bc03_pdva_stelib_chabrier.h5"
_PIN_NAMES = ("Z_SUN_BAGPIPES", "LOG10_ZSUN")


@pytest.fixture(scope="module")
def driver():
    """The SSP-relabeling driver (imports astropy, absent from the CI test extra)."""
    pytest.importorskip("astropy")
    return importlib.import_module("reproduction.bagpipes._drivers.bagpipes_ssp_to_dsps")


@pytest.fixture(scope="module")
def units():
    return importlib.import_module("reproduction.bagpipes._drivers.units")


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text())


def _targets(node: ast.Assign | ast.AnnAssign, name: str) -> bool:
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    return any(isinstance(t, ast.Name) and t.id == name for t in targets)


def _single_assignment_value(path: Path, name: str) -> ast.expr:
    """The value of the one module-level assignment to ``name`` in ``path``."""
    values = [
        node.value
        for node in _tree(path).body
        if (isinstance(node, ast.Assign) and _targets(node, name))
        or (isinstance(node, ast.AnnAssign) and _targets(node, name) and node.value)
    ]
    assert len(values) == 1, f"expected exactly one `{name} =` in {path.name}, found {len(values)}"
    return values[0]


def _evaluate(expr: ast.expr, namespace: dict) -> float:
    code = compile(ast.Expression(expr), "<contract>", "eval")
    return float(eval(code, {"__builtins__": {}}, namespace))


def _met_logzsol_from_source(path: Path) -> float:
    """Evaluate the file's ``MET_LOGZSOL`` against the driver's literal Z_SUN_BAGPIPES."""
    z_sun = _evaluate(_single_assignment_value(_DRIVER_SRC, "Z_SUN_BAGPIPES"), {})
    namespace = {"np": np, "Z_SUN_BAGPIPES": z_sun, "LOG10_ZSUN": LOG10_ZSUN}
    return _evaluate(_single_assignment_value(path, "MET_LOGZSOL"), namespace)


@pytest.mark.parametrize("index", range(len(_BC03_ABSOLUTE_Z)))
def test_absolute_lgmet_matches_bc03_nodes(driver, index):
    """Node ``index`` of ``absolute_lgmet()`` is log10 of the BC03 absolute Z."""
    result = driver.absolute_lgmet()
    assert len(result) == len(_BC03_ABSOLUTE_Z)
    np.testing.assert_allclose(
        result[index], np.log10(_BC03_ABSOLUTE_Z[index]), rtol=0, atol=1e-12
    )


def test_native_bc03_grid_matches_pins(driver):
    """The six nodes of tengri's native BC03 grid are the driver's first six labels."""
    ssp_file = find_data(_NATIVE_SSP)
    if ssp_file is None:
        pytest.skip(f"{_NATIVE_SSP} not found in any tengri data directory")
    with h5py.File(ssp_file, "r") as h:
        native_lgmet = np.asarray(h["ssp_lgmet"][:])
    np.testing.assert_allclose(native_lgmet, driver.absolute_lgmet()[:6], rtol=0, atol=1e-4)


def test_driver_speed_of_light_is_tengris(units):
    """``units.C_ANGSTROM_PER_S`` is tengri's ``C_AA``."""
    assert units.C_ANGSTROM_PER_S == C_AA


@pytest.mark.parametrize("path", [_NOTEBOOK, _VALIDATOR], ids=lambda p: p.name)
def test_met_logzsol_is_the_absolute_pin(path):
    """``MET_LOGZSOL`` is the literal log10(0.02) - log10(0.0142), and appears once."""
    assert _met_logzsol_from_source(path) == pytest.approx(_PIN_EXPECTED, rel=0, abs=1e-12)


def test_driver_z_sun_is_the_bc03_solar_literal():
    """The driver's ``Z_SUN_BAGPIPES`` is the literal 0.02."""
    z_sun = _evaluate(_single_assignment_value(_DRIVER_SRC, "Z_SUN_BAGPIPES"), {})
    assert z_sun == pytest.approx(0.02, rel=0, abs=1e-15)


def _names(node: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def _assignments(tree: ast.Module) -> dict[str, list[ast.AST]]:
    """Map each assigned name to the value expressions it is assigned from."""
    assigned: dict[str, list[ast.AST]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                for n in ast.walk(target):
                    if isinstance(n, ast.Name):
                        assigned.setdefault(n.id, []).append(node.value)
    return assigned


def _logzsol_requests(node: ast.AST, loops: tuple[ast.For, ...] = ()) -> list:
    """``(Fixed argument, enclosing for-loops)`` of every ``"logzsol": Fixed(...)`` entry."""
    found = []
    if isinstance(node, ast.Dict):
        for key, value in zip(node.keys, node.values):
            if isinstance(key, ast.Constant) and key.value == "logzsol":
                assert isinstance(value, ast.Call) and value.args, "logzsol is not Fixed(...)"
                found.append((value.args[0], loops))
    inner = (*loops, node) if isinstance(node, ast.For) else loops
    for child in ast.iter_child_nodes(node):
        found.extend(_logzsol_requests(child, inner))
    return found


def _carries_pin(node: ast.AST) -> bool:
    return set(_PIN_NAMES) <= _names(node)


def _is_pin_built(name: str, loops: tuple[ast.For, ...], assigned: dict) -> bool:
    """True if ``name`` is the pin, assigned from it, or a loop variable over such a list."""
    if name == "MET_LOGZSOL":
        return True
    if any(_carries_pin(v) for v in assigned.get(name, [])):
        return True
    for loop in loops:
        if name in _names(loop.target):
            return any(_carries_pin(v) for src in _names(loop.iter) for v in assigned.get(src, []))
    return False


def test_every_notebook_stellar_request_is_built_from_the_pin():
    """No ``logzsol`` request in the notebook is a bare gas-style ``log10(z)``."""
    tree = _tree(_NOTEBOOK)
    assigned = _assignments(tree)
    requests = _logzsol_requests(tree)
    assert len(requests) >= 4, f"expected >= 4 logzsol requests, found {len(requests)}"
    for arg, loops in requests:
        assert isinstance(arg, ast.Name), f"unexpected logzsol request {ast.unparse(arg)}"
        assert _is_pin_built(arg.id, loops, assigned), (
            f"`{arg.id}` feeds a stellar logzsol request but is not built from "
            "Z_SUN_BAGPIPES and LOG10_ZSUN"
        )


def test_generated_grid_lgmet(driver):
    """If the generated 81 MB grid exists, its ``ssp_lgmet`` equals ``absolute_lgmet()``."""
    if not _DRIVER_GRID.exists():
        pytest.skip(f"generated grid not present at {_DRIVER_GRID}")
    with h5py.File(_DRIVER_GRID, "r") as h:
        stored = np.asarray(h["ssp_lgmet"][:])
    np.testing.assert_allclose(stored, driver.absolute_lgmet(), rtol=0, atol=1e-6)
