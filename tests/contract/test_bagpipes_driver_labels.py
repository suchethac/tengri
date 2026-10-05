# SPDX-License-Identifier: BSD-3-Clause
"""Contract: the BAGPIPES reproduction pins stars at the same absolute Z as BAGPIPES.

BAGPIPES labels its BC03 metallicity nodes as fractions of Z = 0.02; tengri's
``met_logzsol`` is relative to Z_sun = 0.0142. The driver therefore stores the
nodes as absolute log10(Z), and every stellar-metallicity request in the
notebook and the validator is built from that pin, so both codes read the same
BC03 node. These tests pin the node labels, the speed-of-light constant, the
two ``MET_LOGZSOL`` expressions and the notebook's stellar requests. The grid is
written in BAGPIPES's solar luminosity (3.826e33 erg/s), so the writer stamps
``lsun_erg_per_s`` and ``load_ssp_data`` rescales to the IAU value tengri converts
with; the last tests pin the stamp and the rescale.
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


_REQUEST_KEYS = ("logzsol", "met_logzsol")
_Z_CHECKS = ((1.0, 0.02), (2.5, 0.05))
_NODE_ZS = (0.2, 1.0, 2.5)


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


def _fixed_argument(value: ast.expr) -> ast.expr:
    assert isinstance(value, ast.Call) and value.args, "stellar request is not Fixed(...)"
    return value.args[0]


def _stellar_requests(node: ast.AST, loops: tuple[ast.For, ...] = ()) -> list:
    """``(Fixed argument, enclosing for-loops)`` of every ``logzsol`` request."""
    found = []
    if isinstance(node, ast.Dict):
        for key, value in zip(node.keys, node.values):
            if isinstance(key, ast.Constant) and key.value in _REQUEST_KEYS:
                found.append((_fixed_argument(value), loops))
    if isinstance(node, ast.Call):
        found.extend(
            (_fixed_argument(kw.value), loops) for kw in node.keywords if kw.arg in _REQUEST_KEYS
        )
    inner = (*loops, node) if isinstance(node, ast.For) else loops
    for child in ast.iter_child_nodes(node):
        found.extend(_stellar_requests(child, inner))
    return found


def _loop_source(name: str, loops: tuple[ast.For, ...], assigned: dict) -> ast.AST | None:
    """The list expression a loop variable ``name`` iterates over, by zip position."""
    for loop in loops:
        if not isinstance(loop.target, ast.Tuple) or not isinstance(loop.iter, ast.Call):
            continue
        targets = [t.id for t in loop.target.elts if isinstance(t, ast.Name)]
        if name in targets and len(targets) == len(loop.iter.args):
            source = loop.iter.args[targets.index(name)]
            return assigned[source.id][0] if isinstance(source, ast.Name) else source
    return None


def _request_expression(arg: ast.expr, loops: tuple, assigned: dict) -> ast.expr:
    """The expression that computes the request, in terms of the sweep variable ``z``."""
    assert isinstance(arg, ast.Name), f"unexpected stellar request {ast.unparse(arg)}"
    source = _loop_source(arg.id, loops, assigned)
    if source is None:
        assert arg.id in assigned, f"`{arg.id}` is not assigned in the file"
        source = assigned[arg.id][0]
    return source.elt if isinstance(source, ast.ListComp) else source


def _evaluate_requests(path: Path, zs: tuple[float, ...]) -> list[tuple[str, float, float]]:
    """``(expression, z, value)`` for every stellar request in ``path`` at each ``z``."""
    tree = _tree(path)
    assigned = _assignments(tree)
    z_sun = _evaluate(_single_assignment_value(_DRIVER_SRC, "Z_SUN_BAGPIPES"), {})
    results = []
    for arg, loops in _stellar_requests(tree):
        expr = _request_expression(arg, loops, assigned)
        for z in zs:
            ns = {"np": np, "float": float, "z": z, "LOG10_ZSUN": LOG10_ZSUN}
            ns["Z_SUN_BAGPIPES"] = z_sun
            ns["MET_LOGZSOL"] = (
                _met_logzsol_from_source(path) if "MET_LOGZSOL" in _names(expr) else 0
            )
            results.append((ast.unparse(expr), z, _evaluate(expr, ns)))
    return results


@pytest.mark.parametrize("path", [_NOTEBOOK, _VALIDATOR], ids=lambda p: p.name)
def test_every_stellar_request_evaluates_to_the_absolute_pin(path):
    """Each ``logzsol`` request is log10(0.02 z) - log10(0.0142) at z = 1 and z = 2.5."""
    requests = _evaluate_requests(path, tuple(z for z, _ in _Z_CHECKS))
    assert len(requests) >= 2 * (4 if path == _NOTEBOOK else 1)
    expected = {z: float(np.log10(zabs) - LOG10_ZSUN) for z, zabs in _Z_CHECKS}
    for text, z, value in requests:
        constant = "z" not in {n for n in _names(ast.parse(text, mode="eval"))}
        if constant and z != 1.0:
            continue
        assert value == pytest.approx(expected[z], rel=0, abs=1e-12), f"{text} at z={z}"


def test_stellar_requests_land_on_grid_nodes(driver):
    """For z = 0.2, 1, 2.5 the notebook's request plus log10(0.0142) is a grid node."""
    nodes = driver.absolute_lgmet()
    for text, z, value in _evaluate_requests(_NOTEBOOK, _NODE_ZS):
        distance = np.min(np.abs(nodes - (value + LOG10_ZSUN)))
        assert distance < 1e-9, f"{text} at z={z} is {distance:.2e} dex from a node"


def test_generated_grid_lgmet(driver):
    """If the generated 81 MB grid exists, its ``ssp_lgmet`` equals ``absolute_lgmet()``."""
    if not _DRIVER_GRID.exists():
        pytest.skip(f"generated grid not present at {_DRIVER_GRID}")
    with h5py.File(_DRIVER_GRID, "r") as h:
        stored = np.asarray(h["ssp_lgmet"][:])
    np.testing.assert_allclose(stored, driver.absolute_lgmet(), rtol=0, atol=1e-6)


_LSUN_BAGPIPES = 3.826e33


def _writer_stamp_value() -> ast.expr:
    """The right-hand side of ``h.attrs["lsun_erg_per_s"] = ...`` in ``repackage_bc03_miles``."""
    writer = next(
        n
        for n in _tree(_DRIVER_SRC).body
        if isinstance(n, ast.FunctionDef) and n.name == "repackage_bc03_miles"
    )
    values = [
        n.value
        for n in ast.walk(writer)
        if isinstance(n, ast.Assign)
        and any(
            isinstance(t, ast.Subscript)
            and isinstance(t.slice, ast.Constant)
            and t.slice.value == "lsun_erg_per_s"
            for t in n.targets
        )
    ]
    assert len(values) == 1, "repackage_bc03_miles must stamp lsun_erg_per_s exactly once"
    return values[0]


def test_writer_stamps_bagpipes_lsun(units):
    """The writer stamps ``lsun_erg_per_s`` from ``units.L_SUN_ERG_PER_S`` (3.826e33)."""
    value = _writer_stamp_value()
    assert isinstance(value, ast.Name) and value.id == "L_SUN_ERG_PER_S"
    assert units.L_SUN_ERG_PER_S == _LSUN_BAGPIPES


def test_generated_grid_carries_the_lsun_stamp():
    """If the generated grid exists, it carries ``lsun_erg_per_s`` = 3.826e33 erg/s."""
    if not _DRIVER_GRID.exists():
        pytest.skip(f"generated grid not present at {_DRIVER_GRID}")
    with h5py.File(_DRIVER_GRID, "r") as h:
        stamp = h.attrs.get("lsun_erg_per_s")
    assert stamp is not None, (
        "regenerate the grid: python -m reproduction.bagpipes._drivers.bagpipes_ssp_to_dsps"
    )
    assert float(stamp) == _LSUN_BAGPIPES


def test_stamped_grid_loads_in_bagpipes_lsun():
    """``load_ssp_data`` rescales a stamped grid so ``flux * L_SUN`` is the file's erg/s."""
    from tengri import load_ssp_data
    from tengri.utils.physics_constants import L_SUN

    if not _DRIVER_GRID.exists():
        pytest.skip(f"generated grid not present at {_DRIVER_GRID}")
    with h5py.File(_DRIVER_GRID, "r") as h:
        if "lsun_erg_per_s" not in h.attrs:
            pytest.fail("generated grid carries no lsun_erg_per_s stamp")
        raw = np.asarray(h["ssp_flux"][4, 100, :], dtype=np.float64)
    loaded = np.asarray(load_ssp_data(str(_DRIVER_GRID)).ssp_flux[4, 100, :], dtype=np.float64)
    keep = raw > 0
    np.testing.assert_allclose(
        loaded[keep] * L_SUN / (raw[keep] * _LSUN_BAGPIPES), 1.0, rtol=1e-5, atol=0
    )
