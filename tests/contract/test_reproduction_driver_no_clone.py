# SPDX-License-Identifier: BSD-3-Clause
"""Contract: the AGNfitter reproduction driver reads no upstream checkout.

Every AGNFITTER-RX reference template is repackaged into committed HDF5 under
``data/``, so ``reproduction/agnfitter/`` runs on a clean clone and in CI. The
contract has been broken twice by accessors that quietly kept reading
``$AGNFITTER_HOME/models`` and therefore only worked on a machine where a stale
checkout happened to survive:

* ``_s17_tables`` read the two S17 FITS tables from the checkout (fixed in
  #1035, which vendored them into the cold-dust h5).
* ``cold_dust_axes("DH02_CE01")`` read ``DH02_CE01.pickle`` — missed by the
  #792 h5 consolidation, which moved ``cold_dust_template`` onto the h5 but not
  its axes sibling. Dead code at the time, so no notebook caught it.

Both were invisible locally and would have failed for any new user. This test
exercises every public accessor with the module's data directory pointed at a
path that cannot exist, so any re-introduced checkout read fails loudly here
rather than silently on someone else's machine.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.contract

_NO_SUCH_PATH = Path("/nonexistent-agnfitter-checkout")


@pytest.fixture
def driver():
    """The AGNfitter driver, or a skip when the committed grids are absent."""
    agnfitter_driver = pytest.importorskip(
        "reproduction.agnfitter._drivers.agnfitter_driver",
        reason="reproduction package not importable",
    )
    if not agnfitter_driver.available():
        pytest.skip("committed AGNfitter reference grids are not present in data/")
    return agnfitter_driver


def _assert_template(wave_aa, l_nu, name: str) -> None:
    """A template must be a finite, positive, ascending (wave, L_nu) pair."""
    w, lum = np.asarray(wave_aa), np.asarray(l_nu)
    assert w.shape == lum.shape and w.size > 1, f"{name}: degenerate template"
    assert np.all(np.isfinite(w)) and np.all(np.isfinite(lum)), f"{name}: non-finite"
    assert np.all(np.diff(w) > 0), f"{name}: wavelengths not ascending"
    assert np.nanmax(lum) > 0.0, f"{name}: all-zero luminosity"


def test_no_module_level_checkout_path(driver):
    """No module attribute may point into an AGNFITTER-RX checkout.

    ``AGNFITTER_HOME`` / ``_MODELS`` were the seam both regressions reached
    through; the driver must not carry a checkout path at all.
    """
    leaked = [
        attr for attr in ("AGNFITTER_HOME", "_MODELS") if getattr(driver, attr, None) is not None
    ]
    assert not leaked, f"driver still exposes checkout path(s): {leaked}"


class TestAccessorsRunWithoutCheckout:
    """Every public accessor must resolve from the committed h5 alone."""

    @pytest.fixture(autouse=True)
    def _no_checkout(self, driver, monkeypatch):
        """Point any surviving checkout seam at a path that cannot exist."""
        for attr in ("AGNFITTER_HOME", "_MODELS"):
            if hasattr(driver, attr):
                monkeypatch.setattr(driver, attr, _NO_SUCH_PATH, raising=False)
        driver._ref.cache_clear()
        driver._s17_tables.cache_clear()
        yield
        driver._ref.cache_clear()
        driver._s17_tables.cache_clear()

    def test_disk_templates(self, driver):
        for name in driver.list_disks():
            _assert_template(*driver.disk_template(name), name=f"disk {name}")

    def test_torus_templates(self, driver):
        for name in driver.list_tori():
            _assert_template(*driver.torus_template(name), name=f"torus {name}")

    def test_cold_dust_templates(self, driver):
        for name in driver.list_cold_dust():
            _assert_template(*driver.cold_dust_template(name), name=f"cold dust {name}")

    def test_cold_dust_axes(self, driver):
        """The regression this test was written for: DH02_CE01's axes.

        ``cold_dust_template("DH02_CE01")`` read the h5 while
        ``cold_dust_axes("DH02_CE01")`` still unpickled from the checkout.
        """
        for name in driver.list_cold_dust():
            axes = driver.cold_dust_axes(name)
            assert axes, f"cold dust {name}: no axes returned"
            for axis, values in axes.items():
                v = np.asarray(values)
                assert v.size > 0, f"cold dust {name}: empty axis {axis!r}"
                assert np.all(np.isfinite(v)), f"cold dust {name}: non-finite axis {axis!r}"

    def test_dh02_axis_matches_template_grid(self, driver):
        """The h5 axis must be the one ``cold_dust_template`` selects on.

        Guards against reading a same-named-but-different array: every axis
        value must be selectable, and neighboring grid points must give
        different templates.
        """
        axis = np.asarray(driver.cold_dust_axes("DH02_CE01")["log_irlum"])
        lo, hi = float(axis[0]), float(axis[-1])
        _, l_lo = driver.cold_dust_template("DH02_CE01", log_irlum=lo)
        _, l_hi = driver.cold_dust_template("DH02_CE01", log_irlum=hi)
        assert not np.allclose(l_lo, l_hi, rtol=1e-6, atol=0.0), (
            "DH02_CE01 endpoints of the reported axis give identical templates — "
            "the axis is not the grid the template selector indexes"
        )


_CROSSVAL_DIR = Path(__file__).resolve().parents[1] / "crossval"


def _string_constants_containing(source: str, needle: str) -> list[str]:
    """String literals (and f-string literal segments) in ``source`` containing ``needle``.

    Parses via :mod:`ast` rather than matching lines of text, so a comment
    that merely *mentions* ``needle`` (e.g. "# formerly read from /tmp/...")
    is not a hit -- comments are stripped before the AST exists at all -- and
    only an actual string constant a running test could pass to ``open()``
    or ``Path()`` counts (task3 fix round 1, item 6: the line-based version
    this replaces matched comments and docstrings alike).

    Parameters
    ----------
    source : str
        Python source text.
    needle : str
        Substring to search for inside each string constant.

    Returns
    -------
    list of str
        Every matching string constant's value, in source order.
    """
    tree = ast.parse(source)
    hits: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if needle in node.value:
                hits.append(node.value)
        elif isinstance(node, ast.JoinedStr):  # f-string
            for value in node.values:
                if (
                    isinstance(value, ast.Constant)
                    and isinstance(value.value, str)
                    and needle in value.value
                ):
                    hits.append(value.value)
    return hits


def test_agnfitter_crossval_tests_read_no_tmp_clone() -> None:
    """Every ``test_*_vs_agnfitter.py`` file reads committed data only.

    ``test_schreiber2018_vs_agnfitter.py`` used to hardcode
    ``Path("/tmp/AGNfitter-rX/models/STARBURST/...")`` and skip at module
    level whenever that local clone was absent — the only one of the
    ``test_*_vs_agnfitter.py`` files that could not collect on a clean
    checkout (census.md D6; the other nine already read only committed
    ``data/*.h5``). This is a static guard, not a runtime one (unlike the
    accessor-level class above): it inspects every AGNfitter parity test
    file's string constants (via :func:`_string_constants_containing`) for
    the literal ``/tmp/`` substring, so a future test cannot reintroduce a
    live-clone path even somewhere a narrower "no such import" check would
    miss it (a hardcoded string literal, not an import).
    """
    crossval_files = sorted(_CROSSVAL_DIR.glob("test_*_vs_agnfitter.py"))
    assert crossval_files, f"no test_*_vs_agnfitter.py files found under {_CROSSVAL_DIR}"

    offenders: dict[str, list[str]] = {}
    for path in crossval_files:
        hits = _string_constants_containing(path.read_text(), "/tmp/")
        if hits:
            offenders[path.name] = hits

    assert not offenders, (
        "the following AGNfitter parity test file(s) contain a literal "
        "'/tmp/' path -- they must read only committed data/*.h5, never a "
        f"live upstream clone: {offenders}"
    )
