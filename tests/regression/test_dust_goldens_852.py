# SPDX-License-Identifier: BSD-3-Clause
"""Bit-exact goldens for the two dust-emission components the Phase-1 pilot skipped.

The Phase-1 golden capture (#738) skipped ``schreiber2018`` and
``draine2021_pah`` because their template HDF5 grids were absent from the test
env. This module adds their regression protection now that the grids ship
(``data/schreiber2018_templates.h5`` and ``data/pahspec_draine2021.h5``). Every
test **skips cleanly** when its grid is missing, so CI (which does not ship the
gitignored grids) stays green while local / data-present runs exercise them.

Two distinct oracles, because the two templates differ (#852):

* ``schreiber2018`` — the ``DUST_EMISSION_MODELS["schreiber2018"]`` loader is a
  real, independent implementation of the same grid, so the component is pinned
  **bit-exact against the loader** (the strongest check; mirrors the analytic
  ``schreiber2016`` closure-vs-component test).
* ``draine2021_pah_ir`` — its ``DUST_EMISSION_MODELS`` entry is a *deprecated
  alias to ``pah_drude``* (a different, analytic model; #693), so there is no
  independent loader oracle. The component is pinned against a **committed frozen
  golden** plus a physical energy-balance check. This module also regression-
  guards the silent-no-op bug #852 fixed: the component's auto-locate passed a
  ``data/``-prefixed path to ``_find_data_file`` (which prepends its own
  ``/data`` candidate dirs), so it never found the present grid and silently
  emitted zeros.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.dust.emission import (
    DUST_EMISSION_MODELS,
    preload_emission_model,
)
from tengri.components.sed_model_component import _REGISTRY
from tengri.utils.physics_constants import C_AA

pytestmark = pytest.mark.regression_bug

# Same representative grid + absorbed luminosity as the pilot goldens.
_WAVE = jnp.linspace(1000.0, 1.0e7, 512)
_L_IR = 1.0e44

_GOLDEN_DIR = Path(__file__).parent / "data" / "dust_emission_golden"

# Cross-JAX-version XLA drift on JIT-compiled SED goldens is measured at ~1e-7
# relative (#1763: 9.97e-08 on the PAH golden, jax 0.9.1 -> 0.11.0). Goldens
# pinned tighter than that test the compiler version, not the physics; a real
# regression and a JAX upgrade become indistinguishable, and the natural
# response (re-freeze) would silently absorb a real regression.
GOLDEN_SED_RTOL = 1e-6  # ~10x margin over measured drift; ~1e5 below physics changes
GOLDEN_SED_ATOL = 1e-7


def _schreiber2018_available() -> bool:
    """Gate on the grid FILE directly. ``preload_emission_model`` is lazy (does
    not raise on a missing grid); a trial ``loader(...)`` call would fail *and*
    leave the schreiber2018 lazy-loader entry in an inconsistent state that
    pollutes later tests (e.g. ``test_dust_energy_balance``). Checking the file
    up front skips cleanly without touching the loader."""
    from tengri._data_setup import find_data_str

    return find_data_str("schreiber2018_templates.h5") is not None


def _draine2021_available() -> bool:
    """The component loads the PAHspec grid via auto-locate; data present iff load()
    returns a template dict."""
    comp = _REGISTRY["draine2021_pah_ir"]()
    try:
        return comp.load(_WAVE) is not None
    except Exception:
        return False


# ── schreiber2018 — bit-exact vs the independent loader ───────────────


@pytest.mark.skipif(
    not _schreiber2018_available(), reason="schreiber2018 template grid not available"
)
def test_schreiber2018_port_matches_loader_bit_exact():
    """The schreiber2018 SEDModelComponent reproduces the
    ``DUST_EMISSION_MODELS`` loader exactly on the shipped grid. Both the loader
    and the component now use the canonical ``dust_T`` / ``dust_f_pah`` names (#849),
    stripping to ``T`` / ``f_pah`` on the component."""
    preload_emission_model("schreiber2018")
    loader = DUST_EMISSION_MODELS["schreiber2018"]
    golden = np.asarray(loader(_WAVE, _L_IR, dust_T=30.0, dust_f_pah=0.05), dtype=np.float64)

    comp = _REGISTRY["schreiber2018"]()
    sed_out, _ = comp.predict({"T": 30.0, "f_pah": 0.05}, jnp.zeros_like(_WAVE), _WAVE, L_ir=_L_IR)
    # Same-process loader comparison, not a frozen golden — exempt from GOLDEN_SED_RTOL (#1763).
    np.testing.assert_allclose(np.asarray(sed_out), golden, rtol=1e-14, atol=1e-15)


# ── draine2021_pah_ir — frozen golden + energy balance + no-op guard ──


def _draine_port_sed() -> np.ndarray:
    comp = _REGISTRY["draine2021_pah_ir"]()
    comp.data = comp.load(_WAVE)
    sed_out, _ = comp.predict({"lgU": 1.0}, jnp.zeros_like(_WAVE), _WAVE, L_ir=_L_IR)
    return np.asarray(sed_out, dtype=np.float64)


@pytest.mark.skipif(
    not _draine2021_available(), reason="draine2021 PAHspec template grid not available"
)
def test_draine2021_pah_is_not_a_silent_no_op():
    """Regression for the #852 silent-no-op fix: with the grid present, the component
    must load it and emit a nonzero far-IR SED (before the fix, the
    ``data/``-prefixed auto-locate path missed and the component returned zeros)."""
    comp = _REGISTRY["draine2021_pah_ir"]()
    assert comp.load(_WAVE) is not None, "PAHspec grid present but load() returned None"
    sed = _draine_port_sed()
    assert np.all(np.isfinite(sed))
    assert np.nansum(np.abs(sed)) > 0.0, "component emitted all zeros despite present grid"


@pytest.mark.skipif(
    not _draine2021_available(), reason="draine2021 PAHspec template grid not available"
)
def test_draine2021_pah_energy_balance():
    """The emitted IR SED re-radiates the absorbed luminosity: the frequency
    integral of L_nu recovers L_ir to within a few percent (grid-edge losses)."""
    sed = _draine_port_sed()
    nu = C_AA / np.asarray(_WAVE)
    l_emitted = float(-np.trapezoid(sed, nu))
    assert l_emitted == pytest.approx(_L_IR, rel=0.05)
    # Peaks in the IR (not UV) — a dust re-emission sanity check.
    peak_wave = float(np.asarray(_WAVE)[int(np.argmax(sed))])
    assert peak_wave > 1.0e5


@pytest.mark.skipif(
    not _draine2021_available(), reason="draine2021 PAHspec template grid not available"
)
def test_draine2021_pah_matches_frozen_golden():
    """The component reproduces the committed frozen golden bit-exactly — a drift
    lock (no independent loader oracle exists; the DUST_EMISSION_MODELS entry is
    a deprecated pah_drude alias, #693)."""
    golden_npy = _GOLDEN_DIR / "draine2021_pah_ir.npy"
    if not golden_npy.exists():
        pytest.skip(f"golden not found: {golden_npy}")
    golden = np.load(golden_npy)
    sed = _draine_port_sed()
    np.testing.assert_allclose(sed, golden, rtol=GOLDEN_SED_RTOL, atol=GOLDEN_SED_ATOL)


# ── astrodust — frozen golden + energy balance (#871) ─────────────────
#
# The faithful Hensley & Draine 2023 astrodust component (native lgU interpolation of
# the published emission grid) has no independent faithful loader oracle: the
# ``DUST_EMISSION_MODELS["astrodust"]`` entry is the retired DL07-*costume*
# (umin/gamma/qpah over an HD23→DL07-translated grid, with a no-op ``dust_qpah``;
# #871). So — as with draine2021_pah_ir — the component is pinned against a committed
# frozen golden plus a physical energy-balance check.


def _astrodust_available() -> bool:
    """Data present iff the component's ``load()`` returns a template dict. The component
    *raises* FileNotFoundError on a missing grid (no analytic fallback), so the
    try/except is required to skip cleanly."""
    comp = _REGISTRY["astrodust"]()
    try:
        return comp.load(_WAVE) is not None
    except Exception:
        return False


def _astrodust_port_sed() -> np.ndarray:
    comp = _REGISTRY["astrodust"]()  # default config: component="total", no spinning
    comp.data = comp.load(_WAVE)
    sed_out, _ = comp.predict({"lgU": 1.0}, jnp.zeros_like(_WAVE), _WAVE, L_ir=_L_IR)
    return np.asarray(sed_out, dtype=np.float64)


@pytest.mark.skipif(not _astrodust_available(), reason="astrodust template grid not available")
def test_astrodust_is_not_a_silent_no_op():
    """With the grid present the native implementation must load it and emit a nonzero
    far-IR SED (guards against a silent-zeros regression like #852)."""
    comp = _REGISTRY["astrodust"]()
    assert comp.load(_WAVE) is not None, "astrodust grid present but load() returned None"
    sed = _astrodust_port_sed()
    assert np.all(np.isfinite(sed))
    assert np.nansum(np.abs(sed)) > 0.0, "component emitted all zeros despite present grid"


@pytest.mark.skipif(not _astrodust_available(), reason="astrodust template grid not available")
def test_astrodust_energy_balance():
    """The emitted IR SED re-radiates the absorbed luminosity: the frequency
    integral of L_nu recovers L_ir to within a few percent (grid-edge losses),
    and peaks in the IR (not UV)."""
    sed = _astrodust_port_sed()
    nu = C_AA / np.asarray(_WAVE)
    l_emitted = float(-np.trapezoid(sed, nu))
    assert l_emitted == pytest.approx(_L_IR, rel=0.05)
    peak_wave = float(np.asarray(_WAVE)[int(np.argmax(sed))])
    assert peak_wave > 1.0e5


@pytest.mark.skipif(not _astrodust_available(), reason="astrodust template grid not available")
def test_astrodust_matches_frozen_golden():
    """The native lgU component reproduces the committed frozen golden bit-exactly —
    a drift lock (the DUST_EMISSION_MODELS entry is the retired DL07-costume,
    #871)."""
    golden_npy = _GOLDEN_DIR / "astrodust.npy"
    if not golden_npy.exists():
        pytest.skip(f"golden not found: {golden_npy}")
    golden = np.load(golden_npy)
    sed = _astrodust_port_sed()
    np.testing.assert_allclose(sed, golden, rtol=GOLDEN_SED_RTOL, atol=GOLDEN_SED_ATOL)
