# SPDX-License-Identifier: BSD-3-Clause
"""Cross-validate tengri's ``schreiber2018`` cold-dust model against AGNfitter-rX.

The Schreiber et al. (2018) "S17" cold-dust library (dust continuum + PAH) was
repackaged from AGNfitter-rX's ``STARBURST/s17_lowvsg_{dust,pah}.fits`` into
tengri's HDF5 grid by ``scripts/build_schreiber2018_grid.py``. AGNfitter-rX
forms the cold-dust SED as the native mixture
``(1 - f_PAH)·dust + f_PAH·PAH`` (``MODEL_AGNfitter.STARBURST`` S17 branch).

This test reads the committed reference (``data/agnfitter_cold_dust_reference.h5``
group ``s17``, built by ``scripts/build_agnfitter_s17_reference.py`` — the raw
FITS arrays at their native 721-point resolution, no resampling), reconstructs
AGNfitter-rX's mixture at several (T_dust, f_PAH) nodes, and verifies tengri's
``schreiber2018`` reproduces the same *shape* (normalized L_nu). No live
upstream checkout is read: every AGNfitter parity test reads committed data
only (``tests/contract/test_reproduction_driver_no_clone.py``). This test used
to hardcode an absolute path into a local AGNfitter-rX clone's
``models/STARBURST/*.fits`` and skip at module level whenever that clone was
absent — it now collects and runs on any clean checkout (census.md D6).
"""

from __future__ import annotations

from pathlib import Path

import h5py
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = [pytest.mark.crossval, pytest.mark.regression_paper]

_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
_GRID_PATH = _DATA_DIR / "schreiber2018_templates.h5"
_REFERENCE_PATH = _DATA_DIR / "agnfitter_cold_dust_reference.h5"

_C_AA_PER_S = 2.99792458e18  # speed of light [Å·Hz]

if not _GRID_PATH.is_file():
    pytest.skip(
        "Schreiber+2018 grid not found at "
        + str(_GRID_PATH)
        + " (build with: python scripts/build_schreiber2018_grid.py)",
        allow_module_level=True,
    )

if not _REFERENCE_PATH.is_file():
    pytest.skip(
        "AGNfitter reference data not found at " + str(_REFERENCE_PATH),
        allow_module_level=True,
    )


def _load_reference_s17() -> dict:
    """Load the committed S17 reference: native 721-point FITS grid, both tables.

    Returns
    -------
    dict
        ``tdust`` (150,) [K] ascending, ``wave_aa`` (721,) [Å] ascending
        (the dust and PAH tables share one native grid — asserted below),
        ``dust_lnu`` (150, 721), ``pah_lnu`` (150, 721) [L_nu, relative
        units, ascending in ``wave_aa``].
    """
    with h5py.File(_REFERENCE_PATH, "r") as f:
        g = f["s17"]
        tdust = np.array(g["tdust"][:], dtype=np.float64)
        dust_lam_um = np.array(g["dust_lam_um"][:], dtype=np.float64)
        pah_lam_um = np.array(g["pah_lam_um"][:], dtype=np.float64)
        dust_nulnu = np.array(g["dust_sed_nulnu"][:], dtype=np.float64)
        pah_nulnu = np.array(g["pah_sed_nulnu"][:], dtype=np.float64)

    assert np.allclose(pah_lam_um, dust_lam_um), (
        "S17 dust and PAH reference tables have diverged native wavelength grids"
    )
    wave_um = dust_lam_um[0]
    wave_aa = wave_um * 1.0e4
    order = np.argsort(wave_aa)
    wave_aa = wave_aa[order]
    nu_hz = _C_AA_PER_S / wave_aa
    return {
        "tdust": tdust,
        "wave_aa": wave_aa,
        "dust_lnu": dust_nulnu[:, order] / nu_hz,
        "pah_lnu": pah_nulnu[:, order] / nu_hz,
    }


def _agnfitter_s17_mix(ref: dict, t_idx: int, fpah: float) -> np.ndarray:
    """AGNfitter-rX's S17 mixture at reference row ``t_idx`` (nearest-node selection)."""
    return (1.0 - fpah) * ref["dust_lnu"][t_idx] + fpah * ref["pah_lnu"][t_idx]


def _tengri_s17(wave_aa: np.ndarray, tdust: float, fpah: float) -> np.ndarray:
    """tengri ``schreiber2018`` L_nu on the requested grid."""
    from tengri.components.dust.emission_templates import create_schreiber2018_from_grid

    fn = create_schreiber2018_from_grid(str(_GRID_PATH))
    return np.asarray(fn(jnp.asarray(wave_aa), 1.0, dust_T=tdust, dust_f_pah=fpah))


_REF = _load_reference_s17()
_N_TDUST = _REF["tdust"].size
#: Three well-separated T_dust NODES (not off-node values): low, mid, high.
#: Querying tengri at the exact tabulated T isolates the mixture/PAH shape
#: fidelity this test targets from the (expected, unrelated) small
#: interpolated-vs-nearest-node difference that appears at an off-node T —
#: tengri linearly interpolates in T, AGNfitter-rX selects the nearest node.
_NODE_IDXS = (5, _N_TDUST // 2, _N_TDUST - 6)


@pytest.mark.parametrize(
    "tdust,fpah",
    [(20.0, 0.01), (30.0, 0.02), (45.0, 0.05), (55.0, 0.03)],
)
def test_schreiber2018_matches_agnfitter_shape(tdust: float, fpah: float) -> None:
    """tengri schreiber2018 reproduces AGNfitter-rX S17 shape to <3% of peak."""
    t_idx = int(np.argmin(np.abs(_REF["tdust"] - tdust)))
    l_af = _agnfitter_s17_mix(_REF, t_idx, fpah)
    w_af = _REF["wave_aa"]
    l_te = _tengri_s17(w_af, tdust, fpah)

    band = (w_af > 3.0e4) & (w_af < 3.0e6)  # 3–300 µm: where cold dust lives
    af_n = l_af / l_af[band].max()
    te_n = l_te / l_te[band].max()
    resid = np.abs(te_n[band] - af_n[band])

    assert np.all(np.isfinite(l_te))
    assert np.median(resid) < 5.0e-3, f"median |Δ|/peak = {np.median(resid):.2e}"
    assert resid.max() < 3.0e-2, f"max |Δ|/peak = {resid.max():.2e}"


@pytest.mark.parametrize("tdust,expect_um", [(20.0, 130.0), (35.0, 87.0), (55.0, 56.0)])
def test_schreiber2018_peak_tracks_temperature(tdust: float, expect_um: float) -> None:
    """FIR peak shifts blueward with T_dust (Wien), matching AGNfitter-rX S17."""
    t_idx = int(np.argmin(np.abs(_REF["tdust"] - tdust)))
    l_af = _agnfitter_s17_mix(_REF, t_idx, 0.0)
    w_af = _REF["wave_aa"]
    af_peak_um = w_af[np.argmax(l_af)] / 1.0e4
    wave = np.geomspace(1.0e4, 1.0e7, 1500)
    l_te = _tengri_s17(wave, tdust, 0.0)
    te_peak_um = wave[np.argmax(l_te)] / 1.0e4
    # tengri peak within 8% of AGNfitter's (regrid + node-linear vs node-nearest).
    assert abs(te_peak_um - af_peak_um) / af_peak_um < 0.08


def test_schreiber2018_energy_balance() -> None:
    """The frequency integral of the emitted L_nu equals L_absorbed."""
    wave = np.geomspace(1.0e4, 1.0e8, 4000)
    l_te = _tengri_s17(wave, 35.0, 0.03)
    nu = _C_AA_PER_S / wave
    integral = -np.trapezoid(l_te, nu)
    assert abs(integral - 1.0) < 1.0e-2


def test_schreiber2018_pah_grows_with_fraction() -> None:
    """Mid-IR PAH-band power increases monotonically with f_PAH."""
    wave = np.geomspace(1.0e4, 1.0e8, 3000)
    pah_band = (wave > 6.0e4) & (wave < 1.3e5)  # 6–13 µm
    prev = -np.inf
    for fpah in (0.0, 0.02, 0.05):
        l_te = _tengri_s17(wave, 30.0, fpah)
        power = float(l_te[pah_band].max())
        assert power >= prev
        prev = power


_PAH_FEATURES_UM = (3.3, 6.2, 7.7, 8.6, 11.3, 12.7)


@pytest.mark.parametrize(
    "t_idx,fpah", [(_NODE_IDXS[0], 0.02), (_NODE_IDXS[1], 0.03), (_NODE_IDXS[2], 0.04)]
)
def test_schreiber2018_matches_agnfitter_pah(t_idx: int, fpah: float) -> None:
    """PAH-feature amplitude fidelity, node-exact on the reference's native grid.

    Regression test for D1 (colddust_radio.md): the prior builder resampled
    both the dust-continuum and PAH tables onto a shared 1024-point
    ``np.geomspace`` grid, which smeared the ~0.05 µm-scale 3.3 µm PAH
    feature by up to 32%. ``build_schreiber2018_grid.py`` now stores the
    native 721-point grid verbatim, so tengri's mixture, evaluated ON that
    same native grid at an exact ``tdust`` node (no resampling, no
    node-interpolation mismatch), must reproduce AGNfitter-rX's mixture to a
    tight log10-ratio tolerance at every PAH-complex wavelength and across
    the whole cold-dust band.
    """
    tdust = float(_REF["tdust"][t_idx])
    l_af = _agnfitter_s17_mix(_REF, t_idx, fpah)
    w_af = _REF["wave_aa"]
    l_te = _tengri_s17(w_af, tdust, fpah)
    assert np.all(np.isfinite(l_te))

    i100 = np.argmin(np.abs(w_af / 1.0e4 - 100.0))
    scale = l_af[i100] / l_te[i100]
    band = (w_af > 3.0e4) & (w_af < 1.0e7)  # 3-1000 um
    with np.errstate(divide="ignore", invalid="ignore"):
        logratio = np.log10(np.abs((l_te * scale) / l_af))

    for feature_um in _PAH_FEATURES_UM:
        idx = np.argmin(np.abs(w_af / 1.0e4 - feature_um))
        assert abs(logratio[idx]) < 0.005, (
            f"T={tdust:.2f}, f_PAH={fpah}: {feature_um} um max|log10 ratio| = "
            f"{abs(logratio[idx]):.4f} (expect <0.005)"
        )

    band_logratio = logratio[band]
    band_logratio = band_logratio[np.isfinite(band_logratio)]
    worst = float(np.max(np.abs(band_logratio)))
    assert worst < 0.002, (
        f"T={tdust:.2f}, f_PAH={fpah}: max|log10 ratio| over 3-1000 um = "
        f"{worst:.4f} (expect <0.002)"
    )
