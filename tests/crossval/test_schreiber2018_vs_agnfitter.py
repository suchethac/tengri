# SPDX-License-Identifier: BSD-3-Clause
"""Cross-validate tengri's ``schreiber2018`` cold-dust model against AGNfitter-rX.

The Schreiber et al. (2018) "S17" cold-dust library (dust continuum + PAH) was
repackaged from AGNfitter-rX's ``STARBURST/s17_lowvsg_{dust,pah}.fits`` into
tengri's HDF5 grid by ``scripts/build_schreiber2018_grid.py``. AGNfitter-rX
forms the cold-dust SED as the native mixture
``(1 - f_PAH)·dust + f_PAH·PAH`` (``MODEL_AGNfitter.STARBURST`` S17 branch).

This test reads the original FITS tables directly, reconstructs AGNfitter-rX's
mixture at several (T_dust, f_PAH) nodes, and verifies tengri's
``schreiber2018`` reproduces the same *shape* (normalized L_nu) to a tight
tolerance after regridding to a common wavelength axis.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = [pytest.mark.crossval, pytest.mark.regression_paper]

_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
_GRID_PATH = _DATA_DIR / "schreiber2018_templates.h5"
_S17_DUST = Path("/tmp/AGNfitter-rX/models/STARBURST/s17_lowvsg_dust.fits")
_S17_PAH = Path("/tmp/AGNfitter-rX/models/STARBURST/s17_lowvsg_pah.fits")

_C_AA_PER_S = 2.99792458e18  # speed of light [Å·Hz]

if not _GRID_PATH.is_file():
    pytest.skip(
        "Schreiber+2018 grid not found at "
        + str(_GRID_PATH)
        + " (build with: python scripts/build_schreiber2018_grid.py)",
        allow_module_level=True,
    )

if not (_S17_DUST.is_file() and _S17_PAH.is_file()):
    pytest.skip(
        "AGNfitter S17 FITS not found under /tmp/AGNfitter-rX "
        "(clone with: git clone --branch AGNfitter-rX_v0.1 "
        "https://github.com/GabrielaCR/AGNfitter /tmp/AGNfitter-rX)",
        allow_module_level=True,
    )

pytest.importorskip("astropy")


def _agnfitter_s17_mix(tdust: float, fpah: float) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct AGNfitter-rX's S17 cold-dust mixture (wave [Å], L_nu)."""
    from astropy import units as u
    from astropy.table import Table

    dust = Table.read(_S17_DUST)
    pah = Table.read(_S17_PAH)
    lam_um = np.asarray(dust["LAM"][0], dtype=np.float64)  # (n_T, n_wave)
    d_nulnu = np.asarray(dust["SED"][0], dtype=np.float64)
    p_nulnu = np.asarray(pah["SED"][0], dtype=np.float64)
    tdust_ax = np.asarray(dust["TDUST"][0], dtype=np.float64)
    nu = (lam_um * u.micron).to(u.Hz, equivalencies=u.spectral()).value
    d_lnu = d_nulnu / nu
    p_lnu = p_nulnu / nu
    # AGNfitter selects the nearest tabulated temperature node.
    t = int(np.argmin(np.abs(tdust_ax - tdust)))
    wave_aa = lam_um[t] * 1.0e4
    order = np.argsort(wave_aa)
    mix = (1.0 - fpah) * d_lnu[t] + fpah * p_lnu[t]
    return wave_aa[order], mix[order]


def _tengri_s17(wave_aa: np.ndarray, tdust: float, fpah: float) -> np.ndarray:
    """tengri ``schreiber2018`` L_nu on the requested grid."""
    from tengri.components.dust.emission_templates import create_schreiber2018_from_grid

    fn = create_schreiber2018_from_grid(str(_GRID_PATH))
    return np.asarray(fn(jnp.asarray(wave_aa), 1.0, dust_T=tdust, dust_f_pah=fpah))


@pytest.mark.parametrize(
    "tdust,fpah",
    [(20.0, 0.01), (30.0, 0.02), (45.0, 0.05), (55.0, 0.03)],
)
def test_schreiber2018_matches_agnfitter_shape(tdust: float, fpah: float) -> None:
    """tengri schreiber2018 reproduces AGNfitter-rX S17 shape to <3% of peak."""
    w_af, l_af = _agnfitter_s17_mix(tdust, fpah)
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
    w_af, l_af = _agnfitter_s17_mix(tdust, 0.0)
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
