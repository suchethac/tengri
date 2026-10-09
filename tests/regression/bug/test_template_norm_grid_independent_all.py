# SPDX-License-Identifier: BSD-3-Clause
"""Template normalization is grid-independent (owner ruling 2026-10-09).

Every template is normalized by its integral on its OWN native grid (or a fixed
converged internal grid for an analytic shape), never on the caller's wavelength
grid. Two properties are checked for each fixed component:

1. Grid independence: the published spectrum at the nodes shared by a coarse
   (200-point) and a fine (5000-point) caller grid agrees to 1e-6.
2. Declared power: on a converged grid the emitted power equals the declared
   power to 1e-4.

Fixed components: the astrodust+PAH IR emission (its row integrals are taken on
the published microns grid) and the GRAHSP unit-l5100 bolometric (taken on the
fixed NORMALIZATION_GRID_NM).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.regression_bug

C_AA_PER_S = 2.99792458e18
COARSE_N = 200
FINE_N = 5000
CONVERGED_N = 200_000


def _shared_grids(lo_aa: float, hi_aa: float) -> tuple[np.ndarray, np.ndarray]:
    """Coarse and fine caller grids; the fine grid contains every coarse node."""
    coarse = np.geomspace(lo_aa, hi_aa, COARSE_N)
    fine = np.union1d(coarse, np.geomspace(lo_aa, hi_aa, FINE_N))
    return coarse, fine


def _emitted_power_erg_s(spectrum_lnu: np.ndarray, wave_aa: np.ndarray) -> float:
    """Frequency integral of L_nu over the caller grid [erg/s]."""
    nu = C_AA_PER_S / wave_aa
    return float(np.trapezoid(spectrum_lnu[::-1], nu[::-1]))


def _astrodust_spectrum(wave_aa: np.ndarray, l_ir: float) -> np.ndarray:
    from tengri.components.dust.emission.templates.astrodust import AstrodustIRSEDComponent

    comp = AstrodustIRSEDComponent()
    n = len(wave_aa)
    _, extra = comp.predict({"lgU": 1.0}, jnp.zeros(n), jnp.asarray(wave_aa), L_ir=l_ir)
    return np.asarray(extra["sed_dust_ir"])


def _grahsp_spectrum(wave_aa: np.ndarray, log_lbol: float) -> np.ndarray:
    from tengri.components.agn.grahsp.model import compute_grahsp_sed

    return np.asarray(
        compute_grahsp_sed(
            jnp.asarray(wave_aa),
            agn_log_lbol=log_lbol,
            agn_grahsp_ebv=0.0,
            agn_grahsp_ebv_agn=0.0,
        )
    )


def _check_grid_independence(spectrum_fn, lo_aa: float, hi_aa: float) -> None:
    coarse, fine = _shared_grids(lo_aa, hi_aa)
    at_coarse = spectrum_fn(coarse)
    at_fine = spectrum_fn(fine)[np.searchsorted(fine, coarse)]
    scale = np.max(np.abs(at_coarse))
    assert scale > 0.0
    assert np.max(np.abs(at_coarse - at_fine)) / scale <= 1e-6


def test_astrodust_normalization_is_grid_independent():
    _check_grid_independence(lambda w: _astrodust_spectrum(w, 1e44), 1e3, 1e8)


def test_astrodust_emits_declared_power_on_converged_grid():
    wave = np.geomspace(1e3, 1e8, CONVERGED_N)
    power = _emitted_power_erg_s(_astrodust_spectrum(wave, 1e44), wave)
    assert power == pytest.approx(1e44, rel=1e-4)


def test_grahsp_normalization_is_grid_independent():
    _check_grid_independence(lambda w: _grahsp_spectrum(w, 12.0), 912.0, 1e8)


def test_grahsp_emits_declared_bolometric_power_on_converged_grid():
    # lumBolBBB is the integral above the Lyman limit (912 A): the declared
    # power is 10**agn_log_lbol L_sun, emitted above 912 A on a grid that
    # reaches the converged tail (1e11 A).
    wave = np.geomspace(912.0, 1e11, 100_000)
    declared = 10.0**12.0 * 3.828e33
    lnu = _grahsp_spectrum(wave, 12.0)
    above_lyman = wave >= 912.0
    power = _emitted_power_erg_s(np.where(above_lyman, lnu, 0.0), wave)
    assert power == pytest.approx(declared, rel=1e-4)
