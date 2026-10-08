# SPDX-License-Identifier: BSD-3-Clause
"""Tabulated dust templates emit the same SED on any rest-frame wavelength grid.

Each tabulated dust emission closure divides its resampled template by a
frequency integral. Taking that integral on the caller's grid ties the emitted
power to the grid's density and extent. The integral is instead taken on the
template's own native grid, so the template carries the absorbed power
``L_absorbed`` whatever grid it is evaluated on.

Two checks per model:

* the emitted :math:`L_\\nu` integrated on a fine grid over the full template
  support equals ``L_absorbed`` to 1e-4;
* the emission sampled on a coarse (200-point) grid and on a fine grid agrees
  to 1e-6 at the coarse nodes, so top-hat band fluxes integrated over those
  nodes are identical.
"""

from __future__ import annotations

import numpy as np
import pytest

from tengri._data_setup import find_data_str
from tengri.components.dust import emission_templates as et

pytestmark = pytest.mark.regression_bug

C_AA_PER_S = 2.99792458e18
L_ABS = 1.0e44  # erg/s
# Template support of the libraries, widened: resample_template is zero outside it.
FULL_RANGE_AA = (1.0e2, 1.0e9)
# Band-flux nodes: rest-frame 10, 25, 50 micron, +-0.1 dex top-hats.
BAND_CENTRES_AA = (1.0e5, 2.5e5, 5.0e5)
BAND_HALF_DEX = 0.1


def _factories():
    """Model name -> (factory, kwargs for the closure at a fixed parameter point)."""
    return {
        "dale2014": (
            lambda: et.create_dale2014_from_grid(find_data_str("dale2014_templates.h5")),
            {"dust_alpha_dale": 2.0, "dust_frac_agn": 0.0},
        ),
        "dl07": (
            lambda: et.create_dl07_from_grid(find_data_str("dl07_templates_v2.h5")),
            {"dust_umin": 1.0, "dust_gamma_dl": 0.1, "dust_qpah": 2.5},
        ),
        "draine_li2014": (
            lambda: et.create_dl14_from_grid(find_data_str("dl14_templates.h5")),
            {"dust_umin": 1.0, "dust_gamma_dl": 0.1},
        ),
        "schreiber2016": (
            lambda: et.create_schreiber2016_from_grid(find_data_str("schreiber2016_templates.h5")),
            {"dust_T": 30.0, "dust_f_pah": 0.3},
        ),
        "schreiber2018": (
            lambda: et.create_schreiber2018_from_grid(find_data_str("schreiber2018_templates.h5")),
            {"dust_T": 30.0, "dust_f_pah": 0.3},
        ),
        "themis": (
            lambda: et.create_themis_from_grid(find_data_str("themis_templates.h5")),
            {"dust_umin": 1.0, "dust_gamma_dl": 0.1},
        ),
        "astrodust": (
            lambda: et.create_astrodust_from_grid(find_data_str("astrodust_templates.h5")),
            {"dust_umin": 1.0, "dust_gamma_dl": 0.1, "dust_qpah": 2.5},
        ),
        "bosa": (
            lambda: et.create_bosa_from_grid(find_data_str("bosa_templates.h5")),
            {"dust_log_ssfr": -10.0, "log_L_ir": 44.0},
        ),
        "dh02_ce01": (
            lambda: et.create_dh02_ce01_from_grid(find_data_str("dh02_ce01_grid.h5")),
            {"log_L_ir": 44.0},
        ),
    }


def _model(name):
    factory, kwargs = _factories()[name]
    path_ok = all(
        find_data_str(f) is not None
        for f in (
            "dale2014_templates.h5",
            "dl07_templates_v2.h5",
            "dl14_templates.h5",
            "schreiber2016_templates.h5",
            "schreiber2018_templates.h5",
            "themis_templates.h5",
            "astrodust_templates.h5",
            "bosa_templates.h5",
            "dh02_ce01_grid.h5",
        )
    )
    if not path_ok:
        pytest.skip("dust template data not on disk")
    return factory(), kwargs


def _nu_integral(wave_aa: np.ndarray, lnu: np.ndarray) -> float:
    nu = C_AA_PER_S / wave_aa
    order = np.argsort(nu)
    return float(np.trapezoid(lnu[order], nu[order]))


def _band_fluxes(wave_aa: np.ndarray, lnu: np.ndarray) -> np.ndarray:
    """Top-hat band integrals over the nodes of ``wave_aa`` (same quadrature for any grid)."""
    nu = C_AA_PER_S / wave_aa
    order = np.argsort(nu)
    out = []
    for centre in BAND_CENTRES_AA:
        weight = (np.abs(np.log10(wave_aa / centre)) < BAND_HALF_DEX).astype(float)
        out.append(float(np.trapezoid((lnu * weight)[order], nu[order])))
    return np.asarray(out)


@pytest.mark.parametrize("name", sorted(_factories()))
def test_emitted_power_equals_l_absorbed_on_fine_grid(name):
    closure, kwargs = _model(name)
    fine = np.geomspace(*FULL_RANGE_AA, 20000)
    lnu = np.asarray(closure(fine, L_ABS, **kwargs), dtype=float)
    assert np.all(np.isfinite(lnu))
    assert _nu_integral(fine, lnu) / L_ABS == pytest.approx(1.0, abs=1e-4)


@pytest.mark.parametrize("name", sorted(_factories()))
def test_band_fluxes_do_not_depend_on_rest_grid_density(name):
    closure, kwargs = _model(name)
    coarse = np.geomspace(2.0e4, 6.0e7, 200)
    fine = np.union1d(coarse, np.geomspace(2.0e4, 6.0e7, 5000))
    lnu_coarse = np.asarray(closure(coarse, L_ABS, **kwargs), dtype=float)
    lnu_fine = np.asarray(closure(fine, L_ABS, **kwargs), dtype=float)
    at_coarse = lnu_fine[np.searchsorted(fine, coarse)]
    np.testing.assert_allclose(
        _band_fluxes(coarse, at_coarse), _band_fluxes(coarse, lnu_coarse), rtol=1e-6
    )
    np.testing.assert_allclose(at_coarse, lnu_coarse, rtol=1e-6)
