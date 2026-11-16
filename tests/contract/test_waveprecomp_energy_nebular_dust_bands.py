# SPDX-License-Identifier: BSD-3-Clause
"""Energy-convention LUTs for the nebular grid and the dust-emission bands (#3505, B6).

Every band builder that integrates a filter must take the observation's convention,
or an energy (CIGALE) observation gets Bessell bands from that component while the
stellar bands follow the energy weight. Two families are pinned here:

* the grid-served nebular photometry (``neb={'type': 'cue'}`` under ``FeaturePrecomp``),
  on a band that the Halpha line dominates;
* the dust-emission LUT (``dale2014``), on an infrared band the emission dominates.

For each, the energy LUT must agree with the exact energy path to 1e-3 per band, and
the energy exact path must differ from the Bessell exact path by more than that on
the same band, so the convention is really in the number.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from tengri import DEFAULT, FeaturePrecomp, Fixed, Observation, Photometry, SEDModel, WavePrecomp
from tengri.observation.photometry import FilterConvention

pytestmark = pytest.mark.contract

#: Per-band agreement of the energy LUT with the exact energy path (relative).
LUT_TOL = 1e-3

#: sdss_r contains Halpha (6563 A), the line that dominates the nebular continuum-plus-line
#: flux of a star-forming model in that band.
NEB_BANDS = ["sdss_r"]
#: WISE W3 (12 um) is inside the dale2014 emission peak, far from the stellar light.
DUST_BANDS = ["wise_w3"]


def _obs(names, convention):
    return Observation(photometry=Photometry.from_names(names, convention=convention))


def _build(ssp, names, convention, approx, **groups):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel.build(
            ssp_data=ssp,
            observation=_obs(names, convention),
            sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
            redshift=Fixed(0.05),
            approx=approx,
            **groups,
        )


def _phot(model):
    return np.asarray(model.predict_photometry({}), dtype=np.float64)


@pytest.fixture(scope="module")
def ssp(ssp_data_fsps):
    """Bare-stellar grid: Cue refuses a wNE grid, which already carries the nebular lines."""
    return ssp_data_fsps


NEB = {"neb": {"type": "cue", "all_params": Fixed(DEFAULT)}}
DUST = {
    "dust_attenuation": {
        "type": "two_component",
        "law": "calzetti",
        "all_params": Fixed(DEFAULT),
        "tau_diff": 0.5,
    },
    "dust_emission": {"type": "dale2014", "all_params": Fixed(DEFAULT)},
}


def test_energy_nebular_grid_band_matches_the_exact_energy_path(ssp):
    """The grid-served nebular band follows the energy weight, not the Bessell one."""
    approx = (WavePrecomp(), FeaturePrecomp())
    exact = _phot(_build(ssp, NEB_BANDS, FilterConvention.ENERGY, None, **NEB))
    lut = _phot(_build(ssp, NEB_BANDS, FilterConvention.ENERGY, approx, **NEB))
    rel = np.abs(lut / exact - 1.0)
    assert np.all(rel < LUT_TOL), f"energy nebular grid band vs exact: {rel}"


def test_energy_and_bessell_differ_on_the_nebular_line_band(ssp):
    """On the Halpha-dominated band the two conventions must differ by more than LUT_TOL."""
    energy = _phot(_build(ssp, NEB_BANDS, FilterConvention.ENERGY, None, **NEB))
    bessell = _phot(_build(ssp, NEB_BANDS, FilterConvention.BESSELL, None, **NEB))
    rel = float(np.max(np.abs(energy / bessell - 1.0)))
    assert rel > LUT_TOL, f"energy and Bessell nebular bands are indistinguishable: {rel:.3e}"


def test_energy_dust_emission_band_matches_the_exact_energy_path(ssp):
    """The dust-emission LUT follows the energy weight, not the Bessell one."""
    exact = _phot(_build(ssp, DUST_BANDS, FilterConvention.ENERGY, None, **DUST))
    lut = _phot(_build(ssp, DUST_BANDS, FilterConvention.ENERGY, WavePrecomp(), **DUST))
    rel = np.abs(lut / exact - 1.0)
    assert np.all(rel < LUT_TOL), f"energy dust-emission band vs exact: {rel}"


def test_energy_and_bessell_differ_on_the_dust_emission_band(ssp):
    """The dust band the emission dominates must differ between conventions by > LUT_TOL."""
    energy = _phot(_build(ssp, DUST_BANDS, FilterConvention.ENERGY, None, **DUST))
    bessell = _phot(_build(ssp, DUST_BANDS, FilterConvention.BESSELL, None, **DUST))
    rel = float(np.max(np.abs(energy / bessell - 1.0)))
    assert rel > LUT_TOL, f"energy and Bessell dust bands are indistinguishable: {rel:.3e}"
