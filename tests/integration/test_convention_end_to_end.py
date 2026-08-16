# SPDX-License-Identifier: BSD-3-Clause
"""End-to-end wiring of the filter convention through SEDModel.predict_photometry.

The kernel-level convention is covered by
``tests/crossval/test_filter_convention_parity.py``; here we check that a
user-selected convention on ``Photometry`` actually reaches the forward model
(``compile_signature`` distinguishes them; predict outputs differ) and that the
WavePrecomp LUT path refuses the not-yet-supported energy convention.

Skips when the SSP data file is missing.
"""

from __future__ import annotations

from pathlib import Path

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data
from tengri.forward.sed_model import SEDModel, WavePrecomp
from tengri.observation.observation import Observation
from tengri.observation.photometry_config import Photometry
from tengri.parameters.parameters import Parameters
from tengri.parameters.priors import Uniform

pytestmark = pytest.mark.contract

_SSP_FILE = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"
)
_skip = pytest.mark.skipif(not _SSP_FILE.is_file(), reason=f"SSP data not found: {_SSP_FILE}")
_NAMES = ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"]
_PARAMS = {
    "sfh_tsnorm_log_total_mass": 0.7,
    "sfh_tsnorm_peak_lbt_gyr": 3.0,
    "sfh_tsnorm_width_gyr": 1.5,
    "sfh_tsnorm_skew": 0.2,
    "sfh_tsnorm_trunc": 3.0,
    "met_logzsol": -0.3,
    "dust_tau_bc": 0.4,
    "dust_tau_diff": 0.2,
}


def _spec():
    return Parameters(
        mean_sfh_type="tsnorm",
        sfh_tsnorm_log_total_mass=Uniform(7.0, 12.5),
        sfh_tsnorm_peak_lbt_gyr=Uniform(0.5, 12.0),
        sfh_tsnorm_width_gyr=Uniform(0.2, 5.0),
        sfh_tsnorm_skew=Uniform(-1.0, 1.0),
        sfh_tsnorm_trunc=Uniform(1.0, 10.0),
        met_logzsol=Uniform(-2.0, 0.2),
        dust_tau_bc=Uniform(0.0, 3.0),
        dust_tau_diff=Uniform(0.0, 2.0),
        dust_slope=-0.7,
        redshift=0.05,
    )


@pytest.fixture(scope="module")
def ssp():
    return load_ssp_data(str(_SSP_FILE))


@_skip
def test_energy_convention_changes_predict_photometry(ssp):
    """A user-selected energy convention reaches predict_photometry."""
    mb = SEDModel(
        spec=_spec(),
        ssp_data=ssp,
        observation=Observation(photometry=Photometry.from_names(_NAMES)),
    )
    me = SEDModel(
        spec=_spec(),
        ssp_data=ssp,
        observation=Observation(photometry=Photometry.from_names(_NAMES, convention="energy")),
    )
    # Distinct compile signatures (so the cached observables closure is not shared).
    assert mb.compile_signature() != me.compile_signature()

    fb = np.asarray(mb.predict_photometry(_PARAMS))
    fe = np.asarray(me.predict_photometry(_PARAMS))
    assert np.all(fb > 0) and np.all(fe > 0)
    # Conventions must differ on a real (broken-continuum) stellar SED.
    dmag = -2.5 * np.log10(fb / fe)
    assert np.max(np.abs(dmag)) > 1e-3, f"energy vs bessell indistinguishable: {dmag}"


@_skip
def test_string_convention_accepted(ssp):
    """Photometry accepts a plain string convention and normalizes it."""
    from tengri.observation.photometry import FilterConvention

    phot = Photometry.from_names(_NAMES, convention="energy")
    assert phot.convention is FilterConvention.ENERGY
    assert Photometry.from_names(_NAMES).convention is FilterConvention.BESSELL


@_skip
def test_waveprecomp_rejects_energy_convention(ssp):
    """WavePrecomp LUT path refuses energy rather than silently using Bessell."""
    obs = Observation(photometry=Photometry.from_names(_NAMES, convention="energy"))
    with pytest.raises(NotImplementedError, match="bessell"):
        SEDModel(
            spec=_spec(),
            ssp_data=ssp,
            observation=obs,
            approx=WavePrecomp(z_min=0.0, z_max=1.0, n_z=50),
        )


@_skip
def test_waveprecomp_allows_bessell(ssp):
    """WavePrecomp builds fine under the default Bessell convention."""
    obs = Observation(photometry=Photometry.from_names(_NAMES))
    m = SEDModel(
        spec=_spec(),
        ssp_data=ssp,
        observation=obs,
        approx=WavePrecomp(z_min=0.0, z_max=1.0, n_z=50),
    )
    assert np.all(np.asarray(m.predict_photometry(_PARAMS)) > 0)
