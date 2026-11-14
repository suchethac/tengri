#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""A free [alpha/Fe] that the observation cannot constrain warns once per build (#3487).

[alpha/Fe] moves the Mg b / Fe features at 4800-5400 A rest frame. Only spectral indices, or a
spectrum that resolves (R >= 1000) and covers that window, constrain it. Photometry moves light
by about 1-2% at fixed total metallicity, so a free [alpha/Fe] fitted to photometry follows its
prior. The model still builds, so the advisory warns and never refuses.
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np
import pytest

import tengri
from tengri import DEFAULT, FREE, SEDModel, Uniform
from tengri.components.stellar.sps.dsps_wrapper import SSPData
from tengri.config.exceptions import AlphaFeUnconstrainedWarning, ParameterError
from tengri.observation.spectral_indices import STANDARD_INDICES, SpectralIndexData
from tengri.parameters import Fixed
from tengri.utils.physics_constants import LOG10_ZSUN

pytestmark = pytest.mark.contract

ALPHA_NODES = np.array([-0.2, 0.0, 0.2, 0.4, 0.6])


@pytest.fixture(scope="module")
def ssp4d():
    """3 Z x 5 [alpha/Fe] x 20 ages x 300 wavelengths, as in the explicit-prior contract."""
    n_met, n_age, n_wave = 3, 20, 300
    wave = jnp.linspace(3000.0, 10000.0, n_wave)
    ages = jnp.linspace(-1.0, 1.14, n_age)
    base = (
        jnp.abs(jnp.sin(jnp.arange(n_met * n_age * n_wave, dtype=float))).reshape(
            n_met, n_age, n_wave
        )
        * 1e-3
        + 1e-3
    )
    feature = jnp.exp(-0.5 * ((wave - 5175.0) / 40.0) ** 2)
    flux = jnp.stack(
        [base * (1.0 - 0.25 * (1.0 + a) * feature)[None, None, :] for a in ALPHA_NODES], axis=1
    )
    return SSPData(
        ssp_wave=wave,
        ssp_flux=flux,
        ssp_lg_age_gyr=ages,
        ssp_lgmet=jnp.array([-1.5, -0.5, 0.0]) + LOG10_ZSUN,
        ssp_alpha_fe=jnp.asarray(ALPHA_NODES),
    )


@pytest.fixture(scope="module")
def ssp3d(ssp4d):
    """The same grid without an [alpha/Fe] axis: one scaled-solar template per metallicity."""
    return SSPData(
        ssp_wave=ssp4d.ssp_wave,
        ssp_flux=ssp4d.ssp_flux[:, 2],
        ssp_lg_age_gyr=ssp4d.ssp_lg_age_gyr,
        ssp_lgmet=ssp4d.ssp_lgmet,
    )


def _photometry():
    return tengri.Photometry.from_names(["sdss_g", "sdss_r"])


def _photometry_only():
    return tengri.Observation(photometry=_photometry())


def _spectrum(rest_lo, rest_hi, resolution, z=0.05):
    """A spectrum covering rest_lo-rest_hi A at redshift z, sampled on the observed frame."""
    wave_obs = jnp.asarray(np.linspace(rest_lo, rest_hi, 400) * (1.0 + z))
    return tengri.Spectroscopy(wave_obs=wave_obs, resolution=resolution)


def _model(ssp, observation, met, redshift=None):
    return SEDModel.build(
        ssp_data=ssp,
        observation=observation,
        sfh={"type": "delayed", "all_params": Fixed(DEFAULT), "log_total_mass": Fixed(10.0)},
        met={"logzsol": Fixed(-0.3), **met},
        redshift=Fixed(0.05) if redshift is None else redshift,
        neb={"type": "ssp"},
    )


def _alpha_warnings(build):
    """Run ``build`` and return the AlphaFeUnconstrainedWarning messages it emitted."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        build()
    return [str(w.message) for w in caught if issubclass(w.category, AlphaFeUnconstrainedWarning)]


def test_photometry_only_with_free_alpha_warns_once(ssp4d):
    msgs = _alpha_warnings(
        lambda: _model(ssp4d, _photometry_only(), {"alpha_fe": Uniform(-0.2, 0.4)})
    )
    assert len(msgs) == 1
    assert "photometry" in msgs[0]
    assert "met_alpha_fe" in msgs[0]
    assert "4800-5400" in msgs[0]


def test_low_resolution_spectrum_warns_with_the_resolving_power(ssp4d):
    obs = tengri.Observation(photometry=_photometry(), spectroscopy=_spectrum(4500, 6000, 100.0))
    msgs = _alpha_warnings(lambda: _model(ssp4d, obs, {"alpha_fe": Uniform(-0.2, 0.4)}))
    assert len(msgs) == 1
    assert "R = 100" in msgs[0]


def test_out_of_window_spectrum_warns_with_the_coverage(ssp4d):
    obs = tengri.Observation(photometry=_photometry(), spectroscopy=_spectrum(6000, 8000, 2000.0))
    msgs = _alpha_warnings(lambda: _model(ssp4d, obs, {"alpha_fe": Uniform(-0.2, 0.4)}))
    assert len(msgs) == 1
    assert "rest coverage" in msgs[0]
    assert "z = 0.05" in msgs[0]


def test_resolved_covering_spectrum_at_fixed_z_does_not_warn(ssp4d):
    obs = tengri.Observation(photometry=_photometry(), spectroscopy=_spectrum(4500, 6000, 2000.0))
    msgs = _alpha_warnings(lambda: _model(ssp4d, obs, {"alpha_fe": Uniform(-0.2, 0.4)}))
    assert msgs == []


def test_free_redshift_checks_both_prior_ends(ssp4d):
    obs = tengri.Observation(
        photometry=_photometry(), spectroscopy=_spectrum(4800, 5400, 2000.0, z=0.0)
    )
    msgs = _alpha_warnings(
        lambda: _model(
            ssp4d,
            obs,
            {"alpha_fe": Uniform(-0.2, 0.4)},
            redshift=Uniform(0.0, 0.5),
        )
    )
    assert len(msgs) == 1
    assert "z = 0.5" in msgs[0]


def test_spectral_indices_never_warn(ssp4d):
    sid = SpectralIndexData(
        index_defs=(STANDARD_INDICES["Dn4000"],),
        values=jnp.array([1.4]),
        errors=jnp.array([0.02]),
    )
    obs = tengri.Observation(photometry=_photometry(), spectral_indices=sid)
    msgs = _alpha_warnings(lambda: _model(ssp4d, obs, {"alpha_fe": Uniform(-0.2, 0.4)}))
    assert msgs == []


def test_fixed_alpha_with_photometry_does_not_warn(ssp4d):
    msgs = _alpha_warnings(lambda: _model(ssp4d, _photometry_only(), {}))
    assert msgs == []


def test_evolving_pair_warns_and_names_the_free_end(ssp4d):
    met = {"alpha_fe_evolving": True, "alpha_fe_old": Uniform(0.0, 0.4)}
    msgs = _alpha_warnings(lambda: _model(ssp4d, _photometry_only(), met))
    assert len(msgs) == 1
    assert "met_alpha_fe_old" in msgs[0]


def test_3d_grid_free_alpha_is_refused_not_warned(ssp3d):
    # The 3-D refusal runs first and raises, so no advisory is emitted.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ParameterError, match="met_alpha_fe"):
            _model(ssp3d, _photometry_only(), {"alpha_fe": FREE})
    msgs = [w for w in caught if issubclass(w.category, AlphaFeUnconstrainedWarning)]
    assert msgs == []
