#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""``met_alpha_fe`` is fixed by default on every grid; an explicit prior is checked (#2250).

Owner ruling 2026-11-14: [alpha/Fe] is never freed by a default, on any grid and with any
observation. A prior the user writes is honored on a grid with an [alpha/Fe] axis, and is
checked against the grid's [alpha/Fe] nodes, where the lookup clips: a prior that misses the
nodes entirely is refused by name, and one that extends past them is kept with a warning.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tengri
from tengri import DEFAULT, FREE, SEDModel, Uniform
from tengri.components.stellar.sps.dsps_wrapper import SSPData
from tengri.config.exceptions import GridSupportWarning, ParameterError
from tengri.parameters import Fixed
from tengri.utils.physics_constants import LOG10_ZSUN

pytestmark = pytest.mark.contract

ALPHA_NODES = np.array([-0.2, 0.0, 0.2, 0.4, 0.6])


@pytest.fixture(scope="module")
def ssp4d():
    """3 Z x 5 [alpha/Fe] x 20 ages x 300 wavelengths, an Mg b-like trough deepening with alpha."""
    n_met, n_age, n_wave = 3, 20, 300
    wave = jnp.linspace(3000.0, 10000.0, n_wave)
    ages = jnp.linspace(-1.0, 1.14, n_age)
    base = jnp.abs(jax.random.normal(jax.random.PRNGKey(7), (n_met, n_age, n_wave))) * 1e-3 + 1e-3
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


def _photometry_only():
    return tengri.Observation(photometry=tengri.Photometry.from_names(["sdss_g", "sdss_r"]))


def _with_spectrum():
    wave_obs = jnp.linspace(4000.0, 8000.0, 400)
    return tengri.Observation(
        photometry=tengri.Photometry.from_names(["sdss_g", "sdss_r"]),
        spectroscopy=tengri.Spectroscopy(wave_obs=wave_obs, resolution=jnp.full(400, 2000.0)),
    )


def _model(ssp, observation, met):
    return SEDModel.build(
        ssp_data=ssp,
        observation=observation,
        sfh={"type": "delayed", "all_params": Fixed(DEFAULT), "log_total_mass": Fixed(10.0)},
        met={"logzsol": Fixed(-0.3), **met},
        redshift=Fixed(0.05),
        neb={"type": "ssp"},
    )


@pytest.mark.parametrize("observation", [_photometry_only, _with_spectrum], ids=["phot", "spec"])
def test_alpha_fe_is_fixed_by_default_even_with_a_spectrum(ssp4d, observation):
    model = _model(ssp4d, observation(), {"all_params": FREE})
    assert "met_alpha_fe" not in model.spec.free_params
    assert model.spec.get_distribution("met_alpha_fe").is_fixed
    assert float(model.spec.get_distribution("met_alpha_fe").value) == 0.0


@pytest.mark.parametrize("observation", [_photometry_only, _with_spectrum], ids=["phot", "spec"])
def test_a_bare_free_is_refused_by_name(ssp4d, observation):
    with pytest.raises(ParameterError, match="met_alpha_fe"):
        _model(ssp4d, observation(), {"alpha_fe": FREE})


@pytest.mark.parametrize("observation", [_photometry_only, _with_spectrum], ids=["phot", "spec"])
def test_an_explicit_prior_inside_the_nodes_is_honored_silently(ssp4d, observation):
    with warnings.catch_warnings():
        warnings.simplefilter("error", GridSupportWarning)
        model = _model(ssp4d, observation(), {"alpha_fe": Uniform(-0.2, 0.4)})
    assert "met_alpha_fe" in model.spec.free_params
    assert model.spec.get_distribution("met_alpha_fe").bounds == pytest.approx((-0.2, 0.4))


def test_an_explicit_prior_missing_the_nodes_is_refused_by_name(ssp4d):
    with pytest.raises(ParameterError, match=r"met_alpha_fe.*does not overlap"):
        _model(ssp4d, _with_spectrum(), {"alpha_fe": Uniform(0.8, 1.0)})


def test_an_explicit_prior_past_the_nodes_is_kept_with_one_warning(ssp4d):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = _model(ssp4d, _with_spectrum(), {"alpha_fe": Uniform(-0.4, 0.4)})
    grid = [w for w in caught if issubclass(w.category, GridSupportWarning)]
    assert len(grid) == 1, [str(w.message) for w in caught]
    assert "met_alpha_fe" in str(grid[0].message)
    assert model.spec.get_distribution("met_alpha_fe").bounds == pytest.approx((-0.4, 0.4))
