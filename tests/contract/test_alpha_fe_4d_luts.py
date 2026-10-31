#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""A 4D ([alpha/Fe]-axis) SSP grid works on the fixed-redshift lookup-table path.

The stellar component collapses a 4D ``ssp_flux`` at ``met_alpha_fe`` for the CSP path. On the
fixed-z path the spectrum and the photometry come from lookup tables built once at construction
(``precompute_spectroscopy``, ``precompute_photometry``), and the Cue backend fits its
ionizing-parameter tables from the cube as well. Before this contract those three assumed a 3D
cube: model construction raised ``ValueError: too many values to unpack``, and had it not,
[alpha/Fe] would have been inert on that path. Contract:

1. a model with photometry + spectroscopy builds on a 4D grid;
2. at a node value of [alpha/Fe] its predictions equal the 3D model built from that node's slab
   (bit-for-bit up to float reassociation);
3. ``d(prediction)/d(met_alpha_fe)`` is nonzero and matches a central finite difference.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tengri
from tengri import DEFAULT, SEDModel, Uniform
from tengri.components.stellar.sps.dsps_wrapper import SSPData, has_alpha_grid
from tengri.parameters import Fixed
from tengri.utils.physics_constants import LOG10_ZSUN

pytestmark = pytest.mark.contract

ALPHA_NODES = np.array([-0.2, 0.0, 0.2, 0.4, 0.6])


@pytest.fixture(scope="module")
def ssp4d():
    """Build 3 Z x 5 [alpha/Fe] x 20 ages x 300 wavelengths.

    The alpha axis rescales a line-like feature, so every node differs."""
    n_met, n_age, n_wave = 3, 20, 300
    wave = jnp.linspace(3000.0, 10000.0, n_wave)
    ages = jnp.linspace(-1.0, 1.14, n_age)
    key = jax.random.PRNGKey(7)
    base = (
        jnp.abs(jax.random.normal(key, (n_met, n_age, n_wave))) * 1e-3 + 1e-3
    )  # smooth-ish positive continuum
    feature = jnp.exp(-0.5 * ((wave - 5175.0) / 40.0) ** 2)  # a Mg b-like absorption trough
    cubes = [
        base * (1.0 - 0.25 * (1.0 + a) * feature)[None, None, :] for a in ALPHA_NODES
    ]  # deeper with alpha
    flux = jnp.stack(cubes, axis=1)  # (n_met, n_alpha, n_age, n_wave)
    lgmet = jnp.array([-1.5, -0.5, 0.0]) + LOG10_ZSUN
    return SSPData(
        ssp_wave=wave,
        ssp_flux=flux,
        ssp_lg_age_gyr=ages,
        ssp_lgmet=lgmet,
        ssp_alpha_fe=jnp.asarray(ALPHA_NODES),
    )


def _slab(ssp4d, k):
    return SSPData(
        ssp_wave=ssp4d.ssp_wave,
        ssp_flux=ssp4d.ssp_flux[:, k],
        ssp_lg_age_gyr=ssp4d.ssp_lg_age_gyr,
        ssp_lgmet=ssp4d.ssp_lgmet,
    )


@pytest.fixture(scope="module")
def observation():
    wave_obs = jnp.linspace(4000.0, 8000.0, 400)
    return tengri.Observation(
        photometry=tengri.Photometry.from_names(["sdss_g", "sdss_r"]),
        spectroscopy=tengri.Spectroscopy(wave_obs=wave_obs, resolution=jnp.full(400, 2000.0)),
    )


def _model(ssp, observation, alpha_spec):
    return SEDModel.build(
        ssp_data=ssp,
        observation=observation,
        sfh={
            "type": "delayed",
            "all_params": Fixed(DEFAULT),
            "tau_gyr": Fixed(2.0),
            "age_gyr": Fixed(3.0),
            "log_total_mass": Fixed(10.0),
        },
        met={
            "logzsol": Fixed(-0.3),
            **({"alpha_fe": alpha_spec} if alpha_spec is not None else {}),
        },
        redshift=Fixed(0.05),
        neb={"type": "ssp"},
    )


def test_fixture_is_4d(ssp4d):
    assert has_alpha_grid(ssp4d) and ssp4d.ssp_flux.ndim == 4


def test_4d_model_builds_and_matches_the_node_slab(ssp4d, observation):
    m4 = _model(ssp4d, observation, Uniform(-0.2, 0.6))
    for k, a in ((1, 0.0), (3, 0.4)):
        m3 = _model(_slab(ssp4d, k), observation, None)
        o4 = m4.predict_observables({"met_alpha_fe": a})
        o3 = m3.predict_observables({})
        np.testing.assert_allclose(
            np.asarray(o4.spec_fnu), np.asarray(o3.spec_fnu), rtol=1e-10, atol=0
        )
        np.testing.assert_allclose(
            np.asarray(o4.phot_fnu), np.asarray(o3.phot_fnu), rtol=1e-10, atol=0
        )


def test_alpha_moves_the_spectrum_with_a_correct_gradient(ssp4d, observation):
    m4 = _model(ssp4d, observation, Uniform(-0.2, 0.6))

    def mgb(a):
        o = m4.predict_observables({"met_alpha_fe": a})
        pix = (np.asarray(observation.spectroscopy.wave_obs) / 1.05 > 5150) & (
            np.asarray(observation.spectroscopy.wave_obs) / 1.05 < 5200
        )
        return jnp.sum(jnp.where(jnp.asarray(pix), o.spec_fnu, 0.0))

    g = float(jax.grad(mgb)(0.25))
    h = 1e-3
    fd = float((mgb(0.25 + h) - mgb(0.25 - h)) / (2 * h))
    assert g != 0.0
    np.testing.assert_allclose(g, fd, rtol=1e-4)
    assert float(mgb(0.6)) < float(mgb(-0.2))  # deeper trough with alpha, as the fixture is built
