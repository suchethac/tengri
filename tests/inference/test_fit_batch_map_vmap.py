# SPDX-License-Identifier: BSD-3-Clause
"""Sanity test for the vmapped batch MAP fitter.

Verifies that ``fit_batch_map_vmap`` returns finite MAP estimates for a
small catalog of mock photometry. Does not assert parity with the looped
MAP path — adam optimization paths differ subtly, and the point of this
helper is throughput, not bit-exact agreement.
"""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import pytest

SSP_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
    "ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5",
)

pytestmark = [
    pytest.mark.skipif(
        not os.path.exists(SSP_FILE),
        reason="SSP file not available; integration-level fixture",
    ),
    pytest.mark.contract,
]

jax.config.update("jax_enable_x64", True)


def _make_model(ssp):
    from tengri import (
        Fixed,
        Observation,
        Parameters,
        Photometry,
        SEDModel,
        Uniform,
    )

    spec = Parameters(
        sfh_tsnorm_log_total_mass=Uniform(7.0, 12.5),
        sfh_tsnorm_peak_lbt_gyr=Uniform(0.5, 12.0),
        sfh_tsnorm_width_gyr=Uniform(0.3, 5.0),
        sfh_tsnorm_skew=Uniform(-3.0, 3.0),
        sfh_tsnorm_trunc=Uniform(1.0, 10.0),
        met_logzsol=Uniform(-2.0, 0.2),
        dust_tau_bc=Uniform(0.0, 2.0),
        dust_tau_diff=Uniform(0.0, 1.5),
        dust_slope=Fixed(-0.7),
        redshift=Fixed(0.1),
        mean_sfh_type="tsnorm",
    )
    obs = Observation(
        photometry=Photometry.from_names(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])
    )
    return SEDModel(spec, ssp, observation=obs)


def _synthesize_catalog(model, n: int = 4, snr: float = 30.0, seed: int = 0):
    """Generate N mock photometry rows by perturbing one base parameter draw."""
    from tengri import generate_mock

    key = jax.random.PRNGKey(seed)
    base = {**model.spec.sample(key)}
    base["sfh_tsnorm_log_total_mass"] = jnp.array(1.0)
    base["sfh_tsnorm_peak_lbt_gyr"] = jnp.array(3.0)
    base["sfh_tsnorm_width_gyr"] = jnp.array(3.0)

    fluxes = []
    noises = []
    for i in range(n):
        mock = generate_mock(model, base, key=jax.random.fold_in(key, i + 1), snr=snr)
        fluxes.append(jnp.asarray(mock["flux_obs"]))
        noises.append(jnp.asarray(mock["noise"]))
    return jnp.stack(fluxes), jnp.stack(noises)


@pytest.mark.integration
def test_fit_batch_map_vmap_returns_finite_map_for_small_catalog(ssp_data_wne):
    from tengri.forward.convenience import fit_batch_map_vmap

    model = _make_model(ssp_data_wne)
    fluxes, noises = _synthesize_catalog(model, n=4)

    result = fit_batch_map_vmap(
        model,
        fluxes,
        noises,
        n_steps=200,
        learning_rate=0.02,
        verbose=False,
    )

    n_gal = int(fluxes.shape[0])
    for name, arr in result.items():
        a = jnp.asarray(arr)
        # Per-galaxy leading axis
        assert a.shape[0] == n_gal, f"{name!r} has shape {a.shape}, expected leading {n_gal}"
        assert jnp.all(jnp.isfinite(a)), f"{name!r} produced non-finite MAP estimates"

    # Spot-check: log_total_mass should sit well inside its prior bounds
    log_total_mass = result["sfh_tsnorm_log_total_mass"]
    assert jnp.all(log_total_mass > -1.0), f"log_total_mass hit lower prior: {log_total_mass}"
    assert jnp.all(log_total_mass < 2.5), f"log_total_mass hit upper prior: {log_total_mass}"
