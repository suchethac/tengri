# SPDX-License-Identifier: BSD-3-Clause
"""Integration test: spectrum prediction with wavelength chunking end-to-end.

Verifies that spectrum prediction with wave_chunk_size produces numerically
consistent results across multiple calls and configurations.

Requires SSP data on disk; skipped gracefully when missing.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import Fixed, Observation, Parameters, SEDModel, Spectroscopy, Uniform

# Migrated to the synthetic SSP fixture (#613) so this structural population
# spectroscopy-chunking test runs in CI instead of skipping on missing data.


@pytest.fixture(scope="module")
def ssp(synthetic_ssp_wide):
    return synthetic_ssp_wide


@pytest.fixture(scope="module")
def spec():
    """Simple spec for quick fitting tests."""
    return Parameters(
        mean_sfh_type="dense_basis",
        sfh_db_log_total_mass=Uniform(8, 12),
        sfh_db_log_sfr_inst=Uniform(-3, 3),
        sfh_db_tx_frac_0=Uniform(0.05, 0.95),
        sfh_db_tx_frac_1=Uniform(0.05, 0.95),
        sfh_db_tx_frac_2=Uniform(0.05, 0.95),
        met_logzsol=Fixed(0.0),
        dust_tau_bc=Fixed(0.0),
        dust_tau_diff=Uniform(0.0, 2.0),
        redshift=0.1,
    )


@pytest.fixture(scope="module")
def wave_obs():
    """Spectrum grid: 200 pixels, 3000–7500 Å rest-frame at z=0.1."""
    z = 0.1
    wave_rest_min = 3000.0
    wave_rest_max = 7500.0
    wave_obs_min = wave_rest_min * (1.0 + z)
    wave_obs_max = wave_rest_max * (1.0 + z)
    return jnp.logspace(jnp.log10(wave_obs_min), jnp.log10(wave_obs_max), 200)


def test_spectrum_chunked_consistency_multiple_calls(ssp, spec, wave_obs):
    """Spectrum chunking should be consistent across multiple calls."""
    key = jax.random.PRNGKey(100)
    model_unchunked = SEDModel(
        spec,
        ssp,
        wave_chunk_size=None,
        observation=Observation(spectroscopy=Spectroscopy(wave_obs=wave_obs)),
    )
    model_chunked = SEDModel(
        spec,
        ssp,
        wave_chunk_size=64,
        observation=Observation(spectroscopy=Spectroscopy(wave_obs=wave_obs)),
    )

    # Sample a single parameter set
    params = model_unchunked.spec.sample(key)

    # Call predict_spectrum 5 times with each model
    fluxes_unchunked = []
    fluxes_chunked = []

    for _ in range(5):
        flux_u = model_unchunked.predict_spectrum(params, wave_obs)
        flux_c = model_chunked.predict_spectrum(params, wave_obs)

        fluxes_unchunked.append(flux_u)
        fluxes_chunked.append(flux_c)

    # All unchunked calls should be identical
    for i in range(1, 5):
        np.testing.assert_array_equal(fluxes_unchunked[i], fluxes_unchunked[0])

    # All chunked calls should be identical
    for i in range(1, 5):
        np.testing.assert_array_equal(fluxes_chunked[i], fluxes_chunked[0])

    # Chunked should match unchunked (bitwise)
    np.testing.assert_allclose(fluxes_chunked[0], fluxes_unchunked[0], rtol=1e-12, atol=1e-15)


def test_spectrum_chunked_across_wavelengths(ssp, spec):
    """Chunking should work across different wavelength ranges."""
    # Test with shorter wavelength range
    z = 0.1
    for wave_rest_range in [(3000.0, 5000.0), (5000.0, 7500.0), (3000.0, 7500.0)]:
        wave_rest_min, wave_rest_max = wave_rest_range
        wave_obs_min = wave_rest_min * (1.0 + z)
        wave_obs_max = wave_rest_max * (1.0 + z)
        wave_obs = jnp.logspace(jnp.log10(wave_obs_min), jnp.log10(wave_obs_max), 100)

        model_unchunked = SEDModel(
            spec,
            ssp,
            wave_chunk_size=None,
            observation=Observation(spectroscopy=Spectroscopy(wave_obs=wave_obs)),
        )
        model_chunked = SEDModel(
            spec,
            ssp,
            wave_chunk_size=32,
            observation=Observation(spectroscopy=Spectroscopy(wave_obs=wave_obs)),
        )

        params = model_unchunked.spec.sample(jax.random.PRNGKey(101))

        flux_u = model_unchunked.predict_spectrum(params, wave_obs)
        flux_c = model_chunked.predict_spectrum(params, wave_obs)

        np.testing.assert_allclose(
            flux_c,
            flux_u,
            rtol=1e-12,
            atol=1e-15,
            err_msg=f"Mismatch for wave_rest_range={wave_rest_range}",
        )
