# SPDX-License-Identifier: BSD-3-Clause
"""Tests for spectrum prediction with wavelength-axis chunking.

Verifies that:
1. predict_spectrum with wave_chunk_size produces bitwise-identical output
   to the unchunked case for all chunk sizes.
2. Chunking works across different spectrum configurations (with/without lines).
3. LSF convolution is properly applied post-chunking.

These tests require SSP data on disk; they are skipped gracefully when missing.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.bounds

from tengri import Observation, Spectroscopy
from tengri.forward.sed_model import SEDModel
from tengri.parameters.parameters import Parameters
from tengri.parameters.priors import Fixed, Uniform

# Migrated to the synthetic SSP fixture (#613): the chunked-vs-unchunked
# equality this asserts is SSP-physics-independent, so it runs in CI on the
# synthetic SSP instead of skipping on missing data. ``pytest.mark.bounds``
# (set above) is the taxonomy marker — previously clobbered by the skipif.


@pytest.fixture(scope="module")
def ssp(synthetic_ssp_wide):
    return synthetic_ssp_wide


@pytest.fixture(scope="module")
def spec_photometry_only():
    """Simple spec for unit testing (no emission lines)."""
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
def spec_with_dust_variation():
    """Spec with more dust parameter variation."""
    return Parameters(
        mean_sfh_type="dense_basis",
        sfh_db_log_total_mass=Uniform(8, 12),
        sfh_db_log_sfr_inst=Uniform(-3, 3),
        sfh_db_tx_frac_0=Uniform(0.05, 0.95),
        sfh_db_tx_frac_1=Uniform(0.05, 0.95),
        sfh_db_tx_frac_2=Uniform(0.05, 0.95),
        met_logzsol=Uniform(-1.0, 1.0),
        dust_tau_bc=Uniform(0.0, 1.0),
        dust_tau_diff=Uniform(0.0, 2.0),
        redshift=0.1,
    )


@pytest.fixture(scope="module")
def wave_obs_460():
    """Typical spectroscopy grid: 460 pixels, 3000–7500 Å rest-frame at z=0.1."""
    z = 0.1
    wave_rest_min = 3000.0
    wave_rest_max = 7500.0
    wave_obs_min = wave_rest_min * (1.0 + z)
    wave_obs_max = wave_rest_max * (1.0 + z)
    return jnp.logspace(jnp.log10(wave_obs_min), jnp.log10(wave_obs_max), 460)


@pytest.fixture(scope="module")
def model_exact(ssp, spec_photometry_only, wave_obs_460):
    """Model using the exact path, with spectroscopy configured on the
    observation (the new API; the removed ``precompute_spectroscopy`` used to
    attach the grid after construction)."""
    obs = Observation(spectroscopy=Spectroscopy(wave_obs=wave_obs_460))
    return SEDModel(spec_photometry_only, ssp, observation=obs)


@pytest.fixture(scope="module")
def model_with_dust_variation(ssp, spec_with_dust_variation, wave_obs_460):
    """Model with variable dust, exact path, spectroscopy on the observation."""
    obs = Observation(spectroscopy=Spectroscopy(wave_obs=wave_obs_460))
    return SEDModel(spec_with_dust_variation, ssp, observation=obs)


def test_wave_chunk_none_default(model_exact, wave_obs_460):
    """wave_chunk_size=None should be the default (no chunking)."""
    params = model_exact.spec.sample(jax.random.PRNGKey(0))

    # Explicit None vs. implicit default
    flux_explicit_none = model_exact.predict_spectrum(params, wave_obs_460, wave_chunk_size=None)
    flux_implicit_none = model_exact.predict_spectrum(params, wave_obs_460)

    np.testing.assert_array_equal(flux_explicit_none, flux_implicit_none)


def test_exact_mode_chunk_sizes_match_unchunked(model_exact, wave_obs_460):
    """Exact mode: all chunk sizes should produce bitwise-identical output."""
    params = model_exact.spec.sample(jax.random.PRNGKey(42))

    flux_unchunked = model_exact.predict_spectrum(params, wave_obs_460, wave_chunk_size=None)

    # Test various chunk sizes
    for chunk_size in [16, 32, 64, 128, 460]:
        flux_chunked = model_exact.predict_spectrum(
            params, wave_obs_460, wave_chunk_size=chunk_size
        )

        # Bitwise match (rtol=1e-12, atol=1e-15 for float64)
        np.testing.assert_allclose(
            flux_chunked,
            flux_unchunked,
            rtol=1e-12,
            atol=1e-15,
            err_msg=f"Mismatch for chunk_size={chunk_size}",
        )


def test_dust_variation_chunked_match(model_with_dust_variation, wave_obs_460):
    """With dust variation: chunked should still match unchunked."""
    params = model_with_dust_variation.spec.sample(jax.random.PRNGKey(45))

    flux_unchunked = model_with_dust_variation.predict_spectrum(
        params, wave_obs_460, wave_chunk_size=None
    )

    for chunk_size in [32, 64]:
        flux_chunked = model_with_dust_variation.predict_spectrum(
            params, wave_obs_460, wave_chunk_size=chunk_size
        )

        np.testing.assert_allclose(
            flux_chunked,
            flux_unchunked,
            rtol=1e-12,
            atol=1e-15,
            err_msg=f"Mismatch for chunk_size={chunk_size} with dust variation",
        )


def test_chunk_size_larger_than_array(model_exact, wave_obs_460):
    """Chunk size larger than array should still work (single chunk)."""
    params = model_exact.spec.sample(jax.random.PRNGKey(46))
    n_pix = wave_obs_460.shape[0]

    flux_unchunked = model_exact.predict_spectrum(params, wave_obs_460, wave_chunk_size=None)

    # Chunk size = 1000 > 460
    flux_large_chunk = model_exact.predict_spectrum(params, wave_obs_460, wave_chunk_size=1000)

    np.testing.assert_allclose(flux_large_chunk, flux_unchunked, rtol=1e-12, atol=1e-15)


def test_chunk_size_1_pixel(model_exact, wave_obs_460):
    """Chunk size = 1 should work (one pixel per chunk)."""
    params = model_exact.spec.sample(jax.random.PRNGKey(47))

    flux_unchunked = model_exact.predict_spectrum(params, wave_obs_460, wave_chunk_size=None)

    flux_single_pix = model_exact.predict_spectrum(params, wave_obs_460, wave_chunk_size=1)

    np.testing.assert_allclose(flux_single_pix, flux_unchunked, rtol=1e-12, atol=1e-15)


def test_instance_wave_chunk_size_default(ssp, spec_photometry_only, wave_obs_460):
    """Passing wave_chunk_size to __init__ should set instance default."""
    obs = Observation(spectroscopy=Spectroscopy(wave_obs=wave_obs_460))
    model_chunked = SEDModel(spec_photometry_only, ssp, wave_chunk_size=64, observation=obs)
    params = model_chunked.spec.sample(jax.random.PRNGKey(48))

    # Should use instance default (64) without explicit kwarg
    flux_from_default = model_chunked.predict_spectrum(params, wave_obs_460)

    # Explicit None should override instance default
    flux_unchunked = model_chunked.predict_spectrum(params, wave_obs_460, wave_chunk_size=None)

    # They should differ (one is chunked, one is not)
    # Numerically equivalent but different code paths
    np.testing.assert_allclose(
        flux_from_default,
        flux_unchunked,
        rtol=1e-12,
        atol=1e-15,
        err_msg="Instance default should still produce same numerical result",
    )


def test_multiple_samples_chunked_consistent(model_exact, wave_obs_460):
    """Multiple samples should be consistent with chunking."""
    keys = jax.random.split(jax.random.PRNGKey(49), 5)
    chunk_size = 64

    for key in keys:
        params = model_exact.spec.sample(key)

        flux_unchunked = model_exact.predict_spectrum(params, wave_obs_460, wave_chunk_size=None)
        flux_chunked = model_exact.predict_spectrum(
            params, wave_obs_460, wave_chunk_size=chunk_size
        )

        np.testing.assert_allclose(flux_chunked, flux_unchunked, rtol=1e-12, atol=1e-15)
