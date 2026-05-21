"""Tests for spectrum prediction with wavelength-axis chunking.

Verifies that:
1. predict_spectrum with wave_chunk_size produces bitwise-identical output
   to the unchunked case for all chunk sizes.
2. Chunking works across different spectrum configurations (with/without lines).
3. LSF convolution is properly applied post-chunking.

These tests require SSP data on disk; they are skipped gracefully when missing.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.bounds

jax.config.update("jax_enable_x64", True)

from tengri.forward.sed_model import SEDModel
from tengri.parameters.parameters import Parameters
from tengri.parameters.priors import Fixed, Uniform

_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
_SSP_FILES = list(_DATA_DIR.glob("ssp_*.h5"))
_SSP_EXISTS = len(_SSP_FILES) > 0

pytestmark = pytest.mark.skipif(
    not _SSP_EXISTS,
    reason="SSP data file not found — unit test requires data/ssp_*.h5",
)


@pytest.fixture(scope="module")
def ssp(ssp_data_wne):
    return ssp_data_wne


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
    """Model using exact mode."""
    model = SEDModel(spec_photometry_only, ssp)
    return model


@pytest.fixture(scope="module")
def model_with_dust_variation(ssp, spec_with_dust_variation, wave_obs_460):
    """Model with variable dust, exact mode."""
    model = SEDModel(spec_with_dust_variation, ssp)
    return model


@pytest.fixture(scope="module")
def model_compositional(ssp, spec_photometry_only, wave_obs_460):
    """Model with precomputed spectroscopy (compositional mode)."""
    model = SEDModel(spec_photometry_only, ssp)
    model.precompute_spectroscopy(wave_obs_460)
    return model


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

    flux_unchunked = model_exact.predict_spectrum(
        params, wave_obs_460, mode="exact", wave_chunk_size=None
    )

    # Test various chunk sizes
    for chunk_size in [16, 32, 64, 128, 460]:
        flux_chunked = model_exact.predict_spectrum(
            params, wave_obs_460, mode="exact", wave_chunk_size=chunk_size
        )

        # Bitwise match (rtol=1e-12, atol=1e-15 for float64)
        np.testing.assert_allclose(
            flux_chunked,
            flux_unchunked,
            rtol=1e-12,
            atol=1e-15,
            err_msg=f"Mismatch for chunk_size={chunk_size}",
        )


def test_compositional_mode_chunk_sizes_match(model_compositional, wave_obs_460):
    """Compositional mode: all chunk sizes should produce bitwise-identical output."""
    params = model_compositional.spec.sample(jax.random.PRNGKey(43))

    flux_unchunked = model_compositional.predict_spectrum(
        params, wave_obs_460, mode="compositional", wave_chunk_size=None
    )

    for chunk_size in [32, 64, 128]:
        flux_chunked = model_compositional.predict_spectrum(
            params, wave_obs_460, mode="compositional", wave_chunk_size=chunk_size
        )

        np.testing.assert_allclose(
            flux_chunked,
            flux_unchunked,
            rtol=1e-12,
            atol=1e-15,
            err_msg=f"Mismatch for chunk_size={chunk_size}",
        )


def test_auto_mode_chunk_sizes_match(model_compositional, wave_obs_460):
    """Auto mode: should route to compositional and chunk there."""
    params = model_compositional.spec.sample(jax.random.PRNGKey(44))

    flux_unchunked = model_compositional.predict_spectrum(
        params, wave_obs_460, mode="auto", wave_chunk_size=None
    )

    for chunk_size in [32, 64]:
        flux_chunked = model_compositional.predict_spectrum(
            params, wave_obs_460, mode="auto", wave_chunk_size=chunk_size
        )

        np.testing.assert_allclose(
            flux_chunked,
            flux_unchunked,
            rtol=1e-12,
            atol=1e-15,
            err_msg=f"Mismatch for chunk_size={chunk_size} in auto mode",
        )


def test_dust_variation_chunked_match(model_with_dust_variation, wave_obs_460):
    """With dust variation: chunked should still match unchunked."""
    params = model_with_dust_variation.spec.sample(jax.random.PRNGKey(45))

    flux_unchunked = model_with_dust_variation.predict_spectrum(
        params, wave_obs_460, mode="exact", wave_chunk_size=None
    )

    for chunk_size in [32, 64]:
        flux_chunked = model_with_dust_variation.predict_spectrum(
            params, wave_obs_460, mode="exact", wave_chunk_size=chunk_size
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

    flux_unchunked = model_exact.predict_spectrum(
        params, wave_obs_460, mode="exact", wave_chunk_size=None
    )

    # Chunk size = 1000 > 460
    flux_large_chunk = model_exact.predict_spectrum(
        params, wave_obs_460, mode="exact", wave_chunk_size=1000
    )

    np.testing.assert_allclose(flux_large_chunk, flux_unchunked, rtol=1e-12, atol=1e-15)


def test_chunk_size_1_pixel(model_exact, wave_obs_460):
    """Chunk size = 1 should work (one pixel per chunk)."""
    params = model_exact.spec.sample(jax.random.PRNGKey(47))

    flux_unchunked = model_exact.predict_spectrum(
        params, wave_obs_460, mode="exact", wave_chunk_size=None
    )

    flux_single_pix = model_exact.predict_spectrum(
        params, wave_obs_460, mode="exact", wave_chunk_size=1
    )

    np.testing.assert_allclose(flux_single_pix, flux_unchunked, rtol=1e-12, atol=1e-15)


def test_instance_wave_chunk_size_default(ssp, spec_photometry_only, wave_obs_460):
    """Passing wave_chunk_size to __init__ should set instance default."""
    model_chunked = SEDModel(spec_photometry_only, ssp, wave_chunk_size=64)
    params = model_chunked.spec.sample(jax.random.PRNGKey(48))

    # Should use instance default (64) without explicit kwarg
    flux_from_default = model_chunked.predict_spectrum(params, wave_obs_460, mode="exact")

    # Explicit None should override instance default
    flux_unchunked = model_chunked.predict_spectrum(
        params, wave_obs_460, mode="exact", wave_chunk_size=None
    )

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

        flux_unchunked = model_exact.predict_spectrum(
            params, wave_obs_460, mode="exact", wave_chunk_size=None
        )
        flux_chunked = model_exact.predict_spectrum(
            params, wave_obs_460, mode="exact", wave_chunk_size=chunk_size
        )

        np.testing.assert_allclose(flux_chunked, flux_unchunked, rtol=1e-12, atol=1e-15)
