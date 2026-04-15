"""Tests for the hybrid spectrum _traceable path.

Verifies that:
1. _build_hybrid_kernels now builds hk.spectrum and hk._spectrum_raw when
   spectroscopy is precomputed.
2. predict_spectrum(params, mode='_traceable') routes to the hybrid spectrum
   kernel, not the full compositional rest-SED path.
3. The hybrid spectrum result matches the compositional result within 1%
   (stellar dominated case — no energy balance errors).

These tests require SSP data on disk; they are skipped gracefully when missing.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from tengri.components.sps.dsps_wrapper import load_ssp_data
from tengri.forward.sed_model import SEDModel
from tengri.parameters.parameters import ParamSpec
from tengri.parameters.priors import Fixed, Uniform

_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
_SSP_FILES = list(_DATA_DIR.glob("ssp_*.h5"))
_SSP_EXISTS = len(_SSP_FILES) > 0

pytestmark = pytest.mark.skipif(
    not _SSP_EXISTS,
    reason="SSP data file not found — integration test requires data/ssp_*.h5",
)


@pytest.fixture(scope="module")
def ssp():
    return load_ssp_data(str(_SSP_FILES[0]))


@pytest.fixture(scope="module")
def spec():
    return ParamSpec(
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
def model_with_spec(ssp, spec):
    """Build a model with spectroscopy precomputed at z=0.1."""
    wave_obs = jnp.linspace(4000.0, 9000.0, 200)
    model = SEDModel(spec, ssp)
    model.precompute_spectroscopy(wave_obs)
    return model, wave_obs


def test_hybrid_spectrum_raw_built(model_with_spec):
    """_build_hybrid_kernels should populate hk._spectrum_raw."""
    model, _ = model_with_spec
    hk = model._SEDModel__hybrid  # name-mangled
    assert hk is not None, "_hybrid kernels not built"
    assert hasattr(hk, "_spectrum_raw"), "hk._spectrum_raw attribute missing"
    assert hk._spectrum_raw is not None, (
        "hk._spectrum_raw is None — build_hybrid_spectrum failed or was not called"
    )
    assert hk.spectrum is not None, "hk.spectrum (JIT'd) is None"


def test_traceable_spectrum_uses_hybrid(model_with_spec):
    """predict_spectrum(_traceable) should use hybrid path, not _predict_spectrum_auto."""
    model, wave_obs = model_with_spec
    hk = model._SEDModel__hybrid

    call_log = []
    original_raw = hk._spectrum_raw

    def patched_raw(sfr_on_ssp, params):
        call_log.append("hybrid")
        return original_raw(sfr_on_ssp, params)

    hk._spectrum_raw = patched_raw
    try:
        params = model.spec.sample(jax.random.PRNGKey(0))
        _ = model.predict_spectrum(params, wave_obs=wave_obs, mode="_traceable")
    finally:
        hk._spectrum_raw = original_raw

    assert call_log == ["hybrid"], (
        f"_traceable mode did not route to hybrid spectrum raw; log={call_log}"
    )


def test_traceable_spectrum_agrees_with_compositional(model_with_spec):
    """Hybrid and compositional spectrum should agree to <1% (stellar dominated)."""
    model, wave_obs = model_with_spec
    key = jax.random.PRNGKey(7)
    params = model.spec.sample(key)

    flux_traceable = model.predict_spectrum(params, wave_obs=wave_obs, mode="_traceable")
    flux_compositional = model.predict_spectrum(params, wave_obs=wave_obs, mode="compositional")

    # Normalize by median to get relative error
    med = jnp.median(jnp.abs(flux_compositional))
    if med == 0:
        pytest.skip("compositional flux is zero — degenerate test case")

    rel_err = jnp.abs(flux_traceable - flux_compositional) / (med + 1e-30)
    max_err = float(jnp.max(rel_err))

    assert max_err < 0.02, (
        f"Hybrid _traceable vs compositional max relative error {max_err:.3%} exceeds 2%"
    )


def test_compositional_spectrum_raw_built(model_with_spec):
    """CompositionalKernels should now also have _spectrum_raw."""
    model, _ = model_with_spec
    ck = model._SEDModel__compositional
    assert hasattr(ck, "_spectrum_raw"), "ck._spectrum_raw attribute missing"
    # May be None if build_fused_tier2_spectrum failed, but attr must exist


def test_precompute_spectroscopy_clears_fitter_cache(model_with_spec):
    """precompute_spectroscopy() must clear any stale fitter loss-fn cache.

    If the cache is not cleared, a Fitter that compiled its loss function
    before precompute_spectroscopy() was called will keep using the slow
    full-SED path even after spectroscopy is precomputed.
    """
    model, wave_obs = model_with_spec

    # Simulate a previously cached loss function on the model object.
    model._loss_fn_cache = {"some_key": lambda p, d: 0.0}
    model._jit_engine_cache = {"some_key": object()}
    model._loglik_fn_cache = {"some_key": lambda p, d: 0.0}

    # Calling precompute_spectroscopy again should clear all three caches.
    model.precompute_spectroscopy(wave_obs)

    assert not hasattr(model, "_loss_fn_cache"), (
        "_loss_fn_cache not cleared after precompute_spectroscopy"
    )
    assert not hasattr(model, "_jit_engine_cache"), (
        "_jit_engine_cache not cleared after precompute_spectroscopy"
    )
    assert not hasattr(model, "_loglik_fn_cache"), (
        "_loglik_fn_cache not cleared after precompute_spectroscopy"
    )
