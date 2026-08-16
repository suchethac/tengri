# SPDX-License-Identifier: BSD-3-Clause
"""Integration test for HMC structural sharing across Fitter instances (Phase B).

Tests verify that smart-lean (the default) keeps the L3 entry whose key
matches the current run's ``(compile_signature, method)``. We measure
this by inspecting cache state directly — runtime-based speedup tests
are dominated by sampling cost on small models and noisy enough to be
unreliable detectors of cache reuse.
"""

from __future__ import annotations

import jax
import pytest

import tengri
from tengri.inference.jit_engine import _SHARED_LOGDENSITY_FN_CACHE


@pytest.mark.integration
def test_hmc_structural_reuse_same_model():
    """Two Fitters with same model reuse HMC compile (second run faster)."""
    # Cold start: clear all caches so the first fitter.run() pays the full
    # compile cost (otherwise this test is order-dependent on the suite).
    tengri.gc()
    # Load data
    try:
        ssp = tengri.load_ssp_data("data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5")
    except FileNotFoundError:
        pytest.skip("SSP data not available")

    filters = tengri.load_filter_set(
        ["hst_f606w", "hst_f775w", "hst_f814w", "hst_f850lp", "vista_ks", "irac_36"]
    )
    obs = tengri.observation.Observation(
        photometry=tengri.observation.Photometry.from_filter_set(filters)
    )

    # Create model and synthetic data
    spec = tengri.Parameters(
        mean_sfh_type="dpl",
        sfh_dpl_alpha=tengri.Uniform(0.5, 4.0),
        sfh_dpl_beta=tengri.Uniform(0.5, 4.0),
        sfh_dpl_tau_gyr=tengri.Uniform(0.5, 12.0),
        sfh_dpl_log_total_mass=tengri.Uniform(7.0, 12.5),
        met_logzsol=tengri.Uniform(-2.0, 0.2),
        dust_law_bc="calzetti",
        dust_tau_bc=tengri.Uniform(0, 3),
        nebular_ssp=False,
        apply_igm=False,
        redshift=tengri.Fixed(0.5),
    )

    model = tengri.SEDModel(spec, ssp, observation=obs)

    # Generate synthetic photometry
    truth = spec.sample(jax.random.PRNGKey(0))
    phot_true = model.predict_photometry(truth)
    phot_err = 0.05 * phot_true
    key = jax.random.PRNGKey(1)
    phot_obs = phot_true + 0.01 * phot_err * jax.random.normal(key, phot_true.shape)

    # Build two Fitters with the same model and data
    fitter1 = tengri.Fitter(model, phot_obs, phot_err, data_type="photometry")
    fitter2 = tengri.Fitter(model, phot_obs, phot_err, data_type="photometry")

    # Verify they have the same compile signature
    assert fitter1.compile_signature() == fitter2.compile_signature(), (
        "Fitters should have identical signatures"
    )

    # No persistent() wrap — smart lean (default) keeps the L2
    # logdensity cache (preserved unconditionally at inference_body
    # scope) so the second fitter's HMC reuses the compiled closure.
    result1 = fitter1.run(
        "mcmc_hmc",
        n_warmup=100,
        n_samples=100,
        dense_mass_matrix=False,
        key=jax.random.PRNGKey(0),
    )
    logdensity_after_first = dict(_SHARED_LOGDENSITY_FN_CACHE)
    assert len(logdensity_after_first) >= 1, "First run should have populated L2 logdensity cache"

    result2 = fitter2.run(
        "mcmc_hmc",
        n_warmup=100,
        n_samples=100,
        dense_mass_matrix=False,
        key=jax.random.PRNGKey(1),
    )
    logdensity_after_second = dict(_SHARED_LOGDENSITY_FN_CACHE)

    # The matching entry must survive across the second run's lean clear,
    # AND the cached function must be the same object (so JAX's internal
    # JIT cache hits — function identity is its primary key).
    common_keys = set(logdensity_after_first) & set(logdensity_after_second)
    assert common_keys, "L2 logdensity cache lost all entries across runs"
    for k in common_keys:
        assert logdensity_after_first[k] is logdensity_after_second[k], (
            "L2 logdensity entry replaced rather than reused — JAX JIT will recompile"
        )

    # Both results should be valid
    assert result1.samples is not None
    assert result2.samples is not None
    assert next(iter(result1.samples.values())).shape[0] == 100
    assert next(iter(result2.samples.values())).shape[0] == 100


@pytest.mark.integration
def test_hmc_structural_reuse_different_data():
    """Two Fitters with same model but different data reuse HMC compile."""
    tengri.gc()
    try:
        ssp = tengri.load_ssp_data("data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5")
    except FileNotFoundError:
        pytest.skip("SSP data not available")

    filters = tengri.load_filter_set(["hst_f606w", "hst_f775w", "hst_f814w", "hst_f850lp"])
    obs = tengri.observation.Observation(
        photometry=tengri.observation.Photometry.from_filter_set(filters)
    )

    spec = tengri.Parameters(
        mean_sfh_type="dpl",
        sfh_dpl_alpha=tengri.Uniform(0.5, 4.0),
        sfh_dpl_beta=tengri.Uniform(0.5, 4.0),
        sfh_dpl_tau_gyr=tengri.Uniform(0.5, 12.0),
        sfh_dpl_log_total_mass=tengri.Uniform(7.0, 12.5),
        met_logzsol=tengri.Uniform(-2.0, 0.2),
        dust_law_bc="calzetti",
        dust_tau_bc=tengri.Uniform(0, 3),
        nebular_ssp=False,
        apply_igm=False,
        redshift=tengri.Fixed(0.5),
    )

    model = tengri.SEDModel(spec, ssp, observation=obs)

    # Generate two different synthetic datasets
    truth1 = spec.sample(jax.random.PRNGKey(0))
    phot_true1 = model.predict_photometry(truth1)
    phot_err1 = 0.05 * phot_true1
    key1 = jax.random.PRNGKey(10)
    phot_obs1 = phot_true1 + 0.01 * phot_err1 * jax.random.normal(key1, phot_true1.shape)

    truth2 = spec.sample(jax.random.PRNGKey(5))
    phot_true2 = model.predict_photometry(truth2)
    phot_err2 = 0.05 * phot_true2
    key2 = jax.random.PRNGKey(11)
    phot_obs2 = phot_true2 + 0.01 * phot_err2 * jax.random.normal(key2, phot_true2.shape)

    # Build Fitters with different data
    fitter1 = tengri.Fitter(model, phot_obs1, phot_err1, data_type="photometry")
    fitter2 = tengri.Fitter(model, phot_obs2, phot_err2, data_type="photometry")

    # They should have the same compile signature (same model, same data shape)
    assert fitter1.compile_signature() == fitter2.compile_signature(), (
        "Fitters with same model and data shape should have identical signatures"
    )

    # No persistent() wrap — different data, same model+method must
    # still share the L2 logdensity compile (data is a traced argument).
    result1 = fitter1.run(
        "mcmc_hmc",
        n_warmup=100,
        n_samples=100,
        dense_mass_matrix=False,
        key=jax.random.PRNGKey(0),
    )
    logdensity_after_first = dict(_SHARED_LOGDENSITY_FN_CACHE)

    result2 = fitter2.run(
        "mcmc_hmc",
        n_warmup=100,
        n_samples=100,
        dense_mass_matrix=False,
        key=jax.random.PRNGKey(1),
    )
    logdensity_after_second = dict(_SHARED_LOGDENSITY_FN_CACHE)

    common_keys = set(logdensity_after_first) & set(logdensity_after_second)
    assert common_keys, "L2 logdensity cache lost all entries across runs"
    for k in common_keys:
        assert logdensity_after_first[k] is logdensity_after_second[k], (
            "L2 logdensity entry replaced — JAX would recompile despite shared key"
        )

    assert result1.samples is not None
    assert result2.samples is not None
