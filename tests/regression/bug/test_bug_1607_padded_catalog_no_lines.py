# SPDX-License-Identifier: BSD-3-Clause
"""Regression test for #1607: batched MCMC padding breaks with no line columns.

Issue: in the n_pad_extra > 0 branch, n_line_cols == 0 leaves all_line_flux/err
at shape (n_gal, 0) while siblings pad to (n_padded,), and all arrays feed
jax.lax.map which needs a common leading dimension.

Fix: unconditionally build pad_line even for zero-width line arrays, then
concatenate.

Runs on the synthetic SSP (CI-runnable). Auto-marked slow by tests/inference.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

pytestmark = pytest.mark.regression_bug


def _build_model_no_lines(synthetic_ssp, simple_observation):
    """Build a photometry-only model (no line_cols)."""
    from tengri import Fixed, Parameters, SEDModel, Uniform

    spec = Parameters(
        sfh_dpl_log_total_mass=Uniform(7.0, 12.5),
        sfh_dpl_alpha=Fixed(2.0),
        sfh_dpl_age_gyr=Fixed(5.0),
        sfh_dpl_beta=Fixed(2.0),
        sfh_dpl_tau_gyr=Fixed(3.0),
        met_logzsol=Fixed(0.0),
        dust_tau_bc=Fixed(0.3),
        dust_tau_diff=Fixed(0.2),
        redshift=Fixed(0.1),
        mean_sfh_type="dpl",
    )
    # simple_observation is photometry-only, so no line_cols
    return SEDModel(spec, synthetic_ssp, observation=simple_observation)


def _catalog_from_truths(model, truths, key, noise_frac=0.02):
    """Inject galaxies with the given free-param truths."""
    galaxies = []
    for i, overrides in enumerate(truths):
        k = jax.random.fold_in(key, i)
        true_params = dict(model.spec.sample(k))
        true_params.update(overrides)
        flux = model.predict_photometry(true_params)
        noise = jnp.abs(flux) * noise_frac
        flux_obs = flux + noise * jax.random.normal(jax.random.fold_in(k, 1), shape=flux.shape)
        galaxies.append({"flux_obs": flux_obs, "noise": noise})
    return galaxies


def test_mcmc_nuts_no_line_cols_with_padding(synthetic_ssp, simple_observation):
    """MCMC NUTS on photometry-only catalog with n_gal not divisible by chunk size.

    n_gal=3 with forward_chunk_size=4 forces n_pad_extra=1, triggering the
    n_line_cols==0 branch. Before the fix, shape mismatch in jax.lax.map.
    """
    from tengri import CatalogFitter

    model = _build_model_no_lines(synthetic_ssp, simple_observation)

    # 3 galaxies, one per mass decade
    truths = [
        {"sfh_dpl_log_total_mass": jnp.array(9.0)},
        {"sfh_dpl_log_total_mass": jnp.array(10.0)},
        {"sfh_dpl_log_total_mass": jnp.array(11.0)},
    ]
    galaxies = _catalog_from_truths(model, truths, jax.random.PRNGKey(42))

    cat = CatalogFitter(model, galaxies, data_type="photometry")

    # This should complete without shape errors. Tiny n_warmup/n_samples for speed.
    key = jax.random.PRNGKey(0)
    cp = cat.run(
        "mcmc_nuts",
        key=key,
        forward_chunk_size=4,  # > 3, forces padding
        n_warmup=50,
        n_samples=50,
        verbose=False,
    )

    # Verify we got posteriors for all real galaxies
    assert cp.n_galaxies == 3
    assert len(cp.posteriors) == 3
    for i in range(3):
        gal_post = cp[i]
        assert gal_post.samples is not None
        # samples is a dict; check one param for correct shape
        mass_samples = gal_post.samples["sfh_dpl_log_total_mass"]
        assert mass_samples.shape[0] == 50  # n_samples
