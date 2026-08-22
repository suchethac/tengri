# SPDX-License-Identifier: BSD-3-Clause
"""Regression: a *fresh* multi-chain HMC run must sample all ``n_chains`` chains.

Bug: ``Fitter.run(method="mcmc_hmc", n_chains=N)`` on a first (cold-cache)
call silently sampled a **single** chain — the ``else`` branch of
``run_hmc`` ran ``_hmc_full_scan`` (one chain, warmup+sampling bundled) and
returned, while still reporting ``diagnostics["n_chains"] = N``. The genuine
vmap/pmap multi-chain path lived only in the *cached-warmup* branch, so it
fired on the 2nd+ call. Effect: a notebook's "N chains, R-hat X" was really a
split-R-hat on one chain — a silent correctness failure in the convergence
diagnostic.

Fix: split warmup (``_hmc_warmup_only``) from sampling so the first call
adapts once and then dispatches through the ``n_chains``-aware sampler.

These tests assert the posterior actually contains ``n_chains * n_samples``
draws on a FRESH fitter (no warm cache), which is only true once the bug is
fixed.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

pytestmark = pytest.mark.regression_bug

try:
    import blackjax  # noqa: F401

    HAS_BLACKJAX = True
except ImportError:
    HAS_BLACKJAX = False


def _tiny_model_and_data(synthetic_ssp_wide, synthetic_tophat_obs):
    """A small free-parameter model + one noisy synthetic photometry vector."""
    import tengri
    from tengri import FREE, Fixed, SEDModel, Uniform

    model = SEDModel.build(
        ssp_data=synthetic_ssp_wide,
        observation=synthetic_tophat_obs,
        redshift=Fixed(0.1),
        sfh={"type": "dpl", "all_params": FREE},
        met={"logzsol": Uniform(-1.5, 0.3)},
    )
    truth = model.spec.sample(jax.random.PRNGKey(3))
    flux = np.abs(np.asarray(model.predict_photometry(truth)))
    return tengri, model, flux


@pytest.mark.skipif(not HAS_BLACKJAX, reason="blackjax not available")
@pytest.mark.parametrize("n_chains", [1, 2, 3])
def test_fresh_hmc_samples_all_chains(synthetic_ssp_wide, synthetic_tophat_obs, n_chains):
    """Fresh run must yield n_chains * n_samples draws (not one chain's worth)."""
    tengri, model, flux = _tiny_model_and_data(synthetic_ssp_wide, synthetic_tophat_obs)
    n_samples = 25
    # FRESH fitter — no prior run has populated the warmup cache.
    fitter = tengri.Fitter(model, flux, flux / 20.0, data_type="photometry")
    post = fitter.run(
        "mcmc_hmc",
        key=jax.random.PRNGKey(0),
        n_warmup=60,
        n_samples=n_samples,
        n_burnin=0,
        n_chains=n_chains,
        n_leapfrog_steps=5,
        dense_mass_matrix=True,
        verbose=False,
    )
    first = np.asarray(next(iter(post.samples.values())))
    assert first.shape[0] == n_chains * n_samples, (
        f"fresh n_chains={n_chains} yielded {first.shape[0]} draws, "
        f"expected {n_chains * n_samples} (single-chain regression)"
    )


@pytest.mark.skipif(not HAS_BLACKJAX, reason="blackjax not available")
@pytest.mark.parametrize("chain_method", ["vmap", "sequential"])
def test_chain_method_yields_all_chains(synthetic_ssp_wide, synthetic_tophat_obs, chain_method):
    """Both executors produce n_chains genuine chains; 'sequential' is the
    memory-frugal path (peak = one chain) for cheap hardware."""
    tengri, model, flux = _tiny_model_and_data(synthetic_ssp_wide, synthetic_tophat_obs)
    n_chains, n_samples = 3, 20
    fitter = tengri.Fitter(model, flux, flux / 20.0, data_type="photometry")
    post = fitter.run(
        "mcmc_hmc",
        key=jax.random.PRNGKey(0),
        n_warmup=60,
        n_samples=n_samples,
        n_burnin=0,
        n_chains=n_chains,
        n_leapfrog_steps=5,
        dense_mass_matrix=True,
        chain_method=chain_method,
        verbose=False,
    )
    first = np.asarray(next(iter(post.samples.values())))
    assert first.shape[0] == n_chains * n_samples


@pytest.mark.skipif(not HAS_BLACKJAX, reason="blackjax not available")
def test_chain_method_parallel_falls_back_without_devices(
    synthetic_ssp_wide, synthetic_tophat_obs
):
    """chain_method='parallel' warns and falls back to vmap when too few devices.

    With the default single CPU device, requesting parallel chains cannot use
    pmap; it must degrade to vmap (still producing all chains), not crash.
    """
    tengri, model, flux = _tiny_model_and_data(synthetic_ssp_wide, synthetic_tophat_obs)
    n_chains, n_samples = 2, 20
    fitter = tengri.Fitter(model, flux, flux / 20.0, data_type="photometry")
    with pytest.warns(RuntimeWarning, match="parallel"):
        post = fitter.run(
            "mcmc_hmc",
            key=jax.random.PRNGKey(0),
            n_warmup=60,
            n_samples=n_samples,
            n_burnin=0,
            n_chains=n_chains,
            n_leapfrog_steps=5,
            dense_mass_matrix=True,
            chain_method="parallel",
            verbose=False,
        )
    first = np.asarray(next(iter(post.samples.values())))
    assert first.shape[0] == n_chains * n_samples
