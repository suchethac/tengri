# SPDX-License-Identifier: BSD-3-Clause
"""Batch-MCMC vmap kernel: adaptation threaded as args, compiled kernel reused.

Sibling of the batch-MAP memo (test_batch_map_kernel_memo.py). ``_fit_batch_vmap_mcmc``
built a fresh ``jax.jit(jax.vmap(single_galaxy))`` on every ``fit_batch`` call, and
``single_galaxy`` closure-captured the adapted ``step_size`` / ``inv_mass_matrix`` —
so the kernel was both baked (adaptation as XLA Constants) and recompiled on every
call. A catalog processed in repeated same-shape ``fit_batch(mcmc_*)`` calls thus
recompiled the vmapped sampler each time.

The fix threads the adaptation through ``jax.vmap(..., in_axes=(0, 0, 0, None, None))``
as broadcast arguments (Parameter ops, not Constants) and memoizes the wrapper on a
structural key. This test pins:

* the vmap-MCMC kernel is reused across two same-config calls (memo hit — no rebuild),
* a different structural config builds a distinct kernel,
* results stay finite, correctly shaped, and deterministic for a fixed key
  (threading moved WHERE adaptation enters, not the numbers).

Runs on the synthetic wide SSP + tophat filters (no ``data/`` files needed, #613);
the fixed-z photometry model satisfies ``has_fixedz_photometry_precompute`` so
``fit_batch`` takes the vmap-MCMC path.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import FIXED, Fixed, Observation, Photometry, SEDModel, Uniform
from tengri.inference.fitter import Fitter
from tengri.observation.photometry import FilterCurve

pytestmark = pytest.mark.contract

_MCMC_KW = dict(n_warmup=15, n_burnin=5, n_samples=12, verbose=False)


def _tophat(center, frac=0.16, n=40):
    wave = jnp.linspace(center * (1.0 - frac), center * (1.0 + frac), n)
    trans = jnp.sin(jnp.linspace(0.0, jnp.pi, n)) * 0.6
    return FilterCurve(wave=wave, trans=trans, name=f"b{int(center)}")


@pytest.fixture(scope="module")
def fitter(synthetic_ssp_wide):
    obs = Observation(
        photometry=Photometry(filters=tuple(_tophat(c) for c in (3500.0, 4800.0, 6200.0)))
    )
    model = SEDModel.build(
        ssp_data=synthetic_ssp_wide,
        observation=obs,
        sfh={"type": "dpl", "all_params": FIXED, "log_total_mass": Uniform(8.0, 12.0)},
        dust={
            "type": "two_component",
            "law_bc": "calzetti",
            "all_params": FIXED,
            "tau_bc": Uniform(0.0, 1.0),
        },
        neb={"type": "none"},
        redshift=Fixed(0.5),
    )
    assert model.has_fixedz_photometry_precompute, "model must take the vmap-MCMC path"
    # Two galaxies, same band count → same-shape → vmap eligible.
    rng = np.random.default_rng(0)
    flux = np.abs(rng.normal(1.0, 0.1, size=3))
    batch = [
        {"flux_obs": jnp.asarray(flux), "noise": jnp.asarray(0.1 * flux)},
        {"flux_obs": jnp.asarray(1.2 * flux), "noise": jnp.asarray(0.1 * flux)},
    ]
    f = Fitter(model, batch[0]["flux_obs"], batch[0]["noise"], data_type="photometry")
    return f, batch


def _run(fitter, batch, **overrides):
    kw = {**_MCMC_KW, **overrides}
    return fitter.fit_batch(batch, method="mcmc_hmc", key=jax.random.PRNGKey(0), **kw)


def test_vmap_mcmc_kernel_is_reused_across_calls(fitter):
    """Two identical batch-MCMC calls reuse one compiled kernel (memo hit)."""
    f, batch = fitter
    f.__dict__.pop("_batch_mcmc_kernel_cache", None)

    res1 = _run(f, batch)
    cache_after_1 = dict(getattr(f, "_batch_mcmc_kernel_cache", {}))
    res2 = _run(f, batch)
    cache_after_2 = dict(getattr(f, "_batch_mcmc_kernel_cache", {}))

    assert len(cache_after_1) == 1, "first call must memoize exactly one vmap-MCMC kernel"
    assert cache_after_2 == cache_after_1, "second identical call must REUSE, not rebuild"
    # Results well-formed and deterministic (same key → same samples).
    for res in (res1, res2):
        assert len(res) == 2
        for r in res:
            assert np.all(np.isfinite(np.asarray(r.samples["sfh_dpl_log_total_mass"])))
    s1 = np.asarray(res1[0].samples["sfh_dpl_log_total_mass"])
    s2 = np.asarray(res2[0].samples["sfh_dpl_log_total_mass"])
    assert np.array_equal(s1, s2), "same key must give identical samples (pure refactor)"


def test_different_config_builds_distinct_kernel(fitter):
    """A different structural config (n_samples) does not reuse the kernel."""
    f, batch = fitter
    f.__dict__.pop("_batch_mcmc_kernel_cache", None)

    _run(f, batch, n_samples=12)
    _run(f, batch, n_samples=20)
    cache = getattr(f, "_batch_mcmc_kernel_cache", {})
    assert len(cache) == 2, "distinct n_samples must key distinct kernels, not collide"


@pytest.mark.parametrize("method", ["mcmc_nuts", "mcmc_hmc", "mcmc_dynamic_hmc", "mcmc_ghmc"])
def test_all_vmap_mcmc_methods_run_after_adaptation_threading(fitter, method):
    """Each threaded ``_sample_scan`` branch runs end-to-end with finite samples.

    The adaptation-threading touched all four method branches (the GHMC branch
    also moved its momentum-scale derivation inside ``_sample_scan``); this pins
    that none of them broke mechanically (wrong ``in_axes``, arg order, or GHMC
    momentum shape would crash or produce non-finite samples).
    """
    f, batch = fitter
    f.__dict__.pop("_batch_mcmc_kernel_cache", None)
    res = f.fit_batch(
        batch,
        method=method,
        key=jax.random.PRNGKey(0),
        n_warmup=15,
        n_burnin=5,
        n_samples=12,
        verbose=False,
    )
    assert len(res) == len(batch)
    for r in res:
        s = np.asarray(r.samples["sfh_dpl_log_total_mass"])
        assert s.shape[0] == 12
        assert np.all(np.isfinite(s)), f"{method}: non-finite samples after threading"
