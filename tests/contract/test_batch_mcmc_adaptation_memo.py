# SPDX-License-Identifier: BSD-3-Clause
"""Batch-MCMC window adaptation: compiled ONCE and reused, not re-accreted per call.

``_fit_batch_vmap_mcmc`` memoized the vmapped *sampler* (#1208) but still built
``blackjax.window_adaptation(...)`` eagerly with a *fresh* ``ld_first`` closure on
every ``fit_batch`` call. A fresh closure has a fresh function identity, so JAX's
in-memory compilation cache compiled and *retained* a new warmup executable each
call and never evicted it — a ~20 MB/call leak on a long-lived Fitter (measured;
present since before the memo). The single-galaxy MCMC path never had this because
it wraps warmup+scan in a module-level ``jax.jit`` (``_hmc_full_scan``) with the
data threaded as a traced argument.

The fix mirrors that pattern for the batch path: the warmup is wrapped in a
``jax.jit`` that takes the first galaxy's ``data_args`` as a *traced* argument and
is memoized on a structural key (``_batch_adapt_kernel_cache``). The adaptation
still *runs* every call on the current data (numbers unchanged), but the compiled
warmup is reused instead of re-accreted.

This pins:

* the adaptation kernel is compiled once and reused across same-config calls
  (memo hit — cache stays at one entry, so JAX no longer accretes a warmup per call),
* a structurally different adaptation config keys a distinct kernel,
* samples stay finite and deterministic for a fixed key (the fix moved WHERE the
  warmup is compiled, not the numbers).

Runs on the synthetic wide SSP + tophat filters (no ``data/`` files, #613).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import DEFAULT, Fixed, Observation, Photometry, SEDModel, Uniform
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
        sfh={"type": "dpl", "all_params": Fixed(DEFAULT), "log_total_mass": Uniform(8.0, 12.0)},
        dust_attenuation={
            "type": "two_component",
            "law": "calzetti",
            "all_params": Fixed(DEFAULT),
            "tau_bc": Uniform(0.0, 1.0),
        },
        neb={"type": "none"},
        redshift=Fixed(0.5),
    )
    assert model.has_fixedz_photometry_precompute, "model must take the vmap-MCMC path"
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


def test_adaptation_kernel_compiled_once_and_reused(fitter):
    """Repeated same-config calls reuse one compiled warmup (no per-call accretion)."""
    f, batch = fitter
    f.__dict__.pop("_batch_adapt_kernel_cache", None)

    for _ in range(4):
        _run(f, batch)

    cache = getattr(f, "_batch_adapt_kernel_cache", None)
    assert cache is not None, "adaptation warmup must be memoized on the Fitter"
    assert len(cache) == 1, (
        "four same-config fit_batch calls must reuse ONE compiled warmup — a fresh "
        f"warmup per call is the leak; got {len(cache)} cached kernels"
    )


def test_adaptation_kernel_distinct_per_config(fitter):
    """A different adaptation config (n_warmup) keys a distinct warmup kernel."""
    f, batch = fitter
    f.__dict__.pop("_batch_adapt_kernel_cache", None)

    _run(f, batch, n_warmup=15)
    _run(f, batch, n_warmup=25)
    cache = getattr(f, "_batch_adapt_kernel_cache", {})
    assert len(cache) == 2, "distinct n_warmup must key distinct warmup kernels, not collide"


def test_adaptation_memo_preserves_samples(fitter):
    """Threading the warmup through a reused jit does not change the numbers."""
    f, batch = fitter
    f.__dict__.pop("_batch_adapt_kernel_cache", None)

    res1 = _run(f, batch)
    res2 = _run(f, batch)  # second call hits the memo — must be bit-identical
    for res in (res1, res2):
        for r in res:
            assert np.all(np.isfinite(np.asarray(r.samples["sfh_dpl_log_total_mass"])))
    s1 = np.asarray(res1[0].samples["sfh_dpl_log_total_mass"])
    s2 = np.asarray(res2[0].samples["sfh_dpl_log_total_mass"])
    assert np.array_equal(s1, s2), "reused warmup kernel must give identical samples"
