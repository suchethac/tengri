# SPDX-License-Identifier: BSD-3-Clause
"""MAP multistart: the vmapped restart kernel is reused, not rebuilt per call.

Third and last of the batch-kernel memo cohort (after the batch-MAP memo #1201 and
the batch-MCMC memo #1208). ``_run_map_multistart`` already threads the restart
inits AND ``data_args`` as runtime arguments (never closure-captured), but it built
a *fresh* ``jax.jit(jax.vmap(_optimize_one))`` on every call — and a fresh function
object misses ``jax.jit``'s compilation cache. So a catalog fit through the
sequential (non-vmap) path with ``n_restarts > 1`` recompiled the restart kernel for
every galaxy.

Unlike the MCMC sibling this needs no threading refactor (there is no adapted
step-size/mass-matrix to bake — it is plain Adam), so the fix is a pure memo keyed
on the optimizer config. This test pins:

* two same-config ``run("map", n_restarts>1)`` calls reuse one compiled kernel,
* a different config (``n_steps``) keys a distinct kernel,
* results stay finite (the memo never serves a wrong kernel).

Runs on the synthetic wide SSP + tophat filters (no ``data/`` files needed, #613).
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
            "law_diff": "calzetti",
            "type": "two_component",
            "law_bc": "calzetti",
            "all_params": FIXED,
            "tau_bc": Uniform(0.0, 1.0),
        },
        neb={"type": "none"},
        redshift=Fixed(0.5),
    )
    rng = np.random.default_rng(0)
    flux = jnp.asarray(np.abs(rng.normal(1.0, 0.1, size=3)))
    return Fitter(model, flux, 0.1 * flux, data_type="photometry")


def _run(f, **overrides):
    kw = dict(n_restarts=3, n_steps=40, verbose=False)
    kw.update(overrides)
    return f.run("map", key=jax.random.PRNGKey(0), **kw)


def test_multistart_kernel_is_reused_across_calls(fitter):
    """Two identical multistart MAP calls reuse one compiled restart kernel."""
    fitter.__dict__.pop("_map_multistart_kernel_cache", None)

    r1 = _run(fitter)
    cache_after_1 = dict(getattr(fitter, "_map_multistart_kernel_cache", {}))
    r2 = _run(fitter)
    cache_after_2 = dict(getattr(fitter, "_map_multistart_kernel_cache", {}))

    assert len(cache_after_1) == 1, "first call must memoize exactly one restart kernel"
    assert cache_after_2 == cache_after_1, "second identical call must REUSE, not rebuild"
    for r in (r1, r2):
        assert np.isfinite(float(r.params["sfh_dpl_log_total_mass"]))


def test_different_config_builds_distinct_kernel(fitter):
    """A different structural config (n_steps) keys a distinct kernel."""
    fitter.__dict__.pop("_map_multistart_kernel_cache", None)

    _run(fitter, n_steps=40)
    _run(fitter, n_steps=60)
    cache = getattr(fitter, "_map_multistart_kernel_cache", {})
    assert len(cache) == 2, "distinct n_steps must key distinct kernels, not collide"
