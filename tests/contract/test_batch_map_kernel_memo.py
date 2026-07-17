# SPDX-License-Identifier: BSD-3-Clause
"""Batch-MAP kernel memo: same config reuses the compiled wrapper, None disables it.

Regression guard for the JIT recompile-churn bug: ``_fit_batch_vmap_map`` built a
fresh ``jax.jit(jax.vmap(...))`` closure on every call, so a catalog processed in
repeated same-shape ``fit_batch("map")`` calls missed ``jax.jit``'s cache and
recompiled the (expensive) vmapped loss each time. ``Fitter._memo_batch_map_kernel``
caches the wrapper object under a config key so ``jax.jit`` can reuse the compiled
executable — while ``key=None`` (un-fingerprintable custom optimizer) always builds
fresh so a stale kernel can never be served.

The full vmapped-MAP path needs a fixed-z photometry precompute (real filters), so
this pins the memo *mechanism* directly, which is what the fix turns on.
"""

from __future__ import annotations

import jax.numpy as jnp
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
    obs = Observation(photometry=Photometry(filters=tuple(_tophat(c) for c in (3500.0, 4800.0))))
    model = SEDModel.build(
        ssp_data=synthetic_ssp_wide,
        observation=obs,
        sfh={"type": "dpl", "*": FIXED, "log_total_mass": Uniform(8, 12)},
        dust={"type": "two_component", "law_bc": "calzetti", "*": FIXED},
        neb={"type": "none"},
        redshift=Fixed(0.5),
    )
    data = jnp.ones(2)
    return Fitter(model, data, 0.1 * jnp.ones(2), data_type="photometry")


@pytest.fixture(autouse=True)
def _clear_kernel_cache(fitter):
    """Each test starts with an empty batch-MAP kernel cache (module fixture is shared)."""
    fitter.__dict__.pop("_batch_map_kernel_cache", None)
    yield


def test_same_key_reuses_wrapper_and_builds_once(fitter):
    calls = {"n": 0}

    def builder():
        calls["n"] += 1
        return object()

    key = ("optax", "adam", 0.02)
    a = fitter._memo_batch_map_kernel(key, builder)
    b = fitter._memo_batch_map_kernel(key, builder)
    assert a is b, "same config must reuse the cached kernel object"
    assert calls["n"] == 1, "builder must run only once for a repeated key"


def test_different_key_builds_a_distinct_wrapper(fitter):
    calls = {"n": 0}

    def builder():
        calls["n"] += 1
        return object()

    a = fitter._memo_batch_map_kernel(("optax", "adam", 0.02), builder)
    c = fitter._memo_batch_map_kernel(("optax", "adam", 0.01), builder)
    assert a is not c, "a different optimizer config must not reuse a kernel"
    assert calls["n"] == 2


def test_none_key_disables_the_memo(fitter):
    calls = {"n": 0}

    def builder():
        calls["n"] += 1
        return object()

    d = fitter._memo_batch_map_kernel(None, builder)
    e = fitter._memo_batch_map_kernel(None, builder)
    assert d is not e, "key=None must always build fresh (never serve a cached kernel)"
    assert calls["n"] == 2
