# SPDX-License-Identifier: BSD-3-Clause
"""The shared compiled-function caches must be keyed on x64 state (#1392).

``get_or_build_cached`` and ``_get_or_build_engine`` key on
``Fitter.compile_signature()``. That signature omitted the ``jax_enable_x64``
state, so a Fitter constructed under ``jax.enable_x64(False)`` was a cache
**hit** for a loss / gradient / engine traced under ``enable_x64(True)`` — the
float32 caller silently ran the float64-traced program.

Scope note, because #1392 bundles two claims. This file pins the *cache* defect
only, which is a correctness hazard on its own terms. It deliberately does not
assert anything about the float32 NaN reported alongside it: on the models
measured while fixing this, float32 gradients are NaN with no float64 anywhere
in the process, and clearing every shared cache does not change that. Asserting
"float32 is finite after the fix" would be asserting something the fix does not
do, and would fail for a reason unrelated to this key (see #1388).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import FIXED, Fixed, SEDModel, Uniform
from tengri.inference.fitter import Fitter

pytestmark = pytest.mark.regression_bug


def _fitter(ssp, obs):
    """Fitter on the synthetic CI-runnable pair — no gitignored SSP grids."""
    model = SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        sfh={"type": "const", "all_params": FIXED, "log_total_mass": Uniform(9.0, 11.0)},
        dust={"type": "two_component", "all_params": FIXED, "tau_bc": Uniform(0.0, 2.0)},
        redshift=Fixed(0.1),
        approx=None,
    )
    n = obs.photometry.n_filters
    return Fitter(model, jnp.ones(n), jnp.ones(n) * 0.1, data_type="photometry")


def test_signature_differs_across_x64_state(synthetic_ssp_wide, synthetic_tophat_obs):
    """The whole point: two precisions must not produce the same cache key."""
    with jax.enable_x64(True):
        sig_64 = _fitter(synthetic_ssp_wide, synthetic_tophat_obs).compile_signature()
    with jax.enable_x64(False):
        sig_32 = _fitter(synthetic_ssp_wide, synthetic_tophat_obs).compile_signature()

    assert sig_64 != sig_32, (
        "compile_signature() is identical under enable_x64(True) and (False); "
        "a float32 Fitter will reuse a float64-traced engine (#1392)"
    )


def test_x64_flag_is_the_only_difference(synthetic_ssp_wide, synthetic_tophat_obs):
    """Guard the fix's *shape*, not just its effect.

    If some unrelated field started varying with precision the test above would
    pass for the wrong reason. Pinning the differing position to the last entry
    of the engine-cache key keeps the diagnosis honest.
    """
    with jax.enable_x64(True):
        key_64 = _fitter(synthetic_ssp_wide, synthetic_tophat_obs)._engine_cache_key()
    with jax.enable_x64(False):
        key_32 = _fitter(synthetic_ssp_wide, synthetic_tophat_obs)._engine_cache_key()

    assert len(key_64) == len(key_32)
    differing = [i for i, (a, b) in enumerate(zip(key_64, key_32, strict=True)) if a != b]
    assert differing == [len(key_64) - 1], (
        f"expected only the trailing x64 entry to differ, got positions {differing}"
    )
    assert (key_64[-1], key_32[-1]) == (True, False)


def test_signature_is_recomputed_not_frozen_at_construction(
    synthetic_ssp_wide, synthetic_tophat_obs
):
    """The x64 entry has to track the *current* state.

    ``compile_signature`` is recomputed per call (its docstring claimed
    otherwise for a long time). If it were memoized at construction, a Fitter
    built under one precision would keep reporting that precision forever and
    the key would be wrong exactly when it matters.
    """
    with jax.enable_x64(True):
        fitter = _fitter(synthetic_ssp_wide, synthetic_tophat_obs)
        inside = fitter._engine_cache_key()[-1]
    outside = fitter._engine_cache_key()[-1]

    assert inside is True
    assert outside is not None
    assert outside == bool(jax.config.jax_enable_x64), (
        "the x64 entry is stale — compile_signature() appears to be memoized"
    )


def test_float32_fitter_does_not_reuse_the_float64_loss(synthetic_ssp_wide, synthetic_tophat_obs):
    """End-to-end: the shared loss cache must gain an entry for the second precision.

    This is the measurement from the issue, inverted. The probe asserts its own
    setup first — a cache that never populated would make "no new keys" pass
    vacuously, which is how an earlier version of this check fooled me.
    """
    import tengri.inference.jit_engine as jit_engine
    from tengri.inference.context import InferenceContext

    def touch(x64: bool, dtype) -> None:
        with jax.enable_x64(x64):
            ctx = InferenceContext.from_target(_fitter(synthetic_ssp_wide, synthetic_tophat_obs))
            data_args = ctx.data_args
            names = sorted(ctx.initial_params(jax.random.PRNGKey(0)))
            point = {k: jnp.asarray(0.0, dtype=dtype) for k in names}
            jax.grad(lambda q: ctx.neg_log_posterior_fn(q, data_args))(point)

    for cache, _lock in jit_engine._SHARED_CACHES.values():
        cache.clear()

    touch(True, jnp.float64)
    populated = {
        name: set(cache.keys())
        for name, (cache, _lock) in jit_engine._SHARED_CACHES.items()
        if cache
    }
    assert populated, (
        "SETUP FAILURE: no shared cache populated after a float64 gradient, so "
        "this test cannot detect reuse. Fix the probe, not the assertion."
    )

    touch(False, jnp.float32)

    for name, before in populated.items():
        cache, _lock = jit_engine._SHARED_CACHES[name]
        new_keys = set(cache.keys()) - before
        assert new_keys, (
            f"the float32 Fitter added no new {name!r} cache entry — it reused the "
            f"float64-traced one (#1392)"
        )


def test_same_precision_still_shares(synthetic_ssp_wide, synthetic_tophat_obs):
    """The fix must not defeat the reuse the cache exists for.

    Two Fitters at the *same* precision must still collide, otherwise every
    catalog fit recompiles per galaxy.
    """
    with jax.enable_x64(True):
        a = _fitter(synthetic_ssp_wide, synthetic_tophat_obs).compile_signature()
        b = _fitter(synthetic_ssp_wide, synthetic_tophat_obs).compile_signature()
    assert a == b, "two Fitters at the same precision no longer share a cache key"


def test_x64_entry_is_hashable(synthetic_ssp_wide, synthetic_tophat_obs):
    """The signature keys an OrderedDict, so it must stay hashable."""
    with jax.enable_x64(True):
        sig = _fitter(synthetic_ssp_wide, synthetic_tophat_obs).compile_signature()
    assert isinstance(hash(sig), int)
    assert isinstance(np.asarray(sig[1][-1]).item(), bool)
