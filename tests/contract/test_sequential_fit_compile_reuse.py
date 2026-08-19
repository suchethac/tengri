# SPDX-License-Identifier: BSD-3-Clause
"""Fitting galaxy N+1 on the same model must reuse galaxy N's compiled program.

A catalog is often fit as a Python loop of independent ``forward.fit(...)`` calls.
Every ``Fitter`` resolves ``approx`` and **clones** the model, and the compile
caches (``_model_cache``, and the flat log-density built on it) key on model
**identity** -- so each galaxy got a fresh clone, missed every cache, and
recompiled the forward-heavy kernels, even though the fitters'
``_engine_cache_key()`` values were already identical.

Measured in a clean process on a D=23 field-SFH model (photometry + 8 line
fluxes), counting XLA compilations per sequential fit:

===============================================  =====  =====  =====
state                                            fit 0  fit 1  fit 2
===============================================  =====  =====  =====
before                                              17      2      2
after memoizing the approx clone                    17      2      2
after also memoizing the prewarm predict jits       17      0      0
===============================================  =====  =====  =====

The clone memo alone fixed the *expensive* kernels (``val_and_grad``,
``single_step``, ``scan_batch``) and left two per-galaxy compiles:
``_auto_prewarm`` called ``jax.jit(self.model.predict_photometry)``, and
``model.predict_photometry`` builds a **new bound method on every attribute
access**, so ``jax.jit`` saw a different callable each fit. The step whose whole
job is warming was the one thing that never stayed warm.

**Why this file asserts the mechanism rather than counting compiles.** A compile
count is only meaningful from a cold process. Inside a shared pytest session --
and with tengri's persistent on-disk compile cache enabled -- XLA's executable
cache serves the work even when the memos are cleared, so a
``log_compiles``-based neuter cannot fire and a zero would prove nothing. What is
deterministic is the mechanism: both memos must hand back the *same objects*
across fits. Remove either memo and these fail immediately.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from tengri import FIXED, Fixed, ForwardModel, Observation, Photometry, SEDModel, Uniform
from tengri.inference import fitter as fitter_mod
from tengri.inference.fitter import Fitter
from tengri.observation.photometry import FilterCurve

pytestmark = pytest.mark.contract


def _tophat(center, frac=0.16, n=32):
    wave = jnp.linspace(center * (1.0 - frac), center * (1.0 + frac), n)
    trans = jnp.sin(jnp.linspace(0.0, jnp.pi, n)) * 0.6
    return FilterCurve(wave=wave, trans=trans, name=f"b{int(center)}")


_PHOT = Photometry(filters=tuple(_tophat(c) for c in (3500.0, 4800.0, 6200.0)))


def _model(ssp, obs):
    return SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        sfh={"type": "dpl", "*": FIXED, "log_total_mass": Uniform(8, 12)},
        dust={"type": "two_component", "law": "calzetti", "*": FIXED},
        neb={"type": "none"},
        redshift=Fixed(0.5),
    )


def test_every_galaxy_resolves_to_the_same_fit_model(synthetic_ssp_wide):
    """N Fitters over one ForwardModel must share ONE resolved clone.

    This is the whole mechanism: the compile caches are identity-keyed, so a
    distinct clone per galaxy is a guaranteed cache miss per galaxy.
    """
    obs = Observation(photometry=_PHOT)
    forward = ForwardModel.build(sed=_model(synthetic_ssp_wide, obs), observation=obs)
    data = jnp.ones(len(_PHOT.filters))
    noise = 0.1 * jnp.ones_like(data)

    fitters = [
        Fitter(forward, data * (1.0 + 0.17 * i), noise, data_type="photometry") for i in range(3)
    ]
    models = {id(f.model) for f in fitters}
    assert len(models) == 1, (
        f"{len(models)} distinct fit models for 3 galaxies on one ForwardModel — every "
        f"identity-keyed compile cache misses once per galaxy"
    )


def test_the_prewarm_predict_wrappers_are_memoized(synthetic_ssp_wide):
    """``jax.jit(model.predict_*)`` must not be rebuilt per fit.

    ``model.predict_photometry`` is a fresh bound method on every attribute access,
    so wrapping it inline gives ``jax.jit`` a new callable each time and recompiles.
    """
    obs = Observation(photometry=_PHOT)
    forward = ForwardModel.build(sed=_model(synthetic_ssp_wide, obs), observation=obs)
    fitter = Fitter(forward, jnp.ones(len(_PHOT.filters)), 0.1 * jnp.ones(len(_PHOT.filters)))

    first = fitter_mod._memoized_predict_jit(fitter.model, "predict_photometry")
    second = fitter_mod._memoized_predict_jit(fitter.model, "predict_photometry")
    assert first is second, "the memo handed back a different jit wrapper on the second call"

    # A bare re-wrap is what the old code did — different object, hence a recompile.
    assert first is not jax.jit(fitter.model.predict_photometry), (
        "jax.jit(model.predict_photometry) returned the memoized object, so this test "
        "can no longer distinguish the bug from the fix"
    )

    other = fitter_mod._memoized_predict_jit(fitter.model, "predict_properties")
    assert other is not first, "two different accessors must not share one memo slot"


def test_the_approx_clone_memo_keys_on_the_resolved_config(synthetic_ssp_wide):
    """Different resolved configs must not collide onto one clone.

    Keyed on the resolved config rather than the caller's ``approx`` argument,
    because resolution depends on fitter state (whether a line channel is fit), so
    ``approx="auto"`` can legitimately resolve differently for two fits.
    """
    from tengri import WavePrecomp

    obs = Observation(photometry=_PHOT)
    forward = ForwardModel.build(sed=_model(synthetic_ssp_wide, obs), observation=obs)

    # Two genuinely different configs of the same type -- FeaturePrecomp is not
    # usable here, it requires an emission-line channel to tabulate.
    a = fitter_mod._memoized_approx_clone(forward, WavePrecomp(n_z=64))
    b = fitter_mod._memoized_approx_clone(forward, WavePrecomp(n_z=64))
    c = fitter_mod._memoized_approx_clone(forward, WavePrecomp(n_z=128))

    assert a is b, "same config must return the same clone, or the caches keep missing"
    assert c is not a, "distinct configs must not collide onto one clone"
