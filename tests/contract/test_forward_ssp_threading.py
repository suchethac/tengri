# SPDX-License-Identifier: BSD-3-Clause
"""The public forward surfaces must let a caller THREAD the SSP grid, not only bake it.

Sibling of :mod:`tests.contract.test_loss_ssp_threading`, which pins the same
contract for the *loss* path. That path is fixed and stays fixed; this file covers
the other half.

**The gap.** ``SEDModel.predict_observables_jit`` already threads correctly — it
passes ``self.ssp_data`` as an *argument* to an inner, structurally-cached
``jax.jit``, so the grid is a ``Parameter`` op in tengri's own compiled program.
But the moment a caller wraps the bound method in their **own** JAX transform::

    predict = jax.jit(model.predict_photometry)  # notebooks teach this

the inner jit inlines into the outer trace and ``self.ssp_data`` — read off
``self`` as a concrete array — becomes a ``Constant`` of the outer computation.
Threading is defeated from outside, and nothing in the library can see it happen.

Measured on a real SSP ``(15, 93, 5994) float64`` = 66.89 MB, photometry model:

===============  ======================  ==============  =================
``approx``       route                   cold compile    cache entry
===============  ======================  ==============  =================
``WavePrecomp``  internal                0.35 s          1.65 MB
``WavePrecomp``  user ``jax.jit``        0.36 s          0.86 MB
exact            internal                0.53 s          0.23 MB
exact            user ``jax.jit``        0.74 s          **58.82 MB**
===============  ======================  ==============  =================

On the ``WavePrecomp`` path the cube is dead code and XLA eliminates it, so only
the exact path pays — and there it pays 256x on the persistent cache entry, which
is the mechanism behind the 141 GB cache in #1507.

**The contract pinned here.** ``predict_state`` has accepted
``ssp_data=``/``template_data=`` for some time; the surfaces users are actually
told to JIT — ``predict_photometry`` ("the inference hot path"),
``predict_properties`` ("the ONE jit/vmap surface for derived quantities") and
``predict_spectrum`` — did not. They must, on both ``SEDModel`` and
``ForwardModel``, and threading must not move the answer.

Runs on the synthetic wide SSP (no ``data/ssp_*.h5`` needed, #613).
"""

from __future__ import annotations

import dataclasses
import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import FIXED, Fixed, ForwardModel, Observation, SEDModel, Uniform
from tengri.observation.spectroscopy import Spectroscopy

pytestmark = pytest.mark.contract

_SPEC_WAVE = jnp.linspace(3500.0, 9000.0, 120)


def _largest_baked_const_at_any_depth(fn, *call_args):
    """Largest array frozen as a jaxpr const at ANY nesting depth.

    Same walker as the sibling loss-threading suite, and for the same reason: a
    ``ClosedJaxpr`` holds const VALUES on ``.consts`` while ``.jaxpr`` carries only
    ``constvars``, so a top-level ``.consts`` read reports a clean bill of health on
    a graph whose inner ``jax.jit`` has captured the whole grid.
    """
    seen: set[int] = set()
    best = 0

    def walk(node):
        nonlocal best
        if id(node) in seen:
            return
        seen.add(id(node))
        if hasattr(node, "consts") and hasattr(node, "jaxpr"):  # ClosedJaxpr
            for c in node.consts:
                best = max(best, int(getattr(c, "size", 0)))
            walk(node.jaxpr)
            return
        for eqn in getattr(node, "eqns", []) or []:
            for value in eqn.params.values():
                for cand in value if isinstance(value, (tuple, list)) else [value]:
                    if hasattr(cand, "eqns") or (
                        hasattr(cand, "consts") and hasattr(cand, "jaxpr")
                    ):
                        walk(cand)

    walk(jax.make_jaxpr(fn)(*call_args))
    return best


def _build(ssp, observation):
    """Stellar + dust, nebular off — so the SSP flux grid is the only large array."""
    return SEDModel.build(
        ssp_data=ssp,
        observation=observation,
        sfh={"type": "dpl", "all_params": FIXED, "log_total_mass": Uniform(8, 12)},
        dust={"type": "two_component", "law_bc": "calzetti", "all_params": FIXED},
        neb={"type": "none"},
        redshift=Fixed(0.5),
    )


def _models(ssp, synthetic_tophat_obs, with_spectrum=False):
    """Yield the two public surfaces over one physics config."""
    obs = synthetic_tophat_obs
    if with_spectrum:
        obs = Observation(
            photometry=obs.photometry, spectroscopy=Spectroscopy(wave_obs=_SPEC_WAVE)
        )
    sed = _build(ssp, obs)
    return {"sed_model": sed, "forward_model": ForwardModel.build(sed=sed, observation=obs)}


# --------------------------------------------------------------------------
# 1. The signature contract — the channel must be reachable at all.
# --------------------------------------------------------------------------

_JIT_SAFE_SURFACES = ["predict_photometry", "predict_properties", "predict_spectrum"]


@pytest.mark.parametrize("surface", ["sed_model", "forward_model"])
@pytest.mark.parametrize("method", _JIT_SAFE_SURFACES)
def test_jit_safe_surfaces_accept_threaded_data(
    synthetic_ssp_wide, synthetic_tophat_obs, surface, method
):
    """Every documented JIT/vmap-safe surface accepts ``ssp_data``/``template_data``.

    Asserted on the signature rather than through a trace: a missing keyword is
    unambiguous, and it is the whole defect — ``predict_state`` has had this channel
    all along while the surfaces users are told to JIT never got it.

    A ``**kwargs`` passthrough counts. ``ForwardModel`` routes several methods
    through ``_DELEGATED_TO_INNER_SED`` rather than restating each signature, so
    ``predict_properties`` really is ``(*args, **kwargs)`` — it genuinely accepts
    the keyword and forwards it. Demanding a named parameter there would be a test
    demanding a wrapper the delegation table exists to avoid. The behavioral tests
    below are what actually prove the keyword arrives and does something.
    """
    model = _models(synthetic_ssp_wide, synthetic_tophat_obs, with_spectrum=True)[surface]
    sig = inspect.signature(getattr(model, method))
    accepts_any_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    for kw in ("ssp_data", "template_data"):
        assert kw in sig.parameters or accepts_any_kw, (
            f"[{surface}.{method}] has no '{kw}' parameter and no **kwargs passthrough, so "
            f"a caller who wraps it in their own jax.jit/vmap/grad cannot avoid baking the "
            f"SSP grid as a constant. predict_state accepts it; these surfaces must too. "
            f"Signature: {sig}"
        )


# --------------------------------------------------------------------------
# 2. The behavioral contract — threading it actually keeps it out of the graph.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("surface", ["sed_model", "forward_model"])
def test_threaded_photometry_is_not_baked_under_a_user_jit(
    synthetic_ssp_wide, synthetic_tophat_obs, surface
):
    """With the grid passed in, it must be an invar of the user's trace, not a const."""
    ssp = synthetic_ssp_wide
    model = _models(ssp, synthetic_tophat_obs)[surface]
    ssp_size = int(np.asarray(ssp.ssp_flux).size)
    p = model.spec.sample(jax.random.PRNGKey(0)) if hasattr(model, "spec") else None

    def threaded(grid, params):
        return model.predict_photometry(params, ssp_data=grid)

    biggest = _largest_baked_const_at_any_depth(threaded, ssp, p)
    assert biggest < ssp_size, (
        f"[{surface}] the caller threaded the SSP grid and a constant of size {biggest} "
        f">= grid size {ssp_size} is still frozen into the trace. On a real SSP that is "
        f"66.89 MB inlined per compiled program and a 256x persistent-cache entry (#1507)."
    )


@pytest.mark.parametrize("surface", ["sed_model", "forward_model"])
def test_unthreaded_photometry_still_bakes_the_grid(
    synthetic_ssp_wide, synthetic_tophat_obs, surface
):
    """Neuter for the test above: without threading the grid IS baked.

    If this ever stops holding, the test above proves nothing — it would be passing
    because the grid stopped being large, not because threading works.
    """
    ssp = synthetic_ssp_wide
    model = _models(ssp, synthetic_tophat_obs)[surface]
    ssp_size = int(np.asarray(ssp.ssp_flux).size)
    p = model.spec.sample(jax.random.PRNGKey(0))

    biggest = _largest_baked_const_at_any_depth(model.predict_photometry, p)
    assert biggest >= ssp_size, (
        f"[{surface}] expected the un-threaded call to bake the grid (size {ssp_size}) "
        f"so the threaded test has something to prove; largest const was {biggest}"
    )


# --------------------------------------------------------------------------
# 3. Threading moves WHERE the grid enters, never the answer.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("surface", ["sed_model", "forward_model"])
def test_threading_does_not_change_photometry(synthetic_ssp_wide, synthetic_tophat_obs, surface):
    """Same params, threaded vs closure-captured — the flux must agree."""
    ssp = synthetic_ssp_wide
    model = _models(ssp, synthetic_tophat_obs)[surface]
    p = model.spec.sample(jax.random.PRNGKey(1))

    baked = np.asarray(model.predict_photometry(p), dtype=np.float64)
    threaded = np.asarray(model.predict_photometry(p, ssp_data=ssp), dtype=np.float64)

    assert np.all(np.isfinite(threaded)), f"[{surface}] threaded photometry is not finite"
    np.testing.assert_allclose(
        threaded,
        baked,
        rtol=1e-12,
        atol=0.0,
        err_msg=f"[{surface}] threading changed the physics, not just the calling convention",
    )


@pytest.mark.parametrize("surface", ["sed_model", "forward_model"])
def test_threading_does_not_change_properties(synthetic_ssp_wide, synthetic_tophat_obs, surface):
    """Same contract on the derived-quantity surface."""
    ssp = synthetic_ssp_wide
    model = _models(ssp, synthetic_tophat_obs)[surface]
    p = model.spec.sample(jax.random.PRNGKey(2))
    names = ("stellar_mass", "sfr_100myr")

    baked = model.predict_properties(p, names=names)
    threaded = model.predict_properties(p, names=names, ssp_data=ssp)

    for k in names:
        np.testing.assert_allclose(
            np.asarray(threaded[k], dtype=np.float64),
            np.asarray(baked[k], dtype=np.float64),
            rtol=1e-12,
            atol=0.0,
            err_msg=f"[{surface}] threading changed derived quantity {k!r}",
        )


# --------------------------------------------------------------------------
# 4. The topology gate is the shared predicate, not a second copy of it.
# --------------------------------------------------------------------------


def _phot_const(model, ssp, key):
    """Largest const baked into a user-jitted, *threaded* photometry call."""
    params = model.spec.sample(jax.random.PRNGKey(key))

    def threaded(grid, p):
        return model.predict_photometry(p, ssp_data=grid)

    return _largest_baked_const_at_any_depth(threaded, ssp, params)


def test_forward_threading_defers_to_supports_jit_threading(
    synthetic_ssp_wide, synthetic_tophat_obs, monkeypatch
):
    """``ForwardModel`` must gate threading on ``_supports_jit_threading()``.

    The threaded forward is valid only for a single, non-spatial,
    non-``PopulationSEDModel`` population, and that condition already exists as
    :meth:`ForwardModel._supports_jit_threading`. Re-deriving it at the call
    site is what this pins against: the first cut of this fix restated the
    condition inline and dropped the ``spatial is not None`` clause, so a
    spatial population would have been threaded even though the canonical
    predicate refuses it — and the loss path's own docstring records that exact
    bug arriving the exact same way, from a condition stated in two places.

    Forcing the predicate false must make the model decline to thread: the grid
    goes back to being closure-captured rather than routed through ``run``.

    **Each leg builds its own model, and the patch is installed before the
    build.** The forward memoizes its traced graph, so a model that has already
    been traced serves the cached jaxpr and never re-executes the Python body —
    patching afterwards is invisible, and the measurement silently reports the
    pre-patch behavior. An earlier draft of this test did exactly that and read
    "still threaded" off its own warm cache.

    ``ForwardModel`` is a frozen dataclass, so the patch goes on the class;
    instance assignment raises ``FrozenInstanceError``.
    """
    ssp = synthetic_ssp_wide
    ssp_size = int(np.asarray(ssp.ssp_flux).size)

    monkeypatch.setattr(ForwardModel, "_supports_jit_threading", lambda self: False)
    refused = _models(ssp, synthetic_tophat_obs)["forward_model"]
    biggest = _phot_const(refused, ssp, key=3)

    monkeypatch.undo()
    allowed = _models(ssp, synthetic_tophat_obs)["forward_model"]
    assert allowed._supports_jit_threading(), "fixture should be a threadable topology"
    control = _phot_const(allowed, ssp, key=3)

    assert control < ssp_size, (
        f"control leg failed: threading did not work on an unpatched model "
        f"(const {control} >= grid size {ssp_size}), so the forced-false leg "
        f"cannot show that the gate is what disabled it"
    )
    assert biggest >= ssp_size, (
        "with _supports_jit_threading() forced false the SSP grid must NOT be "
        f"threaded into run(), but the largest baked const was {biggest} < grid size "
        f"{ssp_size}, so the gate consulted something other than the shared predicate. "
        "Call _supports_jit_threading(); do not restate its condition."
    )


def test_threading_gate_rejects_spatial_populations(synthetic_ssp_wide, synthetic_tophat_obs):
    """The clause the inline copy dropped, pinned directly on the predicate.

    ``_supports_jit_threading`` excludes a population carrying a spatial
    submodel. The sibling loss-path suite pins the hierarchical clause and the
    positive case but not this one, and the forward gate now depends on it, so
    state it rather than leaving it as the untested half of the condition.

    Both ``ForwardModel`` and ``Population`` are frozen dataclasses, so this
    builds the spatial variant with :func:`dataclasses.replace` instead of
    mutating — the spatial submodel is never run, only inspected by the gate.
    """
    model = _models(synthetic_ssp_wide, synthetic_tophat_obs)["forward_model"]
    assert model._supports_jit_threading(), "fixture should be threadable to start"

    spatial_pop = dataclasses.replace(model.populations[0], spatial=object())
    with_spatial = dataclasses.replace(model, populations=(spatial_pop,))

    assert not with_spatial._supports_jit_threading(), (
        "a spatial population must not be threaded — the threaded forward "
        "mis-broadcasts the galaxy axis against the SFH grid"
    )
