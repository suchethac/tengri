# SPDX-License-Identifier: BSD-3-Clause
"""The loss function must THREAD the SSP grid, never BAKE it as an XLA constant.

Regression guard for the JIT data-threading bug: the loss-function builder had a
fast path that threaded ``ssp_data`` / ``template_data`` as arguments (so they
appear as XLA ``Parameter`` ops) only for **pure photometry** fits. Every other
configuration — spectroscopy, joint, and feature channels (line fluxes / ratios /
indices) — fell through to ``model.predict_spectrum(params)`` /
``model.predict_state(params)``, which closure-capture ``self.ssp_data``. Inlined
into the outer HMC/NUTS/VI/MAP ``jax.jit`` trace, that concrete grid becomes a
``Constant`` op (the SSP flux grid is ~8M floats on a real SSP) — ballooning cold
compile from ~5 s to ~40 s.

The contract these tests pin: for a stellar+dust model (no nebular, so the SSP
flux grid is the only large array), the SSP flux grid must NOT appear among the
constants of the loss-function jaxpr on ANY data channel. It rides in as an
invar via ``data_args["_jit_inputs"]`` instead.

Runs on the synthetic wide SSP (no ``data/ssp_*.h5`` needed, #613).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import FIXED, Fixed, Observation, Photometry, SEDModel, Uniform
from tengri.inference.fitter import Fitter
from tengri.inference.loss_functions import build_loss_fn
from tengri.observation.photometry import FilterCurve
from tengri.observation.spectroscopy import Spectroscopy

pytestmark = pytest.mark.contract


def _tophat(center, frac=0.16, n=40):
    wave = jnp.linspace(center * (1.0 - frac), center * (1.0 + frac), n)
    trans = jnp.sin(jnp.linspace(0.0, jnp.pi, n)) * 0.6
    return FilterCurve(wave=wave, trans=trans, name=f"b{int(center)}")


_PHOT = Photometry(filters=tuple(_tophat(c) for c in (3500.0, 4800.0, 6200.0)))
_SPEC_WAVE = jnp.linspace(3500.0, 9000.0, 200)


def _build_model(ssp, observation):
    return SEDModel.build(
        ssp_data=ssp,
        observation=observation,
        sfh={"type": "dpl", "*": FIXED, "log_total_mass": Uniform(8, 12)},
        dust={"law_diff": 'calzetti', "type": "two_component", "law_bc": "calzetti", "*": FIXED},
        neb={"type": "none"},
        redshift=Fixed(0.5),
    )


def _largest_baked_const(loss_fn, init, data_args):
    """Return the size of the largest array baked as a jaxpr constant."""
    jaxpr = jax.make_jaxpr(loss_fn)(init, data_args)
    sizes = [int(getattr(c, "size", 0)) for c in jaxpr.consts]
    return max(sizes, default=0)


@pytest.mark.parametrize(
    "channel",
    ["photometry", "spectroscopy", "joint"],
)
def test_ssp_grid_is_threaded_not_baked(synthetic_ssp_wide, channel):
    """On every data channel, the SSP flux grid is an invar, not a baked constant."""
    ssp = synthetic_ssp_wide
    ssp_size = int(np.asarray(ssp.ssp_flux).size)

    if channel == "photometry":
        obs = Observation(photometry=_PHOT)
        data = jnp.ones(len(_PHOT.filters))
    elif channel == "spectroscopy":
        obs = Observation(spectroscopy=Spectroscopy(wave_obs=_SPEC_WAVE))
        data = jnp.ones(_SPEC_WAVE.shape[0])
    else:  # joint
        obs = Observation(photometry=_PHOT, spectroscopy=Spectroscopy(wave_obs=_SPEC_WAVE))
        data = jnp.ones(len(_PHOT.filters) + _SPEC_WAVE.shape[0])

    model = _build_model(ssp, obs)
    noise = 0.1 * jnp.ones_like(data)
    fitter = Fitter(model, data, noise, data_type=channel)

    loss_fn = build_loss_fn(fitter)
    init = fitter._initialize_unbounded(jax.random.PRNGKey(0))

    biggest = _largest_baked_const(loss_fn, init, fitter._data_args)
    assert biggest < ssp_size, (
        f"[{channel}] a constant of size {biggest} >= SSP grid size {ssp_size} is baked "
        f"into the loss jaxpr — the SSP grid must be THREADED via data_args, not "
        f"closure-captured. This balloons cold compile."
    )


@pytest.mark.parametrize("channel", ["photometry", "spectroscopy", "joint"])
def test_threaded_and_baked_loss_agree_bit_for_bit(synthetic_ssp_wide, channel):
    """Threading changes WHERE the SSP grid enters the trace, never the physics.

    Evaluate the exact same loss with ``_jit_inputs`` present (threaded path) and
    with it stripped (the eager ``model.predict_*`` baked path). The two must be
    bit-identical — a guard against the threaded route silently computing a
    different number (the classic silent-no-op / wrong-array failure mode).
    """
    ssp = synthetic_ssp_wide
    if channel == "photometry":
        obs = Observation(photometry=_PHOT)
        data = jnp.ones(len(_PHOT.filters))
    elif channel == "spectroscopy":
        obs = Observation(spectroscopy=Spectroscopy(wave_obs=_SPEC_WAVE))
        data = jnp.ones(_SPEC_WAVE.shape[0])
    else:
        obs = Observation(photometry=_PHOT, spectroscopy=Spectroscopy(wave_obs=_SPEC_WAVE))
        data = jnp.ones(len(_PHOT.filters) + _SPEC_WAVE.shape[0])

    model = _build_model(ssp, obs)
    noise = 0.1 * jnp.ones_like(data)
    fitter = Fitter(model, data, noise, data_type=channel)

    loss_fn = build_loss_fn(fitter)
    init = fitter._initialize_unbounded(jax.random.PRNGKey(1))

    threaded_args = fitter._data_args
    baked_args = {k: v for k, v in threaded_args.items() if k != "_jit_inputs"}

    v_threaded = float(loss_fn(init, threaded_args))
    v_baked = float(loss_fn(init, baked_args))

    assert np.isfinite(v_threaded)
    assert v_threaded == pytest.approx(v_baked, rel=1e-12, abs=1e-9), (
        f"[{channel}] threaded loss {v_threaded} != baked loss {v_baked} — "
        f"threading changed the physics"
    )


# ---------------------------------------------------------------------------
# The canonical surface, and a detector that can actually see the failure.
#
# The guards above passed for two years while a real SSP grid was inlined into
# every fit, for two independent reasons:
#
# 1. They only ever build ``Fitter(SEDModel, ...)`` -- the surface NAMING_CONTRACT
#    marks deprecated. The canonical path is ``ForwardModel``, and
#    ``Fitter._build_data_args`` reads ``model.ssp_data``. ``ForwardModel`` did not
#    delegate that attribute, so the read raised ``AttributeError`` inside a
#    ``contextlib.suppress(AttributeError, TypeError)`` and the whole
#    ``args["_jit_inputs"] = {...}`` assignment was skipped -- silently, with no
#    threading, on the surface every user is told to use.
# 2. ``_largest_baked_const`` inspects ``jax.make_jaxpr(...).consts``, which is
#    TOP-LEVEL only. The grid is captured by an INNER ``jax.jit``, whose consts live
#    on its own sub-jaxpr. Measured on a real SSP: top-level consts reported
#    ``n=0, 0.00 MB`` while the lowered HLO carried the grid twice at 133.8 MB each
#    (267.6 MB of a 274.6 MB program, 99.8% constants).
#
# So the fix needs both a threading repair and a detector that recurses.
# ---------------------------------------------------------------------------


def _largest_baked_const_at_any_depth(loss_fn, init, data_args):
    """Largest array frozen as a jaxpr const at ANY nesting depth.

    A ``ClosedJaxpr`` holds const VALUES in ``.consts`` while ``.jaxpr`` is the open
    jaxpr carrying only ``constvars``. Unwrapping to ``.jaxpr`` before reading
    ``.consts`` discards exactly what is being looked for and reports a clean bill
    of health on a graph that is 99.8% baked data.
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

    walk(jax.make_jaxpr(loss_fn)(init, data_args))
    return best


def _obs_and_data(channel):
    if channel == "photometry":
        return Observation(photometry=_PHOT), jnp.ones(len(_PHOT.filters))
    if channel == "spectroscopy":
        obs = Observation(spectroscopy=Spectroscopy(wave_obs=_SPEC_WAVE))
        return obs, jnp.ones(_SPEC_WAVE.shape[0])
    obs = Observation(photometry=_PHOT, spectroscopy=Spectroscopy(wave_obs=_SPEC_WAVE))
    return obs, jnp.ones(len(_PHOT.filters) + _SPEC_WAVE.shape[0])


def _fitter_on_surface(surface, ssp, channel="photometry"):
    """One fitter per inference surface, built the way that surface really is."""
    obs, data = _obs_and_data(channel)
    model = _build_model(ssp, obs)
    noise = 0.1 * jnp.ones_like(data)
    if surface == "sed_model":
        return Fitter(model, data, noise, data_type=channel)
    from tengri import ForwardModel

    forward = ForwardModel.build(sed=model, observation=obs)
    # Exactly what ForwardModel.fit does internally.
    return Fitter(forward, data, noise, data_type=channel)


@pytest.mark.parametrize("channel", ["photometry", "spectroscopy", "joint"])
@pytest.mark.parametrize("surface", ["sed_model", "forward_model"])
def test_jit_inputs_are_populated_on_every_inference_surface(synthetic_ssp_wide, surface, channel):
    """``_jit_inputs`` must carry the SSP grid on every surface AND every channel.

    Asserted directly on ``_data_args`` rather than through a jaxpr: this is the
    step that silently did nothing, and a missing dict key is unambiguous.
    """
    ssp = synthetic_ssp_wide
    fitter = _fitter_on_surface(surface, ssp, channel)

    args = fitter._data_args
    assert "_jit_inputs" in args, (
        f"[{surface}/{channel}] _data_args has no '_jit_inputs' key — JIT threading was skipped "
        f"entirely, so every large array closure-captures into the compiled program"
    )
    ssp_size = int(np.asarray(ssp.ssp_flux).size)
    leaves = jax.tree_util.tree_leaves(args["_jit_inputs"])
    assert any(int(getattr(leaf, "size", 0)) == ssp_size for leaf in leaves), (
        f"[{surface}/{channel}] the SSP grid (size {ssp_size}) is not among the threaded "
        f"_jit_inputs leaves {[int(getattr(x, 'size', 0)) for x in leaves]}"
    )


@pytest.mark.parametrize("channel", ["photometry", "spectroscopy", "joint"])
@pytest.mark.parametrize("surface", ["sed_model", "forward_model"])
def test_ssp_grid_is_not_baked_at_any_depth(synthetic_ssp_wide, surface, channel):
    """No array as large as the SSP grid may be a const at any nesting depth."""
    ssp = synthetic_ssp_wide
    fitter = _fitter_on_surface(surface, ssp, channel)

    loss_fn = build_loss_fn(fitter)
    init = fitter._initialize_unbounded(jax.random.PRNGKey(0))
    ssp_size = int(np.asarray(ssp.ssp_flux).size)
    biggest = _largest_baked_const_at_any_depth(loss_fn, init, fitter._data_args)
    assert biggest < ssp_size, (
        f"[{surface}/{channel}] a constant of size {biggest} >= SSP grid size {ssp_size} is "
        f"frozen into the loss jaxpr at some depth. On a real SSP that is 134 MB of hex "
        f"per copy; XLA then compiles a ~275 MB program and the process is OOM-killed."
    )


def test_forward_model_exposes_ssp_data():
    """The delegation list must carry ``ssp_data``.

    Pinned as its own test because the failure mode is one missing string in
    ``_DELEGATED_TO_INNER_SED``, and the consequence is invisible: the read raises
    ``AttributeError`` inside a ``suppress`` and threading turns off.
    """
    from tengri import ForwardModel

    assert "ssp_data" in ForwardModel._DELEGATED_TO_INNER_SED


def test_catalog_of_n_galaxies_threads_and_reuses_one_composite(synthetic_ssp_wide):
    """N galaxies must share ONE compiled program with the SSP threaded into it.

    The batched/catalog path takes its ``data_args`` template from
    ``Fitter._data_args`` and substitutes only ``data``/``noise``/``presence`` per
    galaxy, so it inherits both the bug and the fix. Two things must hold:

    * the shared composite (SSP grid, templates) rides in as a **traced argument**,
      so it is not re-inlined into the program -- once, let alone once per galaxy;
    * the flat log-density is **cached on the model**, so galaxy 2 reuses galaxy 1's
      compiled code instead of triggering a fresh compile.

    Without threading, the per-galaxy program embeds the whole grid as a constant;
    at a real grid size that is 134 MB of inlined hex per copy.
    """
    from tengri import ForwardModel
    from tengri.inference.backends.mcmc._shared import _get_flat_logdensity

    ssp = synthetic_ssp_wide
    ssp_size = int(np.asarray(ssp.ssp_flux).size)
    obs, data = _obs_and_data("photometry")
    model = _build_model(ssp, obs)
    forward = ForwardModel.build(sed=model, observation=obs)
    noise = 0.1 * jnp.ones_like(data)

    # ONE fitter serves the whole catalog -- that is how the batched path works:
    # per-galaxy data/noise/presence are substituted into this shared template.
    fitter = Fitter(forward, data, noise, data_type="photometry")
    init = fitter._initialize_unbounded(jax.random.PRNGKey(0))
    fn1, _unravel, _flat, args = _get_flat_logdensity(fitter, init)

    assert "_jit_inputs" in args, (
        "the catalog data_args template has no '_jit_inputs' — every per-galaxy "
        "program would bake the shared composite instead of receiving it"
    )
    leaves = jax.tree_util.tree_leaves(args["_jit_inputs"])
    assert any(int(getattr(x, "size", 0)) == ssp_size for x in leaves), (
        f"SSP grid (size {ssp_size}) absent from the threaded composite"
    )

    # Reuse across galaxies: the composite is fetched once and cached, so the second
    # galaxy gets the identical callable rather than a fresh trace.
    fn2, _u2, _f2, args2 = _get_flat_logdensity(fitter, init)
    assert fn1 is fn2, (
        "the flat log-density was rebuilt on a second lookup — every galaxy would "
        "pay its own compile instead of reusing the shared composite"
    )
    assert "_jit_inputs" in args2

    flat, _ = jax.flatten_util.ravel_pytree(init)
    biggest = _largest_baked_const_at_any_depth(fn1, flat, args)
    assert biggest < ssp_size, (
        f"the catalog log-density bakes a constant of size {biggest} >= SSP grid "
        f"{ssp_size}; per-galaxy programs must receive the composite, not embed it"
    )


def test_two_fitters_on_one_forward_model_share_the_composite(synthetic_ssp_wide):
    """Sequential per-galaxy fits must reuse ONE compiled program.

    Every ``Fitter`` resolves ``approx`` and clones the model, and the compile
    caches key on model **identity** — so N sequential fits over one
    ``ForwardModel`` produced N clones, N cache misses and N compiles, even though
    their ``_engine_cache_key()`` values were already identical. Fixed by memoizing
    the resolved clone per (source model, resolved config), so identity-keyed
    caches hit without changing what any cache key means.

    (This docstring came from ``test_two_fitters_on_one_forward_model_share_one_compile``,
    which sat directly above with the same explanation and an empty body — it
    asserted nothing and could never fail, while this test already checked
    exactly what it described.)
    """
    from tengri import ForwardModel
    from tengri.inference.backends.mcmc._shared import _get_flat_logdensity

    obs, data = _obs_and_data("photometry")
    model = _build_model(synthetic_ssp_wide, obs)
    forward = ForwardModel.build(sed=model, observation=obs)
    noise = 0.1 * jnp.ones_like(data)

    f1 = Fitter(forward, data, noise, data_type="photometry")
    f2 = Fitter(forward, 1.7 * data, noise, data_type="photometry")
    assert f1._engine_cache_key() == f2._engine_cache_key()

    init = f1._initialize_unbounded(jax.random.PRNGKey(0))
    assert f1.model is f2.model, (
        "the two fitters resolved to different clone objects, so every "
        "identity-keyed compile cache will miss and each galaxy recompiles"
    )
    fn1, *_ = _get_flat_logdensity(f1, init)
    fn2, *_ = _get_flat_logdensity(f2, init)
    assert fn1 is fn2, "the flat log-density was rebuilt for the second galaxy"


@pytest.mark.parametrize("channel", ["photometry", "spectroscopy", "joint"])
@pytest.mark.parametrize("surface", ["sed_model", "forward_model"])
def test_threading_does_not_change_the_number_on_either_surface(
    synthetic_ssp_wide, surface, channel
):
    """Threading moves WHERE the composite enters; it must not move the answer.

    The original bit-exactness guard only ran the deprecated ``Fitter(SEDModel)``
    surface -- the one where threading already worked. Enabling threading on
    ``ForwardModel`` activated code that had never executed, so it needs its own
    comparison: same loss, ``_jit_inputs`` present vs stripped.
    """
    fitter = _fitter_on_surface(surface, synthetic_ssp_wide, channel)
    loss_fn = build_loss_fn(fitter)
    init = fitter._initialize_unbounded(jax.random.PRNGKey(3))

    threaded = fitter._data_args
    baked = {k: v for k, v in threaded.items() if k != "_jit_inputs"}

    v_threaded = float(loss_fn(init, threaded))
    v_baked = float(loss_fn(init, baked))
    assert np.isfinite(v_threaded), f"[{surface}/{channel}] threaded loss is not finite"
    assert v_threaded == pytest.approx(v_baked, rel=1e-12, abs=1e-9), (
        f"[{surface}/{channel}] threaded {v_threaded} != baked {v_baked} — threading "
        f"changed the physics, not just the calling convention"
    )


def test_hierarchical_forwards_are_excluded_from_threading(synthetic_ssp, simple_observation):
    """A hierarchical forward must NOT get ``_jit_inputs``.

    The threaded forward is written for a single-population SED forward; on a
    :class:`PopulationSEDModel` it mis-broadcasts the galaxy axis against the SFH
    grid (``mul got incompatible shapes (256,), (3,)``).

    This exclusion used to happen by ACCIDENT: ``_build_data_args`` read
    ``model.ssp_data``, ``ForwardModel`` did not delegate it, and a
    ``contextlib.suppress`` ate the ``AttributeError``. Adding that delegation to
    fix single-galaxy threading turned threading *on* for hierarchical fits and
    broke two of them in CI. The exclusion is now stated via
    ``_supports_jit_threading``, and this pins it.
    """
    from tengri import FIXED, SEDModel, Uniform
    from tengri.forward.population_sed_model import PopulationSEDModel

    template = SEDModel.build(
        ssp_data=synthetic_ssp,
        observation=simple_observation,
        sfh={"type": "dpl", "*": FIXED, "log_total_mass": Uniform(8, 12)},
        dust={"law_diff": 'calzetti', "type": "two_component", "law_bc": "calzetti", "*": FIXED},
        neb={"type": "none"},
        redshift=Fixed(0.5),
    )
    from tengri import ForwardModel

    pop = PopulationSEDModel(
        sed=template,
        galaxies=[
            {"flux_obs": jnp.ones(3) * 1e-18, "noise": jnp.ones(3) * 1e-19} for _ in range(3)
        ],
    )
    forward = ForwardModel.build(population=pop, observation=simple_observation)
    assert forward._supports_jit_threading() is False, (
        "a hierarchical forward reports itself threadable; the threaded forward "
        "mis-broadcasts the galaxy axis there"
    )

    fitter = Fitter(forward)
    assert "_jit_inputs" not in fitter._data_args, (
        "a hierarchical fit built _jit_inputs — the threaded forward will raise "
        "TypeError on the galaxy axis"
    )


def test_single_population_forwards_are_still_threadable(synthetic_ssp_wide):
    """Neuter for the exclusion above: it must not switch threading off generally."""
    from tengri import ForwardModel

    obs = Observation(photometry=_PHOT)
    forward = ForwardModel.build(sed=_build_model(synthetic_ssp_wide, obs), observation=obs)
    assert forward._supports_jit_threading() is True
