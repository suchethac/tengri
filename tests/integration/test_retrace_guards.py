# SPDX-License-Identifier: BSD-3-Clause
"""Retrace guards for the user-facing SEDModel forward paths.

`chex.assert_max_traces(n=1)` fails the wrapped function if JAX retraces it
more than once during a single test process — the canonical symptom of an
accidental retrace is a weakly-typed Python scalar leaking into a JIT
boundary, or a PyTree definition changing shape between calls. Both have
historically caused 30-90s compile penalties on tengri (geoVI on first
warmup ≈ 75s; tier-1 photometry ≈ 30s), and are the failure mode behind
the project memory entry on long compiles / OOMs.

The guards here protect the most user-facing entry points
(`SEDModel.predict_photometry`, `SEDModel.predict_rest_sed`, and a
JIT-grad composition over `predict_photometry`). They are tight by design
— `n=1` — and use the session-scoped synthetic SSP + simple observation
fixtures, so they run in the default suite and don't depend on real data.
"""

from __future__ import annotations

import chex
import jax
import jax.numpy as jnp
import pytest

from tengri import Fixed, Parameters, SEDModel
from tengri.components.igm.component import IGMSEDComponent
from tengri.components.radio.component import RadioSEDComponent
from tengri.components.xray.component import XRaySEDComponent
from tengri.protocols.component import ForwardState


def _build_model(synthetic_ssp, simple_observation, *, redshift=0.1):
    spec = Parameters(
        mean_sfh_type="dpl",
        sfh_dpl_alpha=Fixed(1.5),
        sfh_dpl_beta=Fixed(2.0),
        sfh_dpl_tau_gyr=Fixed(5.0),
        sfh_dpl_log_total_mass=Fixed(0.0),
        met_logzsol=Fixed(-0.5),
        dust_tau_bc=Fixed(0.5),
        dust_tau_diff=Fixed(0.2),
        dust_slope=Fixed(-0.7),
        redshift=Fixed(redshift),
    )
    return SEDModel(spec, synthetic_ssp, observation=simple_observation), spec


@pytest.fixture(scope="module")
def model_pair(synthetic_ssp, simple_observation):
    model, spec = _build_model(synthetic_ssp, simple_observation)
    key = jax.random.PRNGKey(0)
    params = spec.sample(key)
    return model, params


def test_predict_photometry_traces_once(model_pair):
    """Repeated same-shape calls must not retrigger XLA compilation."""
    model, params = model_pair

    @jax.jit
    @chex.assert_max_traces(n=1)
    def pred(p):
        return model.predict_photometry(p)

    # First call: traces once (allowed by n=1)
    pred(params)
    # Subsequent same-shape calls: no retrace
    pred(params)
    pred(params)


def test_predict_rest_sed_traces_once(model_pair):
    """Same guard for the rest-frame SED entry point."""
    model, params = model_pair

    @jax.jit
    @chex.assert_max_traces(n=1)
    def pred(p):
        return model.predict_rest_sed(p)

    pred(params)
    pred(params)


def test_jit_grad_predict_photometry_traces_once(model_pair):
    """jit ∘ grad of the user-facing predict_photometry must not retrace
    across same-shape gradient calls — this is the hot path for any
    gradient-based fitter (MAP, VI, NUTS warmup) and the place a stray
    Python-scalar leak typically shows up first."""
    model, params = model_pair

    def loss(p):
        return jnp.sum(model.predict_photometry(p))

    @jax.jit
    @chex.assert_max_traces(n=1)
    def grad_loss(p):
        return jax.grad(loss)(p)

    grad_loss(params)
    grad_loss(params)


# ── Per-component guards (no SSP needed, run in <100 ms each) ───────────────
#
# Radio, X-ray, and IGM components are stellar-population-independent — they
# read cross-component scalars (L_ir, L_agn_bol, log_mstar) from
# ``state.derived`` with documented fallbacks, not from the SSP grid. That
# makes them ideal for fast per-component guards: no real SSP file, no
# stellar synthesis, no nebular emulator. Each guard exercises one
# component's ``apply()`` end-to-end (eager + JIT + grad) and confirms
# (a) the output is finite, (b) JIT matches eager bit-for-bit modulo XLA,
# and (c) no retrace fires across same-shape repeated calls.


@pytest.fixture(scope="module")
def synth_wave():
    # 64 wavelengths spanning the radio→X-ray range so every per-component
    # apply has meaningful flux in its native band.
    return jnp.logspace(0.0, 8.0, 64)  # 1 Å to 10⁸ Å


def _component_default_params(component, **bare):
    """Build a full param dict from a component's ``declared_parameters()``
    defaults, plus any bare (non-prefixed) params supplied explicitly.

    Deriving from the declared set — rather than a hand-maintained dict — keeps
    the guard from drifting when a component adds a parameter: the earlier
    hand-written ``xray_params`` was missing the declared ``xray_log_nh``, so
    ``apply`` KeyError'd (#870, #768). Any newly declared param is now picked up
    automatically at its physically-motivated default.
    """
    params = {d.name: jnp.asarray(d.prior.default) for d in component.declared_parameters()}
    params.update({k: jnp.asarray(v) for k, v in bare.items()})
    return params


@pytest.fixture(scope="module")
def radio_params():
    return _component_default_params(RadioSEDComponent(), redshift=0.1)


@pytest.fixture(scope="module")
def xray_params():
    return _component_default_params(XRaySEDComponent(), redshift=0.1)


@pytest.fixture(scope="module")
def igm_params():
    return _component_default_params(IGMSEDComponent(), redshift=3.0)


COMPONENT_CASES = [
    ("radio", RadioSEDComponent, "radio_params"),
    ("xray", XRaySEDComponent, "xray_params"),
    ("igm", IGMSEDComponent, "igm_params"),
]


def _state0(wave):
    """Build a ForwardState seeded with both rest-frame and observed-frame
    arrays so IGM (which short-circuits when ``sed_observed`` is None) and
    radio/xray (which write into ``sed_intrinsic``) all have something to
    operate on."""
    sed = jnp.zeros_like(wave)
    return ForwardState(wave=wave, sed_intrinsic=sed, sed_observed=jnp.ones_like(wave))


def _component_output(name, state):
    """The published array each component is checked against:

    - radio/xray write into ``state.sed_intrinsic``
    - IGM publishes ``state.derived["igm_transmission"]``
    """
    return state.derived["igm_transmission"] if name == "igm" else state.sed_intrinsic


@pytest.mark.parametrize("name,cls,params_fixture", COMPONENT_CASES)
def test_component_apply_finite_and_shape(name, cls, params_fixture, synth_wave, request):
    """Each component output is finite on its native wave grid and preserves
    the input wavelength shape."""
    params = request.getfixturevalue(params_fixture)
    out = cls().apply(_state0(synth_wave), params)
    arr = _component_output(name, out)
    chex.assert_tree_all_finite(arr)
    chex.assert_equal_shape([arr, synth_wave])


@pytest.mark.parametrize("name,cls,params_fixture", COMPONENT_CASES)
def test_component_jit_matches_eager(name, cls, params_fixture, synth_wave, request):
    """Per-component bit-for-bit (modulo XLA) JIT/eager parity.

    Tighter than the SEDModel-level parity in test_variant_parity.py because
    here we isolate a single component — if this fails, the regression is
    localized to that adapter, not the orchestrator."""
    params = request.getfixturevalue(params_fixture)
    component = cls()
    state0 = _state0(synth_wave)
    out_eager = component.apply(state0, params)
    out_jit = jax.jit(component.apply)(state0, params)
    chex.assert_trees_all_close(
        _component_output(name, out_jit),
        _component_output(name, out_eager),
        rtol=1e-6,
    )


@pytest.mark.parametrize("name,cls,params_fixture", COMPONENT_CASES)
def test_component_apply_traces_once(name, cls, params_fixture, synth_wave, request):
    """Same-shape repeated calls into one component must not retrace.

    The ``chex.clear_trace_counter()`` reset is required because
    ``@chex.assert_max_traces`` uses a function-level closure counter that
    survives across parametrize cases otherwise (chex's own error message
    flags this as the canonical pitfall)."""
    chex.clear_trace_counter()
    params = request.getfixturevalue(params_fixture)
    component = cls()
    state0 = _state0(synth_wave)

    @jax.jit
    @chex.assert_max_traces(n=1)
    def apply(s, p):
        return component.apply(s, p)

    apply(state0, params)
    apply(state0, params)
    apply(state0, params)
