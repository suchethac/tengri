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


def _build_model(synthetic_ssp, simple_observation, *, redshift=0.1):
    spec = Parameters(
        mean_sfh_type="dpl",
        sfh_dpl_alpha=Fixed(1.5),
        sfh_dpl_beta=Fixed(2.0),
        sfh_dpl_tau_gyr=Fixed(5.0),
        sfh_dpl_log_peak_sfr=Fixed(0.0),
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
