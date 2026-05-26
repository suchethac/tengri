# SPDX-License-Identifier: BSD-3-Clause
"""JIT-vs-eager parity for canonical SEDModel forward paths.

`tests/integration/test_components_jit.py` already covers the
orchestrator chain for `Stellar + Radio + XRay + IGM` (+ optional AGN/dust),
asserting bit-exactness with `jnp.allclose(rtol=1e-12)`.  This file adds the
broader user-facing surface: the `SEDModel.predict_*` methods that recipes
expose.  The contract is identical — JIT and eager must produce the same
PyTree of derived quantities to within `rtol=1e-6` — but the entry point
differs, which catches regressions in the SEDModel facade rather than in
`run_components`.

The test uses the session-scoped ``synthetic_ssp`` and ``simple_observation``
fixtures from ``tests/conftest.py``; no real SSP file required, so this runs
in the default suite.
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


def test_predict_photometry_finite(model_pair):
    model, params = model_pair
    phot = model.predict_photometry(params)
    chex.assert_tree_all_finite(phot)


@pytest.mark.parametrize("variant", ["eager", "jit"])
def test_predict_photometry_runs(model_pair, variant):
    model, params = model_pair
    run = jax.jit(model.predict_photometry) if variant == "jit" else model.predict_photometry
    chex.assert_tree_all_finite(run(params))


def test_predict_photometry_jit_matches_eager(model_pair):
    model, params = model_pair
    eager = model.predict_photometry(params)
    jitted = jax.jit(model.predict_photometry)(params)
    chex.assert_trees_all_close(jitted, eager, rtol=1e-6)


def test_predict_rest_sed_jit_matches_eager(model_pair):
    model, params = model_pair
    eager = model.predict_rest_sed(params)
    jitted = jax.jit(model.predict_rest_sed)(params)
    chex.assert_trees_all_close(jitted, eager, rtol=1e-6)


def test_grad_finite_through_predict_photometry(model_pair):
    """Gradients flow through the JIT-compiled predict_photometry."""
    model, params = model_pair

    def loss(p):
        return jnp.sum(model.predict_photometry(p))

    g = jax.jit(jax.grad(loss))(params)
    chex.assert_tree_all_finite(g)
