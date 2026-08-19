# SPDX-License-Identifier: BSD-3-Clause
"""Regression: a redshift override on a ztable model must be a RUNTIME input (#1316).

Spec #1320 §9.4 (binding): "Per-galaxy redshift is explicitly not a [recompile]
trigger" when the model carries a ``catalog_z_range`` ztable. The shipped sequential
path baked every per-row ``params_override={"redshift": z}`` into
``Fitter._fixed_values`` and the engine cache key (#1331's design, correct for
arbitrary fixed params), so each distinct z minted a new loss signature and
recompiled — measured 32 → 48 compiles over one 6-row catalog on the issue.

The fix routes a redshift override through ``data_args`` instead — the seam #1349
built for the batched MCMC engine (``build_loss_fn`` replaces the baked value with
``data_args["redshift"]``) — whenever the model's ztable can take z at runtime.
Overrides on models *without* a ``catalog_z_range`` keep the bake (there the value
genuinely is a compile constant).
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

from tengri import FIXED, Fixed, SEDModel, Uniform, WavePrecomp
from tengri.inference.fitter import Fitter

pytestmark = pytest.mark.regression_bug


def _build(ssp, obs, approx):
    return SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        sfh={"type": "dpl", "all_params": FIXED, "log_total_mass": Uniform(9.0, 11.0)},
        dust={"law_diff": 'calzetti', "type": "two_component", "law_bc": "calzetti", "all_params": FIXED},
        neb={"type": "none"},
        redshift=Fixed(0.3),
        approx=approx,
    )


@pytest.fixture(scope="module")
def ztable_model(synthetic_ssp_wide, synthetic_tophat_obs):
    return _build(
        synthetic_ssp_wide,
        synthetic_tophat_obs,
        WavePrecomp(catalog_z_range=(0.05, 0.9), n_z=32),
    )


@pytest.fixture(scope="module")
def mock_data(ztable_model):
    params = ztable_model.spec.sample(jax.random.PRNGKey(0))
    flux = np.asarray(ztable_model.predict_photometry(params))
    return flux, 0.05 * np.abs(flux)


def test_distinct_z_share_one_engine_key_and_loss(ztable_model, mock_data):
    """Two overrides differing only in z: same engine key, same loss object."""
    flux, err = mock_data
    f1 = Fitter(ztable_model, flux, err, data_type="photometry", params_override={"redshift": 0.2})
    f2 = Fitter(ztable_model, flux, err, data_type="photometry", params_override={"redshift": 0.7})

    assert f1._engine_cache_key() == f2._engine_cache_key(), (
        "distinct runtime redshifts must not fork the compile signature (spec §9.4)"
    )
    assert f1._get_or_build_loss_fn() is f2._get_or_build_loss_fn(), (
        "the model-keyed loss cache must serve both fits with one function"
    )


def test_runtime_z_rides_data_args_and_reaches_the_loss(ztable_model, mock_data):
    """The routed z lands in data_args, and the loss actually depends on it."""
    flux, err = mock_data
    f1 = Fitter(ztable_model, flux, err, data_type="photometry", params_override={"redshift": 0.2})
    assert "redshift" in f1._data_args, "routed override must ride data_args (#1349 seam)"
    assert float(f1._data_args["redshift"]) == 0.2

    from tengri.inference.context import InferenceContext

    ctx = InferenceContext.from_target(f1)
    loss = ctx.neg_log_posterior_fn
    p_u = ctx.initial_params(jax.random.PRNGKey(3))
    l_a = float(loss(p_u, ctx.data_args))
    l_b = float(loss(p_u, {**ctx.data_args, "redshift": 0.7}))
    assert l_a != l_b, "loss is blind to the runtime redshift — routing is a no-op"


def test_override_is_still_reported(ztable_model, mock_data):
    """posterior.params must echo the override even though it is not baked."""
    flux, err = mock_data
    f = Fitter(ztable_model, flux, err, data_type="photometry", params_override={"redshift": 0.7})
    post = f.run("map", n_steps=3, verbose=False)
    assert abs(float(post.params["redshift"]) - 0.7) < 1e-12


def test_no_ztable_model_keeps_the_bake(synthetic_ssp_wide, synthetic_tophat_obs):
    """Without a catalog_z_range the z override IS a compile constant — keys differ."""
    model = _build(synthetic_ssp_wide, synthetic_tophat_obs, None)
    params = model.spec.sample(jax.random.PRNGKey(0))
    flux = np.asarray(model.predict_photometry(params))
    err = 0.05 * np.abs(flux)

    f1 = Fitter(model, flux, err, data_type="photometry", params_override={"redshift": 0.2})
    f2 = Fitter(model, flux, err, data_type="photometry", params_override={"redshift": 0.7})

    assert f1._engine_cache_key() != f2._engine_cache_key(), (
        "no ztable: distinct baked redshifts must keep distinct signatures (#1331)"
    )
    assert "redshift" not in f1._data_args


def test_non_redshift_overrides_still_bake(ztable_model, mock_data):
    """Only redshift is LUT-runtime; other fixed params keep #1331's bake + key."""
    flux, err = mock_data
    f1 = Fitter(
        ztable_model, flux, err, data_type="photometry", params_override={"dust_tau_bc": 0.4}
    )
    f2 = Fitter(
        ztable_model, flux, err, data_type="photometry", params_override={"dust_tau_bc": 0.9}
    )

    assert f1._engine_cache_key() != f2._engine_cache_key()
