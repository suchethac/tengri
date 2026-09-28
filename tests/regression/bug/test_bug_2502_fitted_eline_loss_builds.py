# SPDX-License-Identifier: BSD-3-Clause
r"""Regression test for #2502: loss probe must carry every Fitter latent.

The loss function's reference parameter point was drawn from spec.sample(),
which returns free parameters only (#2296). Under fitted-mode emission lines,
the Fitter registers eline amplitudes as additional free parameters via
merge_observation_params, but model.spec may not include them. The channel
scale probe then fails with KeyError when the likelihood tries to read the
missing amplitude names.

The fix: sample from fitter.spec (which has the merged parameters) and merge
in fitter._fixed_values, so the reference point spans every parameter in
fitter._free_names + fitter._fixed_values.
"""

import types

import jax.numpy as jnp
import pytest

from tengri.inference.fitter import Fitter
from tengri.observation.spectroscopy import Spectroscopy
from tengri.parameters.parameters import Parameters

pytestmark = pytest.mark.regression_bug


@pytest.fixture
def fitted_eline_model():
    """Build a minimal fitted-mode eline model for the probe test."""
    wave = jnp.linspace(6500, 6600, 15)
    model_spec = Parameters(
        mean_sfh_type="dpl",
        sfh_dpl_age_gyr=2.0,
        sfh_dpl_alpha=2.0,
        sfh_dpl_beta=2.5,
        sfh_dpl_log_total_mass=11.0,
        sfh_dpl_tau_gyr=0.5,
        met_logzsol=0.0,
        dust_tau_bc=0.1,
        dust_tau_diff=0.1,
        redshift=0.0,
    )
    cfg = Spectroscopy(wave_obs=wave, eline_mode="fitted")
    model = types.SimpleNamespace(
        spec=model_spec,
        wave_obs=wave,
        _spectral_resolution=2000.0,
        predict_spectrum=lambda params, w=None, **kwargs: jnp.ones(len(wave)),
        observation=types.SimpleNamespace(spectroscopy=cfg, photometry=None),
    )
    return model, wave


def test_fitted_mode_loss_fn_builds(fitted_eline_model):
    """Loss function must build without KeyError under fitted eline mode."""
    model, wave = fitted_eline_model
    n_pix = len(wave)
    data = jnp.ones(n_pix) * 10.0
    noise = jnp.ones(n_pix) * 0.1
    fitter = Fitter(model, data, noise, data_type="spectroscopy")

    # Probe must complete (not raise KeyError on missing eline amplitude names)
    loss_fn = fitter._build_loss_fn()
    assert callable(loss_fn)


def test_fitted_mode_loss_fn_finite_at_zero_amp(fitted_eline_model):
    """Loss at zero amplitude must be finite (no NaN/Inf from probe failure)."""
    model, wave = fitted_eline_model
    n_pix = len(wave)
    data = jnp.ones(n_pix) * 10.0
    noise = jnp.ones(n_pix) * 0.1
    fitter = Fitter(model, data, noise, data_type="spectroscopy")

    loss_fn = fitter._build_loss_fn()
    params_u = {nm: jnp.array(0.0) for nm in fitter._free_names}
    loss_val = loss_fn(params_u, {"data": data, "noise": noise})

    assert jnp.isfinite(loss_val), f"Loss is not finite: {loss_val}"


def test_fitted_mode_loglikelihood_fn_finite(fitted_eline_model):
    """Log-likelihood must be finite at zero amplitude."""
    model, wave = fitted_eline_model
    n_pix = len(wave)
    data = jnp.ones(n_pix) * 10.0
    noise = jnp.ones(n_pix) * 0.1
    fitter = Fitter(model, data, noise, data_type="spectroscopy")

    ll_fn = fitter._build_loglikelihood_fn()
    params_u = {nm: jnp.array(0.0) for nm in fitter._free_names}
    ll_val = ll_fn(params_u, {"data": data, "noise": noise})

    assert jnp.isfinite(ll_val), f"Log-likelihood is not finite: {ll_val}"
