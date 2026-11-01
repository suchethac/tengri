# SPDX-License-Identifier: BSD-3-Clause
r"""A joint observation with an exact spectrum keeps its photometry on the WavePrecomp LUT.

``approx="auto"`` resolves a joint fit to ``WavePrecomp`` and promotes the spectrum
channel onto the spectrum LUT only where that LUT serves the pixels exactly (#2958).
Pixels wider than the model grid need the flux-conserving pixel integral (#2530), so the
spectrum stays exact. Until this guard, that exact spectrum also dragged the *photometry*
onto the exact filter integration: the LUT projection was reachable only inside the
spectrum-LUT branch. Gradient FLOPs of the joint photometry channel, 8 bands, D=7:

* before: 57,907,776, 24x the photometry-only model's 2,409,005;
* after: 2,409,005, identical to the photometry-only model.

FLOPs, not wall clock, because they are a property of the compiled graph (see
``test_bug_1748_feature_precomp_effect.py``).

The photometry is now the LUT's, so it differs from the exact integral by the LUT's own
bias (about 2e-4 per band here), the same bias the photometry-only fit carries. The
spectrum is untouched: bit-identical to the ``approx=None`` model.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.regression_bug

_FILTERS = ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z", "des_g", "des_r", "des_i"]
# 40 pixels over 3000 A: 75 A pixels, wider than the SSP grid, so the pixel integral is required.
_SPEC_WAVE = jnp.linspace(4000.0, 7000.0, 40)


def _model(ssp, *, joint: bool, approx):
    from tengri import Observation, Photometry, SEDModel, Spectroscopy, recipes

    photometry = Photometry.from_names(_FILTERS)
    obs = (
        Observation(photometry=photometry, spectroscopy=Spectroscopy(wave_obs=_SPEC_WAVE))
        if joint
        else Observation(photometry=photometry)
    )
    return SEDModel.build(
        ssp_data=ssp, observation=obs, **{**recipes.mock_recovery_minimal(), "approx": approx}
    )


def _grad_flops(model, fn_name, truth):
    free = list(model.spec.free_params)
    v0 = jnp.asarray([truth[k] for k in free])

    def loss(v):
        params = dict(truth)
        params.update({k: v[i] for i, k in enumerate(free)})
        return jnp.sum(getattr(model, fn_name)(params))

    return int(jax.jit(jax.grad(loss)).lower(v0).compile().cost_analysis()["flops"])


@pytest.fixture(scope="module")
def models(ssp_data_fsps):
    from tengri import WavePrecomp

    return {
        "phot": _model(ssp_data_fsps, joint=False, approx=WavePrecomp()),
        "joint": _model(ssp_data_fsps, joint=True, approx=WavePrecomp()),
        "joint_exact": _model(ssp_data_fsps, joint=True, approx=None),
    }


def test_setup_the_spectrum_really_is_exact(models):
    """Guard the guard: if the spectrum LUT engaged, the mixed route is not what is measured."""
    cfg = models["joint"]._approx
    assert cfg.get("wave_precomp")
    assert not cfg.get("spectrum_precomp")


def test_joint_photometry_gradient_costs_the_photometry_only_model(models):
    truth = models["phot"].spec.sample(jax.random.PRNGKey(1))
    phot = _grad_flops(models["phot"], "predict_photometry", truth)
    joint = _grad_flops(models["joint"], "predict_photometry", truth)
    assert joint <= 1.1 * phot, f"joint photometry {joint:,} FLOPs vs photometry-only {phot:,}"


def test_joint_photometry_is_the_photometry_only_lut_value(models):
    for seed in (1, 2, 3):
        truth = models["phot"].spec.sample(jax.random.PRNGKey(seed))
        np.testing.assert_allclose(
            np.asarray(models["joint"].predict_photometry(truth)),
            np.asarray(models["phot"].predict_photometry(truth)),
            rtol=1e-12,
        )


def test_joint_spectrum_is_untouched(models):
    for seed in (1, 2, 3):
        truth = models["phot"].spec.sample(jax.random.PRNGKey(seed))
        np.testing.assert_allclose(
            np.asarray(models["joint"].predict_spectrum(truth)),
            np.asarray(models["joint_exact"].predict_spectrum(truth)),
            rtol=1e-12,
        )
