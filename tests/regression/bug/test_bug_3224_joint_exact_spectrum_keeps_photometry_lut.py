# SPDX-License-Identifier: BSD-3-Clause
r"""A joint observation with an exact spectrum keeps its photometry on the WavePrecomp LUT (#3224).

``approx="auto"`` resolves a joint fit to ``WavePrecomp`` and puts the spectrum channel on
the spectrum LUT only where that LUT serves the pixels exactly (#2958). Pixels wider than
the model grid need the flux-conserving pixel integral (#2530), so the spectrum stays
exact. The photometry channel still projects through the LUT: its gradient costs the
photometry-only model's FLOPs. FLOPs, not wall clock, because they are a property of the
compiled graph (see ``test_bug_1748_feature_precomp_effect.py``).

Pinned here:

* the joint photometry equals the photometry-only LUT value, fixed and free redshift;
* the joint spectrum equals the spectrum of the same model's lazy ``predict`` state,
  which does not go through the observables kernel. It is the joint ``WavePrecomp``
  spectrum, not ``approx=None`` in general: an ``L_ir`` consumer feeds the energy-balance
  LUT, so the two differ there;
* a model with a nebular grid attached keeps exact photometry, because serving photometry
  from the grid zeroes ``sed_nebular`` and the exact spectrum reads it.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.regression_bug

_FILTERS = ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z", "des_g", "des_r", "des_i"]
# 40 pixels over 3000 A: 75 A pixels, wider than the SSP grid, so the pixel integral is required.
_SPEC_WAVE = jnp.linspace(4000.0, 7000.0, 40)


def _observation(*, joint: bool, filters=_FILTERS):
    from tengri import Observation, Photometry, Spectroscopy

    photometry = Photometry.from_names(filters)
    if joint:
        return Observation(photometry=photometry, spectroscopy=Spectroscopy(wave_obs=_SPEC_WAVE))
    return Observation(photometry=photometry)


def _model(ssp, *, joint: bool, approx, **overrides):
    from tengri import SEDModel, recipes

    return SEDModel.build(
        ssp_data=ssp,
        observation=_observation(joint=joint),
        **{**recipes.mock_recovery_minimal(), "approx": approx, **overrides},
    )


def _grad_flops(model, fn_name, truth):
    free = list(model.spec.free_params)
    v0 = jnp.asarray([truth[k] for k in free])

    def loss(v):
        params = dict(truth)
        params.update({k: v[i] for i, k in enumerate(free)})
        return jnp.sum(getattr(model, fn_name)(params))

    return int(jax.jit(jax.grad(loss)).lower(v0).compile().cost_analysis()["flops"])


def _assert_joint_matches_references(phot_only, joint, seeds=(1, 2, 3)):
    for seed in seeds:
        truth = phot_only.spec.sample(jax.random.PRNGKey(seed))
        np.testing.assert_allclose(
            np.asarray(joint.predict_photometry(truth)),
            np.asarray(phot_only.predict_photometry(truth)),
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            np.asarray(joint.predict_spectrum(truth)),
            np.asarray(joint.predict(truth).spectrum()),
            rtol=1e-12,
        )


@pytest.fixture(scope="module")
def models(ssp_data_fsps):
    from tengri import WavePrecomp

    return {
        "phot": _model(ssp_data_fsps, joint=False, approx=WavePrecomp()),
        "joint": _model(ssp_data_fsps, joint=True, approx=WavePrecomp()),
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


def test_joint_matches_the_photometry_only_lut_and_the_lazy_spectrum(models):
    _assert_joint_matches_references(models["phot"], models["joint"])


def test_free_redshift_with_an_igm_keeps_the_same_parity(ssp_data_fsps):
    from tengri import Uniform, WavePrecomp

    extra = {"redshift": Uniform(0.3, 2.5), "igm": {"type": "inoue14"}}
    phot = _model(ssp_data_fsps, joint=False, approx=WavePrecomp(), **extra)
    joint = _model(ssp_data_fsps, joint=True, approx=WavePrecomp(), **extra)
    assert not joint._approx.get("spectrum_precomp")
    _assert_joint_matches_references(phot, joint, seeds=(1, 2))


def test_a_nebular_grid_keeps_the_photometry_exact(ssp_data_fsps, monkeypatch):
    """A nebular grid zeroes sed_nebular, so the LUT must not serve the photometry."""
    from tengri import DEFAULT, FeaturePrecomp, Fixed, SEDModel, Uniform, WavePrecomp
    from tengri.observation import Observation

    joint = SEDModel.build(
        ssp_data=ssp_data_fsps,
        observation=_observation(joint=True, filters=["sdss_g", "sdss_r", "sdss_i"]),
        sfh={
            "type": "delayed",
            "all_params": Fixed(DEFAULT),
            "log_total_mass": Uniform(9.0, 11.0),
        },
        dust_attenuation={"type": "none"},
        neb={"type": "cue", "all_params": Fixed(DEFAULT)},
        redshift=Fixed(0.1),
        approx=(WavePrecomp(), FeaturePrecomp()),
    )
    assert getattr(joint, "_nebular_grid_table", None) is not None, "setup: no grid attached"

    def refuse(*args, **kwargs):
        raise AssertionError("the LUT served the photometry of a model with a nebular grid")

    monkeypatch.setattr(Observation, "predict_via_precomp", refuse)
    flux = joint.predict_photometry({"sfh_delayed_log_total_mass": 10.0})
    assert bool(jnp.all(jnp.isfinite(flux)))
