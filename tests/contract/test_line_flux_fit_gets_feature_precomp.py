# SPDX-License-Identifier: BSD-3-Clause
"""Contract: fitting a line-flux channel routes through the line LUT by default.

``Fitter(approx="auto")`` has been the default since #1180, and its docstring
promises ``FeaturePrecomp`` "is appended when emission lines are fit". It was
not, for the channel most users mean by that: the trigger read
``_eline_marginalize`` / ``_eline_fitted`` — spectroscopy *nuisance amplitudes* —
and never looked at ``Observation.line_fluxes``. A joint photometry + line-flux
fit therefore stayed on the exact path.

The cost is not marginal. Without the LUT, ``loss_functions`` sets
``needs_state=True`` and every likelihood evaluation reconstructs the full
~6000-wavelength SED via ``predict_state`` purely to obtain a handful of line
fluxes. Measured on a 5-band + 3-line model: **6.95 ms per gradient versus
0.31 ms** with the LUT — 21x, which at ~10,000 NUTS gradients is 70 s versus
3 s.

A second, separate hole: a model built with ``approx=WavePrecomp()`` hit the
"respect the build-time approx" branch and was returned untouched, so naming
WavePrecomp explicitly made a lines fit *slower than passing nothing at all*.
Under ``"auto"`` the build-time config is now topped up with FeaturePrecomp
rather than bailed on, and the existing config objects are preserved — dropping
a ``catalog_z_range`` while "helpfully" adding a LUT would be a silent
behavioral change.

Auto-activation is scoped to the ``"auto"`` policy on purpose. ``approx=None``
means exact and stays exact; an explicit config means what it says. Both warn
instead, so the cost is discoverable rather than silent.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import (
    FIXED,
    FREE,
    FeaturePrecomp,
    Fitter,
    Fixed,
    Observation,
    Photometry,
    SEDModel,
    Uniform,
    WavePrecomp,
)
from tengri.observation.line_flux_data import LineFluxData

pytestmark = pytest.mark.contract

_LINES = ("Halpha", "Hbeta", "OIII_5007")
_WAVES = jnp.array([6564.61, 4862.71, 5008.24])


def _model(ssp, obs, approx):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel.build(
            ssp_data=ssp,
            observation=obs,
            sfh={"type": "dpl", "all_params": FREE},
            dust_attenuation={
                "type": "two_component",
                "law": "calzetti",
                "all_params": FIXED,
                "tau_diff": Uniform(0.0, 2.0),
            },
            neb={"type": "none"},
            redshift=Fixed(0.1),
            approx=approx,
        )


@pytest.fixture
def joint_setup(synthetic_ssp_wide, synthetic_tophat_obs):
    """A photometry + line-flux observation, plus mock data at truth."""
    base = _model(synthetic_ssp_wide, synthetic_tophat_obs, WavePrecomp())
    truth = base.spec.sample(jax.random.PRNGKey(0))
    phot = np.asarray(base.predict_photometry(truth))
    lf = np.asarray(base.measure_line_fluxes(truth, approx=False))[:3]

    obs = Observation(
        photometry=synthetic_tophat_obs.photometry,
        line_fluxes=LineFluxData(
            names=_LINES,
            fluxes=jnp.asarray(lf),
            errors=jnp.asarray(np.abs(lf) * 0.05 + 1e-30),
            wavelengths=_WAVES,
        ),
    )
    return synthetic_ssp_wide, obs, phot


def _fit(ssp, obs, phot, **kw):
    """Build a Fitter. Model-construction noise is silenced; Fitter warnings are NOT.

    The suppression is deliberately scoped to ``_model``: an earlier version
    wrapped the whole thing in ``simplefilter("ignore")``, which swallowed the
    very warnings two of these tests exist to observe, and they failed with
    "DID NOT WARN" against a warning that was in fact raised.
    """
    model = _model(ssp, obs, kw.pop("build_approx", None))
    return Fitter(
        model,
        data=phot,
        noise=0.05 * np.abs(phot) + 1e-31,
        data_type="photometry",
        **kw,
    )


def _wave_cfg(model):
    """The active WavePrecomp config, off a SEDModel or a ForwardModel."""
    inner = getattr(model, "populations", None)
    sed = inner[0].sed if inner else model
    return sed._approx_config_wave


def test_line_flux_fit_gets_the_lut_with_no_build_time_approx(joint_setup):
    """The headline: fitting line fluxes must reach the LUT under the default."""
    fitter = _fit(*joint_setup)
    assert fitter.model.approx.feature_precomp, (
        "a line-flux fit did not get FeaturePrecomp under approx='auto' — every "
        "likelihood evaluation is reconstructing the full-grid SED"
    )


def test_build_time_waveprecomp_is_topped_up_not_bailed_on(joint_setup):
    """approx=WavePrecomp() at build time must not make a lines fit slower."""
    ssp, obs, phot = joint_setup
    fitter = _fit(ssp, obs, phot, build_approx=WavePrecomp())

    assert fitter.model.approx.wave_precomp, "the build-time WavePrecomp was dropped"
    assert fitter.model.approx.feature_precomp, (
        "a build-time WavePrecomp suppressed the line LUT — naming WavePrecomp "
        "explicitly made this fit slower than passing nothing at all"
    )


def test_topping_up_preserves_the_build_time_settings(joint_setup):
    """Adding the LUT must not quietly reset a configured WavePrecomp."""
    ssp, obs, phot = joint_setup
    fitter = _fit(ssp, obs, phot, build_approx=WavePrecomp(catalog_z_range=(0.0, 1.5)))

    assert fitter.model.approx.feature_precomp
    assert _wave_cfg(fitter.model).catalog_z_range == (0.0, 1.5), (
        "topping up FeaturePrecomp discarded the build-time catalog_z_range — a "
        "silent behavioral change, not a speedup"
    )


def _objective(ssp, obs, phot, approx):
    from tengri.inference.context import InferenceContext

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ctx = InferenceContext.from_target(_fit(ssp, obs, phot, approx=approx))
        x0 = ctx.initial_params(jax.random.PRNGKey(1))
        return float(ctx.neg_log_posterior_fn(x0, ctx.data_args))


def test_both_objectives_are_finite_on_the_synthetic_grid(joint_setup):
    """CI-visible structural check; the value comparison needs a real grid.

    The synthetic SSP is a smooth power law with **no emission lines baked in**,
    so a continuum-subtracted "line flux" off it is a near-zero residual, and
    the fitted errors (5% of ~0) make the line chi2 term degenerate. Comparing
    objectives there measures amplified noise, not the LUT: it reads a 1.5%
    shift where the real grid agrees to 5e-08. So this asserts only what the
    synthetic grid can support, and
    :func:`test_adding_the_line_lut_does_not_move_the_objective` carries the
    physics comparison on a real grid.
    """
    ssp, obs, phot = joint_setup
    v_lut = _objective(ssp, obs, phot, (WavePrecomp(), FeaturePrecomp()))
    v_no_lut = _objective(ssp, obs, phot, WavePrecomp())
    assert np.isfinite(v_lut) and np.isfinite(v_no_lut)


def test_adding_the_line_lut_does_not_move_the_objective(ssp_data_wne, real_ssp_only):
    """A default change must not move the posterior it is speeding up.

    Varies **one** thing: WavePrecomp with and without FeaturePrecomp.
    Comparing against ``approx=None`` instead would fold in WavePrecomp's own
    documented, pre-existing approximation and attribute it to a change that did
    not cause it.

    Measured on the wNE grid: relative difference **4.7e-08** with genuine
    line fluxes of order 1e-12 erg/s/cm^2. Data-gated, because that is the only
    configuration where the line channel carries physics rather than noise.
    """
    obs0 = Observation(photometry=Photometry.from_names(["des_g", "des_r", "des_i", "wise_w1"]))
    base = _model(ssp_data_wne, obs0, WavePrecomp())
    truth = base.spec.sample(jax.random.PRNGKey(0))
    phot = np.asarray(base.predict_photometry(truth))
    lf = np.asarray(base.measure_line_fluxes(truth, approx=False))[:3]
    assert np.all(np.abs(lf) > 1e-20), f"setup: the line channel must be real, got {lf}"

    obs = Observation(
        photometry=obs0.photometry,
        line_fluxes=LineFluxData(
            names=_LINES,
            fluxes=jnp.asarray(lf),
            errors=jnp.asarray(np.abs(lf) * 0.05 + 1e-30),
            wavelengths=_WAVES,
        ),
    )
    v_lut = _objective(ssp_data_wne, obs, phot, (WavePrecomp(), FeaturePrecomp()))
    v_no_lut = _objective(ssp_data_wne, obs, phot, WavePrecomp())

    assert abs(v_lut - v_no_lut) / abs(v_no_lut) < 1e-6, (
        f"adding FeaturePrecomp moved the objective: {v_lut} vs {v_no_lut}"
    )


def test_approx_none_stays_exact_and_warns(joint_setup):
    """approx=None means exact — auto-activation must not override an opt-out."""
    ssp, obs, phot = joint_setup
    with pytest.warns(UserWarning, match="FeaturePrecomp"):
        fitter = _fit(ssp, obs, phot, approx=None)
    assert not fitter.model.approx.feature_precomp


def test_an_explicit_config_is_respected_but_warns(joint_setup):
    """An explicit config means what it says; the cost is surfaced, not fixed."""
    ssp, obs, phot = joint_setup
    with pytest.warns(UserWarning, match="FeaturePrecomp"):
        fitter = _fit(ssp, obs, phot, approx=WavePrecomp())
    assert not fitter.model.approx.feature_precomp


def test_no_warning_when_the_lut_is_active(joint_setup):
    """The warning must not fire on the happy path, or it is noise.

    Scoped to *this* warning rather than ``simplefilter("error")``: building the
    model legitimately emits ``BakedInNebularWarning``, and turning every
    UserWarning into an error would fail on an unrelated, correct one.
    """
    ssp, obs, phot = joint_setup
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fitter = _fit(ssp, obs, phot, approx=(WavePrecomp(), FeaturePrecomp()))

    assert fitter.model.approx.feature_precomp
    offenders = [w for w in caught if "FeaturePrecomp" in str(w.message)]
    assert not offenders, f"warned despite the LUT being active: {offenders[0].message}"


def test_a_photometry_only_fit_is_untouched(synthetic_ssp_wide, synthetic_tophat_obs):
    """No line channel, no FeaturePrecomp — the default must stay scoped."""
    base = _model(synthetic_ssp_wide, synthetic_tophat_obs, WavePrecomp())
    phot = np.asarray(base.predict_photometry(base.spec.sample(jax.random.PRNGKey(0))))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitter = Fitter(
            _model(synthetic_ssp_wide, synthetic_tophat_obs, None),
            data=phot,
            noise=0.05 * np.abs(phot) + 1e-31,
            data_type="photometry",
        )
    assert not fitter.model.approx.feature_precomp
    assert fitter.model.approx.wave_precomp, "photometry should still get WavePrecomp"
