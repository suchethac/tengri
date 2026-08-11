# SPDX-License-Identifier: BSD-3-Clause
"""#1596 on REAL models, and in its build-time spelling.

Companion to ``test_issue_1596_photometry_feature_default.py``, which pins the
same policy with stubs. The division of labor is deliberate:

* **that file** answers "which config does the policy resolve", with no SSP
  grid, so the branch logic is pinned in the fast tier;
* **this file** resolves through a real ``SEDModel`` / ``ForwardModel`` and a
  real ``Fitter``, because the stub cannot see the delegation seam — reading
  ``_nebular_backend`` straight off a ``ForwardModel`` returns ``None``, and a
  policy keyed on it would answer "no" for every real fit without saying so.
  Both files passing is the evidence the stubs have not drifted from reality.

The defect: ``Fitter`` resolves ``approx="auto"``, and that policy appended
``FeaturePrecomp`` only when an emission-line channel was fit. A Cue model
fitted to broadband alone — the default shape of a catalog run — therefore ran
the Cue emulator inside every likelihood gradient. Gradient FLOPs from the
compiled HLO, one process per arm, persistent cache disabled:

    photometry + lines      5,021,451  ->  5,021,451   (already had it)
    photometry only        82,526,904  ->   3,737,260   (22x)

The trigger named a *channel* when the win belongs to the *backend*: for a
Cue-like backend the grid replaces the emulator call itself, so it pays with no
lines at all.

Not only speed. On the un-tabulated path the loss is **not smooth** in
``met_logzsol`` — autodiff and central differences disagree by 24%, and the
finite-difference estimate swings between -45.9 and -80.4 as the step shrinks.
Through the grid they agree to 7e-8, so this repairs a gradient every optimizer
and NUTS run was already trusting.

``#1656`` fixed that for a model carrying **no** build-time ``approx=``. It left
the other spelling, reported as **#1683**: a model built ``approx=WavePrecomp()``
was returned untouched by both the single-galaxy and the batch resolver, so
naming the wave LUT explicitly *cost* the feature LUT and fit **slower than
passing nothing at all** — the same pathology ``_add_feature_precomp`` already
documented for lines. That is what ``TestBuildTimeApproxStillGetsTheFeatureTopUp``
pins, on both surfaces; the batch half is the worse one, since population and
catalog fits are exactly those that evaluate the forward model the most.

Assertions are **structural, not timed**: a shared runner cannot measure this
(the same comparison read 13.5x and 15.0x on consecutive runs of one machine,
#1353).
"""

import numpy as np
import pytest

from tengri import FIXED, FREE, ForwardModel, Observation, SEDModel
from tengri.forward.sed_model import WavePrecomp
from tengri.inference.fitter import Fitter, _resolve_batch_fit_approx
from tengri.observation import LineRatioData

pytestmark = [pytest.mark.regression_bug]


def _dummy_data(obs):
    """Correctly-shaped photometry + errors.

    These tests inspect the *resolved approx policy*, never a fit result, so the
    values do not matter — only that the shape matches the observation.
    """
    n = obs.photometry.n_filters
    return np.ones(n), np.full(n, 0.1)


def _cue_model(ssp, obs, approx=None):
    """A Cue model. ``approx=None`` leaves the fit policy to supply the LUT."""
    return SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        sfh={"type": "dpl", "all_params": FREE},
        dust={"type": "none"},
        redshift=0.05,
        neb={"type": "cue", "all_params": FIXED},
        approx=approx,
    )


def _n_wave_precomp(model):
    """How many ``WavePrecomp`` configs the resolved model carries."""
    return sum(isinstance(c, WavePrecomp) for c in getattr(model, "approx_configs", ()))


class TestAutoPrecompForPhotometryOnlyCue:
    """No build-time ``approx=`` — the branch #1656 fixed, on real models."""

    def test_auto_policy_adds_feature_precomp_without_a_line_channel(
        self, ssp_data_fsps, synthetic_tophat_obs
    ):
        """LOAD-BEARING: the default fit must reach the grid with no lines fit.

        Neuter: delete the photometry-only top-up block in
        ``Fitter._resolve_fit_approx`` (the ``contextlib.suppress`` that appends
        ``FeaturePrecomp`` to ``cfg``) and this fails — the resolved state comes
        back ``wave_precomp=True`` with ``feature_precomp`` unset.
        """
        flux, err = _dummy_data(synthetic_tophat_obs)
        model = _cue_model(ssp_data_fsps, synthetic_tophat_obs)
        assert model.observation.line_fluxes is None, "fixture must have no line channel"

        fwd = ForwardModel.build(sed=model)
        fitter = Fitter(fwd, flux, err)  # no approx= : take the "auto" default
        assert fitter.model.approx.feature_precomp, (
            "the 'auto' fit policy left a photometry-only Cue fit without "
            "FeaturePrecomp, so every likelihood gradient re-runs the Cue emulator "
            "(22x the FLOPs, and a gradient that disagrees with finite differences "
            "by 24%) - #1596."
        )

    def test_explicit_approx_none_is_still_respected(self, ssp_data_fsps, synthetic_tophat_obs):
        """The policy may fill a gap; it must not override an explicit choice.

        Without this, the change above would be a silent accuracy change rather
        than a default a user can decline.
        """
        flux, err = _dummy_data(synthetic_tophat_obs)
        model = _cue_model(ssp_data_fsps, synthetic_tophat_obs)
        fwd = ForwardModel.build(sed=model)
        fitter = Fitter(fwd, flux, err, approx=None)
        assert not fitter.model.approx.feature_precomp
        assert not fitter.model.approx.wave_precomp

    def test_non_cue_backend_is_unaffected(self, ssp_data_wne, synthetic_tophat_obs):
        """The top-up is keyed on what the backend can tabulate, not on "any model".

        A baked-in / wNE model has its lines inside the SSP templates; the per-Q_H
        grid does not apply. The attempt raises, that raise IS the detection, and
        the model must keep the wave LUT rather than fall back to the raw path.
        """
        flux, err = _dummy_data(synthetic_tophat_obs)
        model = SEDModel.build(
            ssp_data=ssp_data_wne,
            observation=synthetic_tophat_obs,
            sfh={"type": "dpl", "all_params": FREE},
            dust={"type": "none"},
            redshift=0.05,
            approx=None,
        )
        fwd = ForwardModel.build(sed=model)
        fitter = Fitter(fwd, flux, err)
        assert not fitter.model.approx.feature_precomp
        assert fitter.model.approx.wave_precomp, "the failed top-up must not cost the wave LUT"

    def test_a_line_ratio_channel_is_excluded_from_the_widening(
        self, ssp_data_fsps, synthetic_tophat_obs
    ):
        """Ratios read the backend's DISCRETE catalog, which the grid replaces.

        The first version of this top-up turned FeaturePrecomp on for any Cue
        model, and a ratio fit that used to work started raising "Configured
        nebular backend did not publish a discrete line catalog" from
        ``predict_line_ratios`` (tests/contract/test_line_ratio_data.py).

        Neuter: drop ``_has_line_adjacent_channel`` from the photometry-only
        top-up conditions and that contract test fails again.
        """
        obs = Observation(
            photometry=synthetic_tophat_obs.photometry,
            line_ratios=LineRatioData.from_dict({("Halpha", "Hbeta"): (4.0, 0.3)}),
        )
        flux, err = _dummy_data(obs)
        model = _cue_model(ssp_data_fsps, obs)
        fwd = ForwardModel.build(sed=model)
        fitter = Fitter(fwd, flux, err)
        assert not fitter.model.approx.feature_precomp, (
            "a line-ratio fit must not be given FeaturePrecomp by default: the "
            "per-Q_H grid does not publish the discrete line catalog that "
            "predict_line_ratios reads."
        )


class TestBuildTimeApproxStillGetsTheFeatureTopUp:
    """``approx=WavePrecomp()`` at build time — the spelling #1656 left behind.

    Both resolvers took an early exit for a model that already carried the wave
    LUT, so the *more* explicit call produced the *slower* fit.
    """

    def test_single_galaxy_build_time_wave_precomp_is_topped_up(
        self, ssp_data_fsps, synthetic_tophat_obs
    ):
        """LOAD-BEARING: naming WavePrecomp must not cost FeaturePrecomp.

        Neuter: in ``Fitter._resolve_fit_approx``, drop the photometry-only
        ``_add_feature_precomp(..., warn_on_failure=False)`` call from the
        ``_has_modern_approx()`` branch and this fails.
        """
        flux, err = _dummy_data(synthetic_tophat_obs)
        model = _cue_model(ssp_data_fsps, synthetic_tophat_obs, approx=WavePrecomp())
        fwd = ForwardModel.build(sed=model)
        assert fwd.approx.wave_precomp, "precondition: the build-time LUT is present"

        fitter = Fitter(fwd, flux, err)
        assert fitter.model.approx.feature_precomp, (
            "a photometry-only Cue fit built with approx=WavePrecomp() was left "
            "without FeaturePrecomp, so naming the wave LUT explicitly ran the "
            "fit SLOWER than passing nothing at all - #1596."
        )
        assert fitter.model.approx.wave_precomp
        assert _n_wave_precomp(fitter.model) == 1, "the top-up must not re-append WavePrecomp"

    def test_single_galaxy_build_time_approx_with_a_ratio_channel_stays_wave_only(
        self, ssp_data_fsps, synthetic_tophat_obs
    ):
        """The ratio exclusion must hold on this branch too, not only the other."""
        obs = Observation(
            photometry=synthetic_tophat_obs.photometry,
            line_ratios=LineRatioData.from_dict({("Halpha", "Hbeta"): (4.0, 0.3)}),
        )
        flux, err = _dummy_data(obs)
        model = _cue_model(ssp_data_fsps, obs, approx=WavePrecomp())
        fwd = ForwardModel.build(sed=model)
        fitter = Fitter(fwd, flux, err)
        assert not fitter.model.approx.feature_precomp
        assert fitter.model.approx.wave_precomp

    def test_batch_surface_build_time_wave_precomp_is_topped_up(
        self, ssp_data_fsps, synthetic_tophat_obs
    ):
        """The batch mirror had the same early exit — and fixing one is not fixing both.

        Neuter: restore ``if state is not None and state.wave_precomp: return
        model`` ahead of the top-up in ``_resolve_batch_fit_approx``.
        """
        model = _cue_model(ssp_data_fsps, synthetic_tophat_obs, approx=WavePrecomp())
        resolved = _resolve_batch_fit_approx(model, "auto", "photometry")
        assert resolved.approx.feature_precomp, (
            "the batch resolver returned a build-time WavePrecomp model untouched, "
            "so population and catalog fits — the ones that evaluate the forward "
            "model the most — never reached the per-Q_H grid."
        )
        assert resolved.approx.wave_precomp
        assert _n_wave_precomp(resolved) == 1, "the top-up must not re-append WavePrecomp"
