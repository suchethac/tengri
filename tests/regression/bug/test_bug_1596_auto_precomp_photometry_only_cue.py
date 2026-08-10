# SPDX-License-Identifier: BSD-3-Clause
"""The ``"auto"`` fit policy must give a photometry-only Cue fit FeaturePrecomp (#1596).

#1353 pinned that **explicitly** passing ``approx=(WavePrecomp(), FeaturePrecomp())``
reaches Cue's per-Q_H grid for a photometry-only model. It never pinned the case a user
actually hits — passing nothing at all. ``Fitter`` then resolves ``approx="auto"``, and
that policy appended ``FeaturePrecomp`` only when an emission-line channel was fit.

So a Cue model fitted to broadband alone — the default shape of a catalog run — ran the
Cue emulator inside every likelihood gradient. Gradient FLOPs from the compiled HLO,
one process per arm, persistent cache disabled:

    photometry + lines      5,021,451  ->  5,021,451   (already had it)
    photometry only        82,526,904  ->   3,737,260   (22x)

The trigger named a *channel* when the win belongs to the *backend*: for a Cue-like
backend the grid replaces the emulator call itself, so it pays with no lines at all.

Not only speed. On the un-tabulated path the loss is **not smooth** in ``met_logzsol``
— autodiff and central differences disagree by 24%, and the finite-difference estimate
swings between -45.9 and -80.4 as the step shrinks. Through the grid they agree to
7e-8, so this repairs a gradient every optimizer and NUTS run was already trusting.

Assertions are **structural, not timed**: a shared runner cannot measure this (the same
comparison read 13.5x and 15.0x on consecutive runs of one machine, #1353).
"""

import numpy as np
import pytest

from tengri import FIXED, FREE, ForwardModel, Observation, SEDModel
from tengri.inference.fitter import Fitter
from tengri.observation import LineRatioData

pytestmark = [pytest.mark.regression_bug]


def _dummy_data(obs):
    """Correctly-shaped photometry + errors.

    These tests inspect the *resolved approx policy*, never a fit result, so the
    values do not matter — only that the shape matches the observation.
    """
    n = obs.photometry.n_filters
    return np.ones(n), np.full(n, 0.1)


def _cue_model(ssp, obs):
    """A Cue model built with NO approx= — the fit policy must supply it."""
    return SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        sfh={"type": "dpl", "all_params": FREE},
        dust={"type": "none"},
        redshift=0.05,
        neb={"type": "cue", "all_params": FIXED},
        approx=None,
    )


class TestAutoPrecompForPhotometryOnlyCue:
    def test_auto_policy_adds_feature_precomp_without_a_line_channel(
        self, ssp_data_fsps, synthetic_tophat_obs
    ):
        """LOAD-BEARING: the default fit must reach the grid with no lines fit.

        Neuter: restore ``self._fits_lines(model)`` as the sole condition in
        ``Fitter._wants_feature_precomp`` and this fails — the resolved state comes
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
        """The widening is keyed on the Cue-like backend, not on "any model".

        A baked-in / wNE model has its lines inside the SSP templates; the per-Q_H
        grid does not apply, and the policy must leave it exactly as it was.
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

    def test_a_line_ratio_channel_is_excluded_from_the_widening(
        self, ssp_data_fsps, synthetic_tophat_obs
    ):
        """Ratios read the backend's DISCRETE catalog, which the grid replaces.

        The first version of this widening turned FeaturePrecomp on for any Cue
        model, and a ratio fit that used to work started raising "Configured
        nebular backend did not publish a discrete line catalog" from
        ``predict_line_ratios`` (tests/contract/test_line_ratio_data.py).

        Neuter: drop the ratio/index early-return in ``_wants_feature_precomp``
        and that contract test fails again.
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
