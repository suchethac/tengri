# SPDX-License-Identifier: BSD-3-Clause
"""Regression tests for #1222 — the approximation state must be observable.

``Fitter`` routes photometry fits onto the WavePrecomp LUT by default
(``approx="auto"``), then warned that they were on the exact path anyway. The
guard probed ``getattr(model, "_approx", None) or {}``; ``_approx`` is a lowered
flag dict that lives on ``SEDModel``, so on the canonical ``Fitter(forward, ...)``
path the probe found nothing, read that as "exact", and warned unconditionally —
carrying no information either way.

The fix is an accessor, not a corrected probe: ``model.approx`` answers the same
question with the same spelling on ``SEDModel`` and ``ForwardModel`` alike, so a
future guard cannot re-invent a private probe that silently reports "exact".

Design: docs/internal/specs/2026-07-17-consolidate-approx-and-bootstrap-design.md
"""

import warnings

import numpy as np
import pytest

from tengri import FIXED, Fitter, Fixed, ForwardModel, SEDModel, Uniform, WavePrecomp
from tengri.forward.sed_model import ApproxState
from tengri.inference.fitter import _warn_if_exact_forward_path

pytestmark = pytest.mark.regression_bug


def _sed(ssp, obs, approx=None):
    """A small 2-free-parameter photometry model at fixed redshift."""
    return SEDModel.build(
        ssp,
        observation=obs,
        sfh={"type": "delayed", "*": FIXED},
        dust_attenuation={
            "law": "power_law",
            "type": "two_component",
            "*": FIXED,
            "tau_diff": Uniform(0.0, 1.5),
            "tau_bc": Uniform(0.0, 1.0),
        },
        dust_emission=None,
        neb={"type": "none"},
        redshift=Fixed(0.05),
        approx=approx,
    )


def _warned(model) -> bool:
    """Whether the exact-path guard fires for a many-evaluation sampler."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _warn_if_exact_forward_path(model, "mcmc_nuts")
    return any("exact forward path" in str(w.message) for w in caught)


def _fitter(fwd, **kwargs):
    n_bands = fwd.observation.photometry.n_filters
    return Fitter(
        fwd,
        data=np.full(n_bands, 1e-28),
        noise=np.full(n_bands, 1e-29),
        data_type="photometry",
        **kwargs,
    )


# ── the accessor exists on both model types and agrees ───────────────────


def test_forward_model_approx_matches_inner_sed(synthetic_ssp_wide, synthetic_tophat_obs):
    """ForwardModel.approx delegates: the wrapper reports the inner SED's state.

    The bug: the wrapper silently answered "exact" for a LUT-equipped model
    because the state was read off an inner attribute it does not carry.
    """
    for approx in (None, WavePrecomp()):
        sed = _sed(synthetic_ssp_wide, synthetic_tophat_obs, approx=approx)
        fwd = ForwardModel.build(sed=sed, observation=synthetic_tophat_obs)
        assert fwd.approx == sed.approx, (
            f"ForwardModel.approx ({fwd.approx}) disagrees with the SEDModel it "
            f"wraps ({sed.approx}) for approx={approx!r}"
        )


def test_approx_reports_wave_precomp(synthetic_ssp_wide, synthetic_tophat_obs):
    """The accessor distinguishes a LUT model from an exact one, on both types."""
    exact = _sed(synthetic_ssp_wide, synthetic_tophat_obs)
    lut = _sed(synthetic_ssp_wide, synthetic_tophat_obs, approx=WavePrecomp())

    assert not exact.approx.wave_precomp
    assert lut.approx.wave_precomp
    assert not exact.approx, "an exact model's ApproxState must be falsy"
    assert lut.approx, "a WavePrecomp model's ApproxState must be truthy"

    for model in (exact, lut):
        fwd = ForwardModel.build(sed=model, observation=synthetic_tophat_obs)
        assert fwd.approx.wave_precomp == model.approx.wave_precomp


def test_approx_state_is_frozen():
    """ApproxState is a read-only view — mutating it cannot desync from the model."""
    state = ApproxState(wave_precomp=True)
    with pytest.raises((AttributeError, TypeError)):
        state.wave_precomp = False  # type: ignore[misc]


# ── the guard tells the truth in both directions ─────────────────────────


def test_no_warning_when_fit_is_on_the_lut(synthetic_ssp_wide, synthetic_tophat_obs):
    """The bug, pinned: a default fit is ON the LUT and must not warn.

    ``approx="auto"`` clones the model onto WavePrecomp inside ``Fitter``, so
    warning here tells the user to do the thing the Fitter already did.
    """
    sed = _sed(synthetic_ssp_wide, synthetic_tophat_obs)  # no build-time approx
    fwd = ForwardModel.build(sed=sed, observation=synthetic_tophat_obs)
    fitter = _fitter(fwd)  # approx="auto" is the default

    assert fitter.model.approx.wave_precomp, (
        "precondition failed: approx='auto' did not route this fit onto the LUT, "
        "so this test would pass for the wrong reason"
    )
    assert not _warned(fitter.model), (
        "warned that the fit is on the exact path while the LUT is live on the "
        "very model the Fitter is about to sample (#1222)"
    )


def test_warning_still_fires_on_a_genuinely_exact_fit(synthetic_ssp_wide, synthetic_tophat_obs):
    """The complement: ``approx=None`` forces the exact path and must still warn.

    Without this, "fix the false positive" is indistinguishable from "delete the
    guard".
    """
    sed = _sed(synthetic_ssp_wide, synthetic_tophat_obs)
    fwd = ForwardModel.build(sed=sed, observation=synthetic_tophat_obs)
    fitter = _fitter(fwd, approx=None)

    assert not fitter.model.approx.wave_precomp, (
        "precondition failed: approx=None left a LUT on the fit model"
    )
    assert _warned(fitter.model), "a genuinely exact photometry fit must still warn"


def test_guard_verdict_tracks_the_accessor(synthetic_ssp_wide, synthetic_tophat_obs):
    """The guard's verdict is a function of ``model.approx`` — never of a probe.

    Pins the invariant rather than the two cases above: whatever the accessor
    reports, the warning is exactly its negation. A guard reading anything else
    can drift away from what the forward pipeline actually does.
    """
    sed = _sed(synthetic_ssp_wide, synthetic_tophat_obs)
    fwd = ForwardModel.build(sed=sed, observation=synthetic_tophat_obs)

    for approx in ("auto", None, WavePrecomp()):
        fitter = _fitter(fwd, approx=approx)
        on_lut = fitter.model.approx.wave_precomp
        assert _warned(fitter.model) is (not on_lut), (
            f"approx={approx!r}: model.approx.wave_precomp={on_lut} but the guard "
            f"{'warned' if not on_lut else 'stayed silent'} — verdict and state disagree"
        )
