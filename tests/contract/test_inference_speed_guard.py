# SPDX-License-Identifier: BSD-3-Clause
"""A many-evaluation sampler on the exact forward path warns the user.

Fresh-user audit (2026-07): running an MCMC/nested sampler on a hand-built
model *without* ``approx=WavePrecomp()`` is catastrophically slow — the sampler
evaluates the forward model thousands of times, and the exact wave-grid
photometry path is >100x slower per call than the WavePrecomp look-up table
(a small NUTS fit that finishes in ~15 s with WavePrecomp did not finish in
420 s without it). The recipes enable WavePrecomp, but a hand-built model may
not, so :func:`~tengri.inference.fitter._warn_if_exact_forward_path` nudges the
user to the fast path instead of letting the fit crawl.
"""

from __future__ import annotations

import warnings

import pytest

from tengri import FIXED, FREE, Fixed, ForwardModel, SEDModel, WavePrecomp
from tengri.inference.fitter import _warn_if_exact_forward_path

pytestmark = [pytest.mark.contract, pytest.mark.regression_bug]


def _model(ssp, obs, approx):
    return SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        redshift=Fixed(0.05),
        sfh={"type": "dpl", "all_params": FREE},
        dust={"type": "two_component", "all_params": FIXED},
        approx=approx,
    )


def _warned(model, backend):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _warn_if_exact_forward_path(model, backend)
    return any("exact forward path" in str(x.message) for x in w)


def test_exact_path_warns_for_mcmc(synthetic_ssp_wide, synthetic_tophat_obs):
    model = _model(synthetic_ssp_wide, synthetic_tophat_obs, approx=None)
    assert _warned(model, "mcmc_nuts")
    assert _warned(model, "mcmc_hmc")
    assert _warned(model, "nss")


def test_wave_precomp_does_not_warn(synthetic_ssp_wide, synthetic_tophat_obs):
    model = _model(synthetic_ssp_wide, synthetic_tophat_obs, approx=WavePrecomp())
    assert not _warned(model, "mcmc_nuts")


def test_cheap_backends_never_warn(synthetic_ssp_wide, synthetic_tophat_obs):
    """MAP / Laplace / Pathfinder / raytrace do few evals — no nudge even on the
    exact path (they were fast on it in the audit)."""
    model = _model(synthetic_ssp_wide, synthetic_tophat_obs, approx=None)
    for backend in ("map", "laplace", "pathfinder", "mcmc_raytrace", "vi"):
        assert not _warned(model, backend), backend


# ── The canonical ForwardModel path (#1222) ───────────────────────────
# The tests above only ever pass a bare SEDModel — the *deprecated* Fitter
# target. That gap is why #1222 shipped: ``ForwardModel`` carries no ``_approx``
# of its own, so the guard's ``getattr(model, "_approx", None) or {}`` probe read
# {} and warned unconditionally on the path the README actually teaches.


def _forward(ssp, obs, approx):
    return ForwardModel.build(sed=_model(ssp, obs, approx), observation=obs)


def test_forward_model_delegates_approx(synthetic_ssp_wide, synthetic_tophat_obs):
    """``ForwardModel._approx`` reports the inner SED's LUT flags (#1222)."""
    sed = _model(synthetic_ssp_wide, synthetic_tophat_obs, approx=WavePrecomp())
    forward = ForwardModel.build(sed=sed, observation=synthetic_tophat_obs)
    assert forward._approx == sed._approx
    assert forward._approx.get("wave_precomp") is True

    exact = _model(synthetic_ssp_wide, synthetic_tophat_obs, approx=None)
    assert not ForwardModel.build(sed=exact, observation=synthetic_tophat_obs)._approx.get(
        "wave_precomp", False
    )


def test_forward_model_with_wave_precomp_does_not_warn(synthetic_ssp_wide, synthetic_tophat_obs):
    """The canonical Fitter target must not be told to enable what is already on (#1222)."""
    forward = _forward(synthetic_ssp_wide, synthetic_tophat_obs, approx=WavePrecomp())
    assert forward._has_modern_approx()  # precondition: the LUT really is live
    assert not _warned(forward, "mcmc_nuts")


def test_forward_model_on_exact_path_still_warns(synthetic_ssp_wide, synthetic_tophat_obs):
    """The guard is corrected, not neutered: a genuinely exact forward still nudges."""
    forward = _forward(synthetic_ssp_wide, synthetic_tophat_obs, approx=None)
    assert not forward._has_modern_approx()  # precondition: really on the exact path
    assert _warned(forward, "mcmc_nuts")
