# SPDX-License-Identifier: BSD-3-Clause
"""Regression contract: the 'exact forward path' fit warning must read the
``approx`` flag *through* the ForwardModel wrapper.

Fresh-user audit (2026-07): ``_warn_if_exact_forward_path`` read ``_approx``
straight off ``Fitter.model``. On the documented README path that model is a
``ForwardModel``, which does not carry ``_approx`` — the flag lives on the SED
it wraps — so the lookup silently returned ``{}`` and *every* wrapped model
looked like the exact path. A fit already routed through the WavePrecomp LUT
was told to "rebuild with approx=WavePrecomp() — or start from a recipe", which
the user had already done.

Two properties are pinned here:

* the warning stays **silent** when the LUT is active behind the wrapper (the
  false alarm), and
* it still **fires** when the exact path is genuinely in use, so the fix does
  not simply neuter the warning.

``Fitter.model`` is the model *after* ``_resolve_fit_approx``, so under the
default ``approx="auto"`` the fit is already on the LUT and the warning is
silent; it fires for ``Fitter(..., approx=None)``.
"""

from __future__ import annotations

import warnings

import pytest

from tengri import FIXED, Fixed, ForwardModel, SEDModel, WavePrecomp
from tengri.inference.fitter import _warn_if_exact_forward_path

pytestmark = [pytest.mark.contract, pytest.mark.regression_bug]

_MESSAGE_KEY = "exact forward path"


def _build(ssp, obs, approx):
    return SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        sfh={"type": "dpl", "all_params": FIXED},
        redshift=Fixed(0.1),
        approx=approx,
    )


def _warnings_from(model, backend="mcmc_nuts"):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _warn_if_exact_forward_path(model, backend)
    return [w for w in caught if _MESSAGE_KEY in str(w.message)]


def test_approx_accessor_sees_through_forward_model_wrapper(
    synthetic_ssp_wide, synthetic_tophat_obs
):
    """The wrapper does not carry ``_approx``; the public accessor delegates inward.

    The warning now reads ``model.approx`` — the accessor SEDModel and
    ForwardModel both implement — instead of the private ``_effective_approx``
    probe that #1218 introduced. Same fix, one public spelling.
    """
    sed = _build(synthetic_ssp_wide, synthetic_tophat_obs, WavePrecomp())
    fwd = ForwardModel.build(sed=sed, observation=synthetic_tophat_obs)

    assert sed._approx.get("wave_precomp") is True, "SED should carry the LUT flag"
    # the bug: a raw getattr on the wrapper finds nothing
    assert getattr(fwd, "_approx", None) is None
    # the fix: the accessor delegates to the wrapped SED
    assert fwd.approx.wave_precomp is True
    # and it must still work on a bare SEDModel
    assert sed.approx.wave_precomp is True


def test_no_false_warning_when_lut_active_behind_wrapper(synthetic_ssp_wide, synthetic_tophat_obs):
    """THE BUG: a LUT-backed model wrapped in ForwardModel must not be told to
    'rebuild with approx=WavePrecomp()' — it already has it."""
    sed = _build(synthetic_ssp_wide, synthetic_tophat_obs, WavePrecomp())
    fwd = ForwardModel.build(sed=sed, observation=synthetic_tophat_obs)
    assert _warnings_from(fwd) == [], "false 'exact forward path' warning on a LUT model"


def test_still_warns_when_exact_path_genuinely_in_use(synthetic_ssp_wide, synthetic_tophat_obs):
    """The fix must not neuter the warning: a genuinely exact model still warns."""
    sed = _build(synthetic_ssp_wide, synthetic_tophat_obs, None)
    fwd = ForwardModel.build(sed=sed, observation=synthetic_tophat_obs)
    assert len(_warnings_from(fwd)) == 1, "exact path should still warn"


def test_cheap_samplers_never_warn(synthetic_ssp_wide, synthetic_tophat_obs):
    """Only many-evaluation samplers warn; MAP does not."""
    sed = _build(synthetic_ssp_wide, synthetic_tophat_obs, None)
    fwd = ForwardModel.build(sed=sed, observation=synthetic_tophat_obs)
    assert _warnings_from(fwd, backend="map") == []


def test_message_does_not_claim_the_fictitious_100x(synthetic_ssp_wide, synthetic_tophat_obs):
    """The old text claimed '>100x for NUTS'. Measured gap is ~2-6x; quoting a
    fictitious number is how a warning loses its credibility."""
    sed = _build(synthetic_ssp_wide, synthetic_tophat_obs, None)
    fwd = ForwardModel.build(sed=sed, observation=synthetic_tophat_obs)
    msg = str(_warnings_from(fwd)[0].message)
    assert "100x" not in msg
