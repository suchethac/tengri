#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""A free ``met_alpha_fe`` must move the model, and move it differentiably.

``met_alpha_fe`` is opt-in and has two implementations: bilinear interpolation
across a 4D SSP grid whose fourth axis is [alpha/Fe], and, on an ordinary 3D
grid, the effective-metallicity fallback
``log_z_eff = met_logzsol + 0.75 * met_alpha_fe``. No hosted SSP currently
carries the 4D axis -- every one reports ``ssp_alpha_fe is None`` -- so what
these tests cover is the fallback, and one of them says so out loud rather
than letting a future 4D grid quietly change which path is under test.

Three things can go wrong here and none of them raises:

1. the parameter is **inert**: declared free, accepted by the builder, and
   never read, so a fit samples it against a flat likelihood and reports a
   posterior that is just the prior;
2. the parameter moves the prediction but carries **no usable gradient**, so
   NUTS, HMC and MAP cannot move it even though a grid search could. A
   ``grad != 0`` assertion is not enough to catch the near miss -- a wrong
   gradient is also nonzero -- so the gradient is checked against a central
   finite difference of the same function;
3. ``Fixed(x)`` and a free parameter evaluated at ``x`` **disagree**, which
   would mean the pinned and sampled paths are different models. The docs
   describe the correction as applying when the parameter is "declared free",
   and the reader of ``params.get("met_alpha_fe", default)`` in
   ``components/stellar/component.py`` has to assume a ``Fixed`` value
   actually reaches ``params``; measured, it does, and the two paths are
   bit-identical.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tengri
from tengri import DEFAULT, SEDModel, Uniform
from tengri.parameters import Fixed

pytestmark = pytest.mark.contract

ALPHA_LO = -0.2
ALPHA_HI = 0.6
PROBE = 0.25


@pytest.fixture(scope="module")
def observation():
    return tengri.Observation(photometry=tengri.Photometry.from_names(["sdss_g", "sdss_r"]))


def _model(ssp, observation, alpha_spec):
    """One model in which [alpha/Fe] is the only thing that varies."""
    return SEDModel.build(
        ssp_data=ssp,
        observation=observation,
        sfh={
            "type": "delayed",
            "all_params": Fixed(DEFAULT),
            "tau_gyr": Fixed(2.0),
            "age_gyr": Fixed(3.0),
            "log_total_mass": Fixed(10.0),
        },
        met={"logzsol": Fixed(-0.3), "alpha_fe": alpha_spec},
        redshift=Fixed(0.1),
        neb={"type": "ssp"},
    )


def test_the_fallback_path_is_what_these_tests_cover(ssp_data_wne):
    """State the path under test, so a 4D grid cannot silently replace it.

    If this ever fails, a grid with a real [alpha/Fe] axis has arrived and the
    tests below have quietly changed meaning: they would then be exercising
    the bilinear interpolation rather than the effective-metallicity
    correction. That is the moment to add the 4D cases, not to relax this.
    """
    assert getattr(ssp_data_wne, "ssp_alpha_fe", None) is None, (
        "this SSP carries a 4D [alpha/Fe] axis; these tests were written "
        "against the effective-metallicity fallback and now cover the "
        "interpolated path instead -- add the 4D cases"
    )


def test_a_free_alpha_fe_is_declared_free(ssp_data_wne, observation):
    model = _model(ssp_data_wne, observation, Uniform(ALPHA_LO, ALPHA_HI))
    assert "met_alpha_fe" in model.spec.free_params


def test_a_free_alpha_fe_is_not_inert(ssp_data_wne, observation):
    """A parameter that changes nothing gives back its own prior."""
    model = _model(ssp_data_wne, observation, Uniform(ALPHA_LO, ALPHA_HI))
    lo = np.asarray(model.predict_photometry({"met_alpha_fe": ALPHA_LO}), dtype=np.float64)
    hi = np.asarray(model.predict_photometry({"met_alpha_fe": ALPHA_HI}), dtype=np.float64)

    relative = np.max(np.abs(hi - lo) / np.abs(lo))
    assert relative > 1e-3, (
        f"[alpha/Fe] moved across its whole prior and the photometry changed by "
        f"{relative:.3e}; the parameter is inert and a fit would return the prior"
    )


def test_the_alpha_fe_gradient_agrees_with_a_finite_difference(ssp_data_wne, observation):
    """Nonzero is not enough: a wrong gradient is nonzero too.

    Gradient-based inference needs the derivative to be right, not merely
    present, so it is measured against a central difference of the same
    function rather than against zero.
    """
    model = _model(ssp_data_wne, observation, Uniform(ALPHA_LO, ALPHA_HI))

    def total(alpha):
        return jnp.sum(model.predict_photometry({"met_alpha_fe": alpha}))

    analytic = float(jax.grad(total)(PROBE))
    step = 0.01
    numeric = (float(total(PROBE + step)) - float(total(PROBE - step))) / (2.0 * step)

    assert np.isfinite(analytic), "the [alpha/Fe] gradient is not finite"
    assert analytic != 0.0, (
        "the [alpha/Fe] gradient is identically zero: the parameter is severed from "
        "the photometry, and a sampler could not move it"
    )
    assert abs(numeric) > 0.0, "the finite difference is zero, so this check is vacuous"

    # Compare the RATIO, not the difference. These fluxes are ~1e-27, and both
    # derivatives are ~1e-28, while ``pytest.approx`` keeps a default
    # ``abs=1e-12`` even when ``rel=`` is given. Written as
    # ``analytic == approx(numeric, rel=1e-3)`` the absolute floor swallows the
    # whole comparison: a severed gradient of exactly 0.0 differs from the
    # finite difference by 8.6e-28, which is under 1e-12, so it passes. Checked
    # by mutation -- wrapping the parameter in ``jax.lax.stop_gradient`` leaves
    # the predictions untouched and is invisible to every other test here, so
    # this assertion is the only thing standing between a non-differentiable
    # parameter and a sampler that cannot move it.
    ratio = analytic / numeric
    assert ratio == pytest.approx(1.0, rel=1e-3), (
        f"autodiff gradient {analytic:.6e} disagrees with the central finite "
        f"difference {numeric:.6e} (ratio {ratio:.6f}); a sampler would be led "
        "by the wrong slope"
    )


def test_fixed_alpha_fe_matches_the_free_model_at_the_same_value(ssp_data_wne, observation):
    """The pinned and sampled paths must be one model, not two."""
    pinned = _model(ssp_data_wne, observation, Fixed(PROBE))
    free = _model(ssp_data_wne, observation, Uniform(ALPHA_LO, ALPHA_HI))

    assert "met_alpha_fe" not in pinned.spec.free_params

    from_pinned = np.asarray(pinned.predict_photometry({}), dtype=np.float64)
    from_free = np.asarray(free.predict_photometry({"met_alpha_fe": PROBE}), dtype=np.float64)
    np.testing.assert_allclose(
        from_pinned,
        from_free,
        rtol=0.0,
        atol=0.0,
        err_msg=(
            "Fixed(alpha) and a free alpha evaluated at the same value give "
            "different photometry, so pinning it selects a different model"
        ),
    )


def test_a_fixed_alpha_fe_is_not_silently_discarded(ssp_data_wne, observation):
    """``params.get(name, default)`` drops a Fixed value that never lands.

    A ``Fixed`` parameter is legitimately absent from ``params``, so a reader
    written as ``params.get("met_alpha_fe", DEFAULT)`` returns the registry
    default and the user's pinned value disappears with nothing raised. Two
    different pinned values must give two different models.
    """
    zero = np.asarray(
        _model(ssp_data_wne, observation, Fixed(0.0)).predict_photometry({}),
        dtype=np.float64,
    )
    enhanced = np.asarray(
        _model(ssp_data_wne, observation, Fixed(ALPHA_HI)).predict_photometry({}),
        dtype=np.float64,
    )

    relative = np.max(np.abs(enhanced - zero) / np.abs(zero))
    assert relative > 1e-3, (
        f"Fixed(0.0) and Fixed({ALPHA_HI}) give photometry agreeing to "
        f"{relative:.3e}; the pinned value is being dropped before it reaches "
        "the stellar component"
    )
