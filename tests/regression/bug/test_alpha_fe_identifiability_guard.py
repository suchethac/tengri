# SPDX-License-Identifier: BSD-3-Clause
"""A free ``met_alpha_fe`` that cannot be identified must say so (#1764, #1095).

On an SSP grid with no [alpha/Fe] axis, ``met_alpha_fe`` reaches the SED only
through ``effective_metallicity``, and only from the ``"delta"`` metallicity
branch. Two silent failures follow, and this module pins both plus the
measurement behind each:

* any non-delta metallicity mode never reads the parameter at all — sweeping it
  leaves the SED bit-identical and the gradient is exactly ``0.0`` (#1764);
* under delta it is a pure additive shift of ``met_logzsol``, so freeing both
  gives an exactly flat ridge (#1095).

The negative cases carry the weight. A guard that also fired on a pinned
parameter, or on a 4D alpha-enhanced grid (#226) where the axis is real, would
be filtered wholesale and stop protecting anything.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import Fixed, Parameters, SEDModel, Uniform
from tengri.components.stellar.sps.dsps_wrapper import (
    _ALPHA_TO_Z_COEFF,
    has_alpha_grid,
)
from tengri.config.exceptions import (
    DeadGradientParameterWarning,
    DegenerateParameterPairWarning,
    measurements_of,
)

pytestmark = pytest.mark.regression_bug

BASE = dict(
    sfh_dpl_alpha=Fixed(1.0),
    sfh_dpl_beta=Fixed(1.5),
    sfh_dpl_tau_gyr=Fixed(8.0),
    sfh_dpl_log_total_mass=Fixed(1.0),
    dust_tau_bc=Fixed(0.3),
    dust_tau_diff=Fixed(0.2),
    sfh_dpl_age_gyr=Fixed(13.0),
    redshift=Fixed(0.1),
    mean_sfh_type="dpl",
)


def _build(ssp, **kwargs):
    """Construct a model, returning it alongside every warning it raised."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = SEDModel(Parameters(**{**BASE, **kwargs}), ssp, precompute=False)
    return model, caught


def _of(caught, category):
    return [w for w in caught if issubclass(w.category, category)]


# ── the two guards fire ───────────────────────────────────────────


def test_non_delta_metallicity_reports_the_dead_gradient(ssp_data_fsps):
    """``met_alpha_fe`` under a ramp metallicity is never read (#1764)."""
    assert not has_alpha_grid(ssp_data_fsps), "fixture must be a 3D grid for this test"

    _, caught = _build(
        ssp_data_fsps,
        met_logzsol_0=Fixed(-1.0),
        met_logzsol_final=Fixed(-0.2),
        met_alpha_fe=Uniform(-0.2, 0.6),
    )

    dead = _of(caught, DeadGradientParameterWarning)
    assert dead, (
        "a free met_alpha_fe under a non-delta metallicity mode produced no "
        f"warning; got {[w.category.__name__ for w in caught]}"
    )
    text = str(dead[0].message)
    assert "met_alpha_fe" in text
    # The message must name the mode it is complaining about, not just the symptom.
    assert "ramp" in text, f"warning did not name the metallicity mode: {text}"
    assert measurements_of(dead[0].message).get("gradient") == 0.0


def test_delta_metallicity_reports_the_degenerate_pair(ssp_data_fsps):
    """Freeing ``met_alpha_fe`` with ``met_logzsol`` is an exact ridge (#1095)."""
    _, caught = _build(
        ssp_data_fsps,
        met_logzsol=Uniform(-2.0, 0.5),
        met_alpha_fe=Uniform(-0.2, 0.6),
    )

    degenerate = _of(caught, DegenerateParameterPairWarning)
    assert degenerate, (
        "a free met_alpha_fe + met_logzsol pair produced no warning; got "
        f"{[w.category.__name__ for w in caught]}"
    )
    text = str(degenerate[0].message)
    assert "met_logzsol" in text and "met_alpha_fe" in text
    # The coefficient is what makes the ridge predictable, so it must ride along.
    assert measurements_of(degenerate[0].message).get("coefficient") == _ALPHA_TO_Z_COEFF


# ── and stay silent everywhere else ───────────────────────────────


def test_silent_when_alpha_fe_is_pinned(ssp_data_fsps):
    """The default ``Fixed(0.0)`` must not warn — it is the common case."""
    _, caught = _build(ssp_data_fsps, met_logzsol=Uniform(-2.0, 0.5))

    assert not _of(caught, DeadGradientParameterWarning)
    assert not _of(caught, DegenerateParameterPairWarning)


def test_silent_when_only_alpha_fe_is_free(ssp_data_fsps):
    """Alpha alone under delta is a reparameterization, not a degeneracy.

    With ``met_logzsol`` pinned there is exactly one free metallicity direction,
    so the pair-degeneracy warning must not fire: the fit is well-posed even
    though the parameter is really acting as a metallicity shift.
    """
    _, caught = _build(
        ssp_data_fsps,
        met_logzsol=Fixed(-0.3),
        met_alpha_fe=Uniform(-0.2, 0.6),
    )

    assert not _of(caught, DegenerateParameterPairWarning)
    assert not _of(caught, DeadGradientParameterWarning)


# ── the measurements the guards encode ────────────────────────────


def test_non_delta_gradient_is_exactly_zero(ssp_data_fsps):
    """The #1764 claim itself: sweeping alpha under a ramp moves nothing."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SEDModel(
            Parameters(
                **BASE,
                met_logzsol_0=Fixed(-1.0),
                met_logzsol_final=Fixed(-0.2),
                met_alpha_fe=Uniform(-0.2, 0.6),
            ),
            ssp_data_fsps,
            precompute=False,
        )

    sed_lo = np.asarray(model.predict({"met_alpha_fe": 0.0}).rest_sed())
    sed_hi = np.asarray(model.predict({"met_alpha_fe": 0.6}).rest_sed())
    assert np.array_equal(sed_lo, sed_hi), "ramp metallicity unexpectedly reads met_alpha_fe"

    grad = float(jax.grad(lambda a: jnp.sum(model.predict({"met_alpha_fe": a}).rest_sed()))(0.3))
    assert grad == 0.0, f"expected an identically dead gradient, got {grad!r}"


def test_delta_degeneracy_is_bit_exact(ssp_data_fsps):
    """The #1095 claim: a compensating shift reproduces the SED exactly."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SEDModel(
            Parameters(
                **BASE,
                met_logzsol=Uniform(-2.0, 0.5),
                met_alpha_fe=Uniform(-0.2, 0.6),
            ),
            ssp_data_fsps,
            precompute=False,
        )

    logzsol, alpha = -0.3, 0.6
    enhanced = np.asarray(
        model.predict({"met_logzsol": logzsol, "met_alpha_fe": alpha}).rest_sed()
    )
    shifted = np.asarray(
        model.predict(
            {"met_logzsol": logzsol + _ALPHA_TO_Z_COEFF * alpha, "met_alpha_fe": 0.0}
        ).rest_sed()
    )
    assert np.array_equal(enhanced, shifted), (
        "alpha enhancement is not bit-identical to the compensating metallicity "
        "shift; the folding relation in effective_metallicity has changed"
    )

    # Control: without the compensation the SED must move, so the assertion
    # above is testing a degeneracy rather than an insensitive model.
    uncompensated = np.asarray(
        model.predict({"met_logzsol": logzsol + 0.10, "met_alpha_fe": alpha}).rest_sed()
    )
    assert not np.array_equal(enhanced, uncompensated)
