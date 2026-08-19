# SPDX-License-Identifier: BSD-3-Clause
r"""Regression: a ``Fixed`` redshift was silently dropped by the exact projectors.

Found while building API Phase 4 (#1048); the defect shipped in Phase 2 (#1097).

**The bug.** Fixed parameters are not required at predict time — that is what
fixing them *means* — so a user's params dict legitimately omits ``redshift``.
The forward model resolved it internally when building the SED, but
:func:`~tengri.observation.photometry.project_photometry` took the luminosity
distance from the **dict**::

    z = jnp.asarray(params.get("redshift", 0.0))  # <- absent => 0.0

With ``redshift=Fixed(0.5)``, that silently computed the flux at :math:`d_L(0)`
= **10 pc** instead of at z = 0.5 — about **16 orders of magnitude** wrong, with
no warning. The lean ``predict_photometry`` (the likelihood's path) merged the
fixed params and was correct, so the exploration surface and the fitting surface
disagreed by 1e16 and nothing noticed.

**Why it survived review.** Every existing test passed ``redshift`` explicitly in
the params dict. Fixing the redshift — the single most natural thing an
astronomer does, because you usually *know* the redshift — was never exercised
against ``pred.photometry()``.

The fix resolves fixed numeric values into ``Prediction._params`` once, so every
consumer of that dict (photometry, magnitudes, spectrum, obs_sed, the property
catalog, ``tengri.measure.from_prediction``) sees the true redshift.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from tengri import FIXED, Fixed, SEDModel, Uniform

pytestmark = pytest.mark.regression_bug


@pytest.fixture
def model_fixed_z(synthetic_ssp_wide, synthetic_tophat_obs):
    """A model with the redshift FIXED — so params legitimately omit it."""
    return SEDModel.build(
        ssp_data=synthetic_ssp_wide,
        observation=synthetic_tophat_obs,
        sfh={"type": "dpl", "*": FIXED, "log_total_mass": Uniform(9.0, 11.0)},
        dust={"law_diff": 'calzetti', "type": "two_component", "law_bc": "calzetti", "*": FIXED},
        neb={"type": "none"},
        redshift=Fixed(0.5),
    )


def test_prediction_photometry_honors_a_fixed_redshift(model_fixed_z):
    """``pred.photometry()`` must agree with the likelihood's ``predict_photometry``.

    These are the exploration surface and the fitting surface. If they disagree,
    one of them is lying about the galaxy — and here they disagreed by ~1e16.
    """
    params = {"sfh_dpl_log_total_mass": jnp.asarray(10.0)}  # no redshift: it's Fixed

    exploration = np.asarray(model_fixed_z.predict(params).photometry())
    fitting = np.asarray(model_fixed_z.predict_photometry(params))

    assert np.all(np.isfinite(exploration))
    # Compare RATIOS: these fluxes are ~1e-14, where a default atol=1e-8 would
    # make np.allclose vacuously true.
    np.testing.assert_allclose(exploration / fitting, 1.0, rtol=1e-10)


def test_fixed_redshift_flux_is_not_the_10pc_answer(model_fixed_z):
    """The specific failure: falling back to z=0 gives the 10 pc (absolute-mag) flux.

    Pin the magnitude of the error so a regression cannot hide inside a loose
    tolerance — the bug moved the answer by ~16 orders of magnitude, not by a few
    percent.
    """
    params = {"sfh_dpl_log_total_mass": jnp.asarray(10.0)}
    got = np.asarray(model_fixed_z.predict(params).photometry())

    # What the bug produced: the same SED integrated at d_L(z=0) = 10 pc.
    at_10pc = np.asarray(
        model_fixed_z.predict({**params, "redshift": jnp.asarray(0.0)}).photometry()
    )

    # VACUITY GUARD: the two must be wildly different, else this proves nothing.
    ratio = at_10pc / got
    assert np.all(ratio > 1e6), (
        "z=0 and z=0.5 give comparable fluxes here, so this test cannot detect "
        "the dropped-redshift bug — pick a redshift further from 0"
    )

    # And the real answer must be the z=0.5 one.
    expected = np.asarray(model_fixed_z.predict_photometry(params))
    np.testing.assert_allclose(got / expected, 1.0, rtol=1e-10)


def test_explicit_params_still_win_over_the_fixed_value(model_fixed_z):
    """Resolving fixed values must never clobber a value the user passed."""
    params = {"sfh_dpl_log_total_mass": jnp.asarray(10.0), "redshift": jnp.asarray(0.0)}
    pred = model_fixed_z.predict(params)

    assert float(pred._params["redshift"]) == 0.0


def test_measure_from_prediction_inherits_the_fixed_redshift(model_fixed_z):
    """Phase 3's façade reads ``pred._params`` — it inherited the same bug."""
    from tengri import measure

    params = {"sfh_dpl_log_total_mass": jnp.asarray(10.0)}
    pred = model_fixed_z.predict(params)

    out = measure.from_prediction(pred, filters=None)
    np.testing.assert_allclose(
        np.asarray(out["photometry"]) / np.asarray(model_fixed_z.predict_photometry(params)),
        1.0,
        rtol=1e-10,
    )


def test_measure_line_fluxes_honors_a_fixed_redshift(model_fixed_z):
    """The same bug, at a boundary the original fix missed (#1127).

    ``measure_line_fluxes`` is public, takes a raw user params dict, and does not
    route through ``Prediction`` — so resolving fixed values inside
    ``Prediction.__init__`` did nothing for it. It read the redshift out of the
    dict and converted L to F through :math:`4\\pi d_L^2` at **10 pc**, coming back
    ~1e17 too bright with no warning.

    This is the line flux you compare against an observed Halpha, so a wrong
    answer here is a wrong scientific answer, not a wrong plot.
    """
    omitted = {"sfh_dpl_log_total_mass": jnp.asarray(10.0)}  # legal: z is Fixed
    explicit = {**omitted, "redshift": jnp.asarray(0.5)}

    got = np.asarray(model_fixed_z.measure_line_fluxes(omitted))
    expected = np.asarray(model_fixed_z.measure_line_fluxes(explicit))

    # Vacuity guard: under the bug both arms are the *same* number, so an
    # equivalence assertion alone would pass on a model where z barely matters.
    # Pin that the redshift genuinely moves this quantity by orders of magnitude.
    at_zero = np.asarray(
        model_fixed_z.measure_line_fluxes({**omitted, "redshift": jnp.asarray(0.0)})
    )
    assert np.nanmax(np.abs(at_zero / expected)) > 1e3, (
        "z=0 and z=0.5 give comparable line fluxes here, so this test cannot "
        "detect the dropped-redshift bug — pick a redshift further from 0"
    )

    np.testing.assert_allclose(got, expected, rtol=1e-10)
