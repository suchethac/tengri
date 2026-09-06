# SPDX-License-Identifier: BSD-3-Clause
"""Denominator floors in ``sed_quantities`` must be sized for the derivative (#1860).

A floor sized for a *value* is not sized for a *derivative*. Division's VJP
carries ``-num/den**2``, so a denominator floored at ``1e-30`` squares to
``1e-60`` — exactly ``0.0`` in float32, whose smallest normal is 1.175e-38 —
and the reverse pass divides by zero. The forward value is unaffected, which is
what let this sit unnoticed: ``0/1e-30`` is a clean ``0.0``.

``representable_floor`` cannot see this class. ``1e-30`` is *above* float32's
``tiny``, so the helper returns it unchanged and a floor census reports the site
clean while its VJP produces NaN. The derivative-safe bound is ``sqrt(tiny)``
(1.084e-19 in float32) — already written in ``representable_floor``'s own Notes
before this bug was found, and implemented as ``representable_denominator`` in
#1863.

Two shapes are fixed here, and they need different remedies:

* **plain divide** (``compute_dn4000``, ``compute_balmer_break``) — the floor is
  the only guard, so raise it to the derivative-safe bound.
* **double ``where``** (``compute_mass_weighted_age``,
  ``compute_luminosity_weighted_metallicity``, ``_mean_flux_in_band``,
  ``compute_uv_slope_beta``) — an outer ``jnp.where`` selects NaN on the
  degenerate branch, but **both** branches are differentiated and the discarded
  one contributes ``0 * inf = NaN`` to the survivor. Raising the floor is not
  enough; the denominator must be selected *before* the divide.

Follow-up to #1860; sibling of the ``_filter_integral_union`` fix in #1863.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.utils.scale import representable_denominator, representable_floor

pytestmark = pytest.mark.regression_bug


def test_the_bound_is_what_the_docstring_said_and_the_floor_helper_misses_it():
    """Non-vacuity: pin why a floor census cannot catch this class.

    If ``1e-30`` ever fell below ``tiny``, ``representable_floor`` would start
    catching these sites and the fix below would be redundant for a reason
    unrelated to the fix.
    """
    with jax.enable_x64(False):
        tiny = float(np.finfo(np.float32).tiny)
        # The value-sized helper passes 1e-30 through untouched: not its job.
        assert representable_floor(1e-30) == 1e-30
        assert tiny < 1e-30
        # The derivative-sized helper raises it, because 1e-60 is not a number.
        assert representable_denominator(1e-30) == pytest.approx(np.sqrt(tiny))
        assert np.float32(1e-30) ** 2 == 0.0


#: Degenerate inputs — an all-zero weight vector and an SED with no flux in the
#: measured band. These are the states each guard exists to survive, and the
#: states where the old floor actually bound.
def _zero_weights():
    return jnp.zeros(8), jnp.asarray(np.geomspace(1e6, 1e10, 8))


def _sed_and_wave(flat: bool = False):
    wave = np.geomspace(1.0e3, 1.0e5, 400)
    sed = np.zeros_like(wave) if flat else 1.0e28 * (wave / 3000.0) ** -1.0
    return jnp.asarray(sed), jnp.asarray(wave)


@pytest.mark.parametrize("degenerate", [False, True])
def test_mass_weighted_age_gradient_is_finite_in_float32(degenerate):
    from tengri.utils.sed_quantities import compute_mass_weighted_age

    with jax.enable_x64(False):
        w, ages = _zero_weights()
        w = w if degenerate else w.at[:].set(jnp.linspace(0.1, 1.0, 8))
        g = jax.grad(lambda ww: jnp.sum(compute_mass_weighted_age(ww, ages)))(w)
        assert np.asarray(g).dtype == np.float32
        # grad-assert: finite-only — zero weights here; a zero age gradient is correct
        assert np.all(np.isfinite(np.asarray(g))), (
            f"mass-weighted-age gradient non-finite in float32 (degenerate={degenerate})"
        )


@pytest.mark.parametrize("empty_band", [False, True])
def test_mean_flux_in_band_gradient_is_finite_in_float32(empty_band):
    from tengri.utils.sed_quantities import _mean_flux_in_band

    with jax.enable_x64(False):
        sed, wave = _sed_and_wave()
        # An empty band is the state whose denominator the floor guards.
        lo, hi = (1.0e9, 2.0e9) if empty_band else (3000.0, 4000.0)
        g = jax.grad(lambda s: jnp.sum(_mean_flux_in_band(s, wave, lo, hi)))(sed)
        # grad-assert: finite-only — empty band here; a zero gradient is correct
        assert np.all(np.isfinite(np.asarray(g))), (
            f"_mean_flux_in_band gradient non-finite in float32 (empty_band={empty_band})"
        )


@pytest.mark.parametrize("flat", [False, True])
def test_break_indices_gradients_are_finite_in_float32(flat):
    from tengri.utils.sed_quantities import compute_balmer_break, compute_dn4000

    with jax.enable_x64(False):
        sed, wave = _sed_and_wave(flat=flat)
        for fn in (compute_dn4000, compute_balmer_break):
            g = jax.grad(lambda s, f=fn: jnp.sum(f(s, wave)))(sed)
            assert np.all(np.isfinite(np.asarray(g))), (
                f"{fn.__name__} gradient non-finite in float32 (flat={flat})"
            )
            assert np.any(np.asarray(g) != 0.0), (
                "`np.asarray(g)` is identically zero — finite is not enough, "
                "a value that has collapsed to zero is as unusable as a NaN one (#2100)"
            )


@pytest.mark.parametrize("flat", [False, True])
def test_uv_slope_beta_gradient_is_finite_in_float32(flat):
    """Three denominators in one expression, two of them the same ``sw``."""
    from tengri.utils.sed_quantities import compute_uv_slope_beta

    with jax.enable_x64(False):
        sed, wave = _sed_and_wave(flat=flat)
        g = jax.grad(lambda s: jnp.sum(jnp.nan_to_num(compute_uv_slope_beta(s, wave))))(sed)
        # grad-assert: finite-only — zeroed window here; a zero slope gradient is correct
        assert np.all(np.isfinite(np.asarray(g))), (
            f"uv_slope_beta gradient non-finite in float32 (flat={flat})"
        )


# ── float64 must not move ─────────────────────────────────────────


def test_float64_values_are_unchanged_by_the_derivative_sized_floor():
    """The floors move only in float32.

    ``representable_denominator`` resolves against the working dtype, and
    float64's bound is ``sqrt(2.225e-308) = 1.49e-154`` — 124 decades below the
    ``1e-30`` these sites use, so it returns them unchanged and no float64
    number moves. That is what makes this a float32-only fix rather than a
    physics change; it is asserted rather than asserted-about.
    """
    assert representable_denominator(1e-30) == 1e-30
    assert representable_denominator(1e-300) > 1e-300  # would move, correctly


def test_float64_break_indices_match_the_prefix_formula():
    from tengri.utils.sed_quantities import compute_dn4000

    sed, wave = _sed_and_wave()
    from tengri.utils.sed_quantities import _mean_flux_in_band

    red = _mean_flux_in_band(sed, wave, 4000.0, 4100.0)
    blue = _mean_flux_in_band(sed, wave, 3850.0, 3950.0)
    old = red / jnp.maximum(blue, 1e-30)
    np.testing.assert_allclose(np.float64(compute_dn4000(sed, wave)), np.float64(old), rtol=1e-13)
