# SPDX-License-Identifier: BSD-3-Clause
"""Regression (#1404): a zero total weight yields NaN, not a plausible 0.0.

The four weighted averages in ``utils/sed_quantities`` all compute

    <x> = sum_i (w_i x_i) / sum_i w_i

Guarding the denominator with ``jnp.maximum(den, 1e-30)`` alone keeps the result
finite when ``sum_i w_i == 0``, but the value it produces is ``0.0`` — and a
mass-weighted age of exactly zero Gyr reads as "every star just formed" rather
than "this model has no stellar mass". The zero is wrong *and* plausible, so it
propagates silently into ``pred.properties``.

The module already prescribed the correct construction for exactly this case at
``compute_band_mean_fnu`` (return NaN when the band has no coverage); these four
siblings did not use it. NaN is correct here because the quantity is undefined,
and it is loud because it propagates.

Note the metallicity pair returns ``log10(max(mean_z, 1e-30))``. NaN survives
that wrapper: ``jnp.maximum(nan, 1e-30)`` is ``nan`` and ``log10(nan)`` is
``nan``. It is the *clamp* that is not NaN-stopping, which is precisely why the
``jnp.where`` has to sit upstream of it.
"""

import jax.numpy as jnp
import pytest

from tengri.utils.sed_quantities import (
    compute_luminosity_weighted_age,
    compute_luminosity_weighted_metallicity,
    compute_mass_weighted_age,
    compute_mass_weighted_metallicity,
)
from tests._grad_parity import assert_grad_matches_fd

N_AGE = 8
AGES_YR = jnp.logspace(6.0, 10.0, N_AGE)
# One SSP flux column per age bin; values are irrelevant to the zero-weight case,
# they only have to be finite so a nonzero-weight control produces a real number.
WAVE = jnp.linspace(1000.0, 10000.0, 16)
SSP_FLUX = jnp.ones((N_AGE, WAVE.shape[0]))

# The metallicity pair short-circuits to ``log_z`` unless BOTH ramp endpoints are
# supplied (``sed_quantities.py``: "if log_z_initial is None or log_z_final is
# None: return log_z"). The weighted average — and therefore this fix — is only
# reachable on the evolving-metallicity path, so both are passed here and they
# must differ for the ramp to be non-degenerate.
LOG_Z, LOG_Z_INI, LOG_Z_FIN = -2.0, -2.5, -1.5


@pytest.mark.regression_bug
def test_mass_weighted_quantities_are_nan_at_zero_weight():
    """Zero mass has no mass-weighted age or metallicity."""
    zero_w = jnp.zeros(N_AGE)

    age = compute_mass_weighted_age(zero_w, AGES_YR)
    assert jnp.isnan(age), f"expected NaN for zero weights, got {float(age)!r}"

    met = compute_mass_weighted_metallicity(zero_w, AGES_YR, LOG_Z, LOG_Z_INI, LOG_Z_FIN)
    assert jnp.isnan(met), f"expected NaN for zero weights, got {float(met)!r}"


@pytest.mark.regression_bug
def test_luminosity_weighted_quantities_are_nan_at_zero_luminosity():
    """A population that emits nothing has no luminosity-weighted average."""
    zero_w = jnp.zeros(N_AGE)

    age = compute_luminosity_weighted_age(zero_w, SSP_FLUX, AGES_YR, WAVE)
    assert jnp.isnan(age), f"expected NaN for zero luminosity, got {float(age)!r}"

    met = compute_luminosity_weighted_metallicity(
        zero_w, SSP_FLUX, AGES_YR, WAVE, LOG_Z, LOG_Z_INI, LOG_Z_FIN
    )
    assert jnp.isnan(met), f"expected NaN for zero luminosity, got {float(met)!r}"


@pytest.mark.regression_bug
def test_constant_metallicity_shortcut_is_untouched():
    """With one ramp endpoint missing the function returns log_z before the average.

    Pinned so the fix is not mistaken for a change to the constant-Z path.
    """
    zero_w = jnp.zeros(N_AGE)
    assert compute_mass_weighted_metallicity(zero_w, AGES_YR, LOG_Z) == LOG_Z


@pytest.mark.regression_bug
def test_nonzero_weight_is_unchanged_and_finite():
    """The guard must not disturb the ordinary path: it is a pure gate on den>0."""
    w = jnp.zeros(N_AGE).at[3].set(1.0)

    age = compute_mass_weighted_age(w, AGES_YR)
    assert jnp.isfinite(age)
    # A single populated bin must return that bin's age exactly.
    assert jnp.allclose(age, AGES_YR[3] / 1e9)

    met = compute_mass_weighted_metallicity(w, AGES_YR, LOG_Z, LOG_Z_INI, LOG_Z_FIN)
    assert jnp.isfinite(met)
    # Single populated bin ⇒ that bin's ramp value, in log10(Z).
    t_frac = AGES_YR[3] / jnp.max(AGES_YR)
    expected = LOG_Z_FIN + (LOG_Z_INI - LOG_Z_FIN) * t_frac
    assert jnp.allclose(met, expected)


@pytest.mark.regression_bug
def test_gradient_is_not_poisoned_by_the_nan_branch():
    """The NaN is a literal constant, so no NaN gradient flows through it.

    ``jnp.where`` evaluates both branches, so a NaN produced *from the inputs*
    would poison the gradient. Here the untaken branch is a constant and the
    taken branch stays finite (``0 / 1e-30 == 0``), so grad is clean wherever
    the weights are nonzero.
    """
    w = jnp.zeros(N_AGE).at[3].set(2.0)
    g = assert_grad_matches_fd(lambda ww: compute_mass_weighted_age(ww, AGES_YR), w)
    assert jnp.all(jnp.isfinite(g)), f"non-finite gradient: {g}"
    assert jnp.any(g != 0.0), (
        "`g` is identically zero — finite is not enough, "
        "a value that has collapsed to zero is as unusable as a NaN one (#2100)"
    )
