# SPDX-License-Identifier: BSD-3-Clause
r"""The "is there any light" guard must not depend on the units of the light.

``compute_luminosity_weighted_age`` and ``compute_luminosity_weighted_metallicity``
decide whether a weighted mean is meaningful, returning NaN rather than 0.0 when
the population emits nothing (#1404). That test used to be a bare
``l_total > 1e-20``.

The constant was chosen when the per-bin helper returned erg/s. Making the
bolometric reduction float32-safe (#1206) dropped the ``L_sun`` factor and
divided by the peak of ``ssp_flux_at_z``, rescaling the quantity by ~3.8e18 — so
the threshold stopped being anchored to anything. It kept passing because the
live regime sits tens of decades clear of it either way. The constant did not
change; its meaning did.

These tests pin the property that makes the guard correct regardless: it is
**scale-free**. Rescaling the input must not change which branch is taken.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from tengri.utils.sed_quantities import (
    compute_luminosity_weighted_age,
    compute_luminosity_weighted_metallicity,
)

pytestmark = pytest.mark.regression_bug

_WAVE = jnp.asarray(np.logspace(np.log10(1000.0), np.log10(50000.0), 128))
_AGES = jnp.asarray(np.logspace(6.0, 10.1, 12))


def _flux(scale=1.0):
    """A plain positive SSP cube, uniform in wavelength, declining with age."""
    decline = np.linspace(1.0, 0.2, _AGES.shape[0])[:, None]
    return jnp.asarray(np.ones((_AGES.shape[0], _WAVE.shape[0])) * decline * scale)


def _weights(scale=1.0):
    return jnp.asarray(np.linspace(1.0, 0.1, _AGES.shape[0]) * scale)


#: The **weights** are the knob that moves ``l_total``, and the only one.
#:
#: Scaling ``ssp_flux_at_z`` cannot: :func:`_per_bin_luminosity_relative` divides
#: by the peak of that very array, so ``l_total`` came back as 1.4134e+16 at flux
#: scales of 1e-25, 1.0 and 1e+25 — bit-identical on macOS/ARM, and within 1 ULP
#: (~1e-16 relative) on Linux/x86, where the peak division rounds differently.
#: A first version of this file swept the flux and passed with the old
#: unit-bearing threshold restored — green for a reason that had nothing to do
#: with what it claimed to test.
#:
#: Measured against the old ``l_total > 1e-20``: w=1e-36 gives l_total=1.41e-20
#: (still passes), w=1e-40 gives 1.41e-24 (NaN). So 1e-40 is inside the flip
#: window and 1.0 is far outside it — the pair discriminates.
_WEIGHT_SCALES = [1.0, 1e-20, 1e-40]


@pytest.mark.parametrize("scale", _WEIGHT_SCALES)
def test_weighted_age_is_invariant_under_rescaling_the_weights(scale):
    """Same galaxy, different mass normalization — the answer must not move.

    ``sum(l*a)/sum(l)`` is a ratio, so a common factor on the weights cancels
    exactly. Any dependence on ``scale`` means the emptiness guard is reading
    magnitude rather than distribution.
    """
    reference = float(compute_luminosity_weighted_age(_weights(1.0), _flux(), _AGES, _WAVE))
    assert np.isfinite(reference), "setup: the unit-scale reference is already NaN"

    got = float(compute_luminosity_weighted_age(_weights(scale), _flux(), _AGES, _WAVE))
    assert np.isfinite(got), (
        f"weights rescaled by {scale:.0e} returned NaN while the identical galaxy at "
        "unit scale returns a finite age — the emptiness guard is anchored to a "
        "magnitude whose units have since changed"
    )
    assert abs(got - reference) / reference < 1e-9, (
        f"weighted age moved from {reference:.6e} to {got:.6e} under a pure rescaling"
    )


@pytest.mark.parametrize("scale", _WEIGHT_SCALES)
def test_weighted_metallicity_is_invariant_under_rescaling_the_weights(scale):
    """The second consumer carried the identical constant, so it gets the identical test."""
    kwargs = dict(log_z=-2.0, log_z_initial=-3.0, log_z_final=-2.0)
    reference = float(
        compute_luminosity_weighted_metallicity(_weights(1.0), _flux(), _AGES, _WAVE, **kwargs)
    )
    assert np.isfinite(reference), "setup: the unit-scale reference is already NaN"

    got = float(
        compute_luminosity_weighted_metallicity(_weights(scale), _flux(), _AGES, _WAVE, **kwargs)
    )
    assert np.isfinite(got), f"weights rescaled by {scale:.0e} returned NaN"
    assert abs(got - reference) < 1e-9, (
        f"weighted metallicity moved from {reference:.6e} to {got:.6e} under a pure rescaling"
    )


@pytest.mark.parametrize("flux_scale", [1e-25, 1.0, 1e25])
def test_rescaling_the_flux_cannot_reach_the_guard_at_all(flux_scale):
    """Documents why the flux is the wrong knob, so nobody re-writes the weak test.

    Pins the peak-division invariance directly: ``l_total`` is independent of
    the flux normalization, so a flux sweep can never exercise the threshold no
    matter how many decades it spans.

    Stated as a margin rather than as bit-equality. The peak division is not
    bit-exact on every platform — it is exact on macOS/ARM and 1 ULP off on
    Linux/x86 — and an earlier ``got == reference`` here failed on CI for that
    reason alone, while the property it meant to pin was untouched. The claim
    that matters is quantitative: reaching the guard needs ``l_total`` to fall
    ~4.8e12x relative to the peak bin, so a ~1e-16 rounding wobble is some
    twenty-eight orders of magnitude short of being able to.
    """
    from tengri.utils.sed_quantities import _WEIGHT_SUM_REL_FLOOR, _per_bin_luminosity_relative

    ref_bins = _per_bin_luminosity_relative(_weights(1.0), _flux(1.0), _WAVE)
    reference = float(jnp.sum(ref_bins))
    got = float(jnp.sum(_per_bin_luminosity_relative(_weights(1.0), _flux(flux_scale), _WAVE)))

    rel = abs(got - reference) / abs(reference)
    assert rel < 1e-12, (
        f"l_total moved from {reference:.6e} to {got:.6e} ({rel:.3e} relative) under a flux "
        "rescaling — the peak factorization is no longer scale-free, and a flux-swept "
        "threshold test would now be meaningful where it previously was not"
    )

    # The setup half: reaching the guard needs l_total to fall by essentially its
    # whole magnitude (a factor ~4.8e12), i.e. a *relative* change of ~1. The
    # assertion above caps the flux sweep's effect at 1e-12 relative, so the two
    # together say rounding is twelve decades short. Deliberately not phrased as a
    # product of the two numbers: that had ~1.5x headroom and would have failed on
    # a platform that rounded 2 ULP instead of 1 — the very trap this file is fixing.
    margin = (reference / float(jnp.max(jnp.abs(ref_bins)))) / _WEIGHT_SUM_REL_FLOOR
    assert margin > 1e6, (
        f"this configuration now sits only {margin:.3e}x clear of the emptiness guard. "
        "The sweep above is no longer testing an unreachable threshold, so pick weights "
        "that put l_total back in the live regime"
    )


def test_a_genuinely_dark_population_still_returns_nan():
    """The guard must still fire for the case it exists to catch (#1404).

    Scale-invariance is only the right property if it does not also make the
    guard unable to trigger. Zero flux is zero at every scale.
    """
    dark = jnp.zeros((_AGES.shape[0], _WAVE.shape[0]))
    age = float(compute_luminosity_weighted_age(_weights(1.0), dark, _AGES, _WAVE))
    assert np.isnan(age), f"a population emitting nothing returned age {age}, expected NaN"


def test_zero_weights_also_return_nan():
    """The other route to an empty sum: light exists but nothing selects it."""
    no_weight = jnp.zeros(_AGES.shape[0])
    age = float(compute_luminosity_weighted_age(no_weight, _flux(), _AGES, _WAVE))
    assert np.isnan(age), f"zero weights returned age {age}, expected NaN"
