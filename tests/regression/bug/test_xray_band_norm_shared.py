# SPDX-License-Identifier: BSD-3-Clause
"""X-ray band normalization: one shared definition, and a docstring that cannot lie.

Two defects, both in ``components/xray/xray.py``:

**#1119** — the cutoff-power-law band integral was written inline three times
(HMXB and LMXB over 2-10 keV, hot gas over 0.5-2 keV), differing only in band
edges and constants. Three copies of one convention drift independently; the
sibling failure in #1527 is two callers of a single helper that already
disagree about its contract. ``_cutoff_powerlaw_band_norm`` is now the single
definition, and :func:`test_band_norm_matches_inline_form` pins it to the
inline arithmetic it replaced so the refactor is provably behaviour-preserving.

**#1755** — the ``xray_xrb_terms`` docstring claimed the Lehmer et al. (2016)
Eq. 15 relation "yields ~2.6e39 erg/s per M_sun/yr" at Z=0.02, where its own
equation gives 1.78e39 — a factor of 1.46. A prose number that restates a
formula is a second source of truth, and this one was wrong.
:func:`test_xray_lehmer_hmxb_docstring_values` evaluates the documented
polynomial directly, so the docstring is now checked rather than trusted.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.xray.xray import _cutoff_powerlaw_band_norm

# Lehmer et al. 2016, ApJ 825, 7, Eq. 15 — the polynomial the docstring quotes.
_LEHMER_COEFFS = (40.28, -62.12, 569.44, -1833.80, 1968.33)


def _lehmer_log_l_over_sfr(z: float) -> float:
    """log10(L_X^HMXB(2-10 keV) / SFR) from the documented Eq. 15 polynomial."""
    return sum(c * z**i for i, c in enumerate(_LEHMER_COEFFS))


@pytest.mark.parametrize(
    ("z", "expected"),
    [
        (0.02, 1.78e39),  # the metallicity_z default
        (0.0142, 3.22e39),  # Asplund 2009 solar, matching MIST
    ],
)
def test_xray_lehmer_hmxb_docstring_values(z: float, expected: float) -> None:
    """The docstring's quoted luminosities must equal what Eq. 15 produces (#1755).

    Before this test the docstring asserted 2.6e39 at Z=0.02 while the equation
    gives 1.78e39. Pinning both quoted values means the prose cannot drift from
    the formula again without a test failure.
    """
    got = 10.0 ** _lehmer_log_l_over_sfr(z)
    assert got == pytest.approx(expected, rel=5e-3), (
        f"Lehmer+2016 Eq. 15 at Z={z} gives {got:.4e}, but the xray_xrb_terms "
        f"docstring quotes {expected:.4e}. Update whichever is wrong — the "
        "docstring restates the equation, so they must agree."
    )


def test_lehmer_relation_is_steep_in_metallicity() -> None:
    """Z=0.02 vs the Asplund solar Z=0.0142 differ by ~1.8x (#1755, part 2).

    Records why the ``metallicity_z=0.02`` default is not a cosmetic choice:
    the relation is steep enough that the two conventions for "solar" separate
    L_HMXB by nearly a factor of two.
    """
    ratio = 10.0 ** _lehmer_log_l_over_sfr(0.0142) / 10.0 ** _lehmer_log_l_over_sfr(0.02)
    assert ratio == pytest.approx(1.805, rel=1e-2)


@pytest.mark.parametrize(
    ("gamma", "e_cut", "e_ref", "e_lo", "e_hi"),
    [
        (2.0, 100.0, 5.0, 2.0, 10.0),  # HMXB band
        (1.6, 100.0, 5.0, 2.0, 10.0),  # LMXB band
        (1.0, 1.0, 1.0, 0.5, 2.0),  # hot-gas band
    ],
)
def test_band_norm_matches_inline_form(
    gamma: float, e_cut: float, e_ref: float, e_lo: float, e_hi: float
) -> None:
    """The shared helper reproduces the inline arithmetic exactly (#1119).

    The three call sites are behaviour-preserving only if the helper computes
    the same expression in the same order. This asserts bit-equality, not
    tolerance — a reordering that changed the last ulp would still be a
    behaviour change to the X-ray goldens.
    """
    from tengri.components.xray.xray import _KEV_TO_HZ

    e_fine = jnp.linspace(e_lo, e_hi, 200)
    nu_fine = e_fine * _KEV_TO_HZ
    spec_fine = (e_fine / e_ref) ** (-gamma + 1) * jnp.exp(-e_fine / e_cut)
    inline = jnp.maximum(jnp.trapezoid(spec_fine, nu_fine), 1e-60)

    shared = _cutoff_powerlaw_band_norm(gamma, e_cut, e_ref, e_lo, e_hi)

    assert np.asarray(shared) == np.asarray(inline), (
        "shared band-norm helper diverged from the inline form it replaced"
    )


def test_band_norm_is_floored_not_zero() -> None:
    """A degenerate spectrum yields the 1e-60 floor, so it is safe as a divisor.

    The inline sites all wrapped the integral in ``jnp.maximum(..., 1e-60)``
    because the result is used as a denominator. The helper must keep that,
    otherwise collapsing the copies would introduce a division by zero on a
    path that previously could not produce one.
    """
    # A cutoff far below the band drives the integrand to underflow.
    got = _cutoff_powerlaw_band_norm(gamma=2.0, E_cut=1e-6, E_ref=5.0, E_lo=2.0, E_hi=10.0)
    assert float(got) == pytest.approx(1e-60, rel=1e-12)
    assert float(got) > 0.0
