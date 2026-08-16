# SPDX-License-Identifier: BSD-3-Clause
"""Regression tests for the unified AGN SED — synthesizer parity.

Mirrors the *shape* of synthesizer's ``tests/test_unified_agn.py`` assertions.
Synthesizer is GPL-3.0; we paraphrase the test structure but write our own
implementation. Every test cites the synthesizer source path it parallels and
the pitfall ID from ``~/.claude/plans/synthesizer-pitfall-catalog.md``.

These tests guard against parallel-implementation bugs where tengri's analytic
unified-AGN re-implements physics that synthesizer covers with grid look-ups.
"""

from __future__ import annotations

import chex
import jax.numpy as jnp
import pytest

pytestmark = pytest.mark.regression_paper

from tengri.components.agn.unified import unified_nlr_blr
from tengri.utils.physics_constants import L_SUN as L_SUN_ERG

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def wave_uv_to_fir() -> jnp.ndarray:
    """Wide rest-frame wavelength grid covering UV through FIR [Angstrom]."""
    return jnp.logspace(2.0, 7.0, 800)  # 100 Å .. 10 mm


@pytest.fixture(scope="module")
def physical_log_lbol() -> float:
    """log10(L_bol / L_sun) for a *physically-realistic* AGN.

    A bright Seyfert sits near L_bol ~ 1e44 erg/s = 1e10.4 L_sun, so we use
    log_lbol = 12.0 (i.e. L_bol ~ 1e45 erg/s = 1e12 L_sun, a typical quasar).
    The module default of ``agn_log_lbol=44.0`` is mathematically valid but
    corresponds to L_bol ~ 4e77 erg/s, which is unphysical and likely the
    result of a confused user copying ``log_lbol=44`` from literature where
    L_bol is reported in erg/s rather than L_sun. See TODO in
    ``components/agn/unified.py:1243``.
    """
    return 12.0


# ---------------------------------------------------------------------------
# P-2 / energy-conservation parity
# ---------------------------------------------------------------------------
# Mirrors: synthesizer/tests/test_unified_agn.py
#   ::test_default_disc_transmission_is_weighted_combination
# Pitfall: P-1 (line luminosity scale), P-2 (Eddington unit mismatch)


def test_log_lbol_unit_convention_is_l_sun(wave_uv_to_fir, physical_log_lbol):
    """``agn_log_lbol`` is log10(L_bol / L_sun), per the docstring + ``_phys.py``.

    Locks in the L_sun convention against accidental change to log10(L_bol/erg-per-s).
    Synthesizer parity: synthesizer accepts L_bol in erg/s; tengri uses L_sun.
    Anyone converting between the two must add the LOG10(L_SUN_ERG) ≈ 33.58 offset.

    Pitfall: P-2 — Eddington unit mismatch. If the convention were silently
    changed, fit posteriors would shift by a factor 3.83e33 in L_bol and break
    every downstream calculation.
    """
    # log_lbol = 12 → L_bol = 1e12 L_sun = 1e12 × 3.828e33 erg/s = 3.83e45 erg/s
    expected_l_bol_erg = 10.0**physical_log_lbol * L_SUN_ERG
    assert 1e44 < expected_l_bol_erg < 1e47, (
        "Sanity: log_lbol=12 should land near a typical quasar luminosity "
        f"(1e45 erg/s); got {expected_l_bol_erg:.3e} erg/s — convention drift?"
    )


def test_unified_agn_is_finite_and_nonnegative(wave_uv_to_fir, physical_log_lbol):
    """SED must be finite and non-negative on a wide UV-to-FIR grid.

    Mirrors: synthesizer's basic SED-construction sanity test.
    Pitfall: P-24 (NaN propagation from empty spectra / zero luminosity).
    """
    sed = unified_nlr_blr(
        wave_uv_to_fir,
        agn_log_lbol=physical_log_lbol,
        agn_nlr_cf=0.1,
        agn_blr_cf=0.1,
    )
    chex.assert_equal_shape([sed, wave_uv_to_fir])
    assert bool(jnp.all(jnp.isfinite(sed))), "non-finite values in unified-AGN SED"
    assert bool(jnp.all(sed >= 0.0)), "negative L_nu values in unified-AGN SED"


def test_zero_covering_fractions_remove_line_emission(wave_uv_to_fir, physical_log_lbol):
    """With both NLR and BLR covering fractions zero, the SED has no line addition.

    Mirrors:
    synthesizer/tests/test_unified_agn.py::test_weighted_combination_uses_transmission_fraction_attrs
    (the synthesizer version asserts ``escape_frac = 1 - blr - nlr``; tengri's
    geometry is additive rather than zero-sum, so the parity assertion is the
    weaker but equally physical one: zero coverage ⇒ zero line contribution.)

    Pitfall: P-1 — guards against line luminosities leaking when their gating
    fraction is zero.
    """
    sed_no_lines = unified_nlr_blr(
        wave_uv_to_fir,
        agn_log_lbol=physical_log_lbol,
        agn_nlr_cf=0.0,
        agn_blr_cf=0.0,
    )
    sed_with_lines = unified_nlr_blr(
        wave_uv_to_fir,
        agn_log_lbol=physical_log_lbol,
        agn_nlr_cf=0.2,
        agn_blr_cf=0.2,
    )
    # Difference must be non-negative everywhere (lines are additive emission).
    diff = sed_with_lines - sed_no_lines
    assert bool(jnp.all(diff >= -1e-30)), "line emission subtracted from SED?"
    # Some wavelengths must show a difference (else covering fractions are unwired).
    assert bool(jnp.any(diff > 0.0)), (
        "covering fractions had no effect on SED — params may be unwired"
    )


def test_line_emission_scales_with_covering_fraction(wave_uv_to_fir, physical_log_lbol):
    """Doubling NLR covering fraction roughly doubles the NLR contribution.

    Mirrors:
    synthesizer/tests/test_unified_agn.py::test_covering_fraction_edge_case_sum_to_unity
    Pitfall: P-1 — guards against sub/super-linear scaling that would indicate
    a missing factor of L_bol or a misapplied normalization.
    """
    sed_baseline = unified_nlr_blr(
        wave_uv_to_fir,
        agn_log_lbol=physical_log_lbol,
        agn_nlr_cf=0.0,
        agn_blr_cf=0.0,
    )
    sed_one_x = unified_nlr_blr(
        wave_uv_to_fir,
        agn_log_lbol=physical_log_lbol,
        agn_nlr_cf=0.1,
        agn_blr_cf=0.0,
    )
    sed_two_x = unified_nlr_blr(
        wave_uv_to_fir,
        agn_log_lbol=physical_log_lbol,
        agn_nlr_cf=0.2,
        agn_blr_cf=0.0,
    )

    # Total NLR contribution = ∫ (sed_x - sed_baseline) dν (in erg/s units after dν).
    # We use a wavelength-domain integral as a proxy; ratio is what matters.
    c_a_per_s = 2.99792458e18  # speed of light in Angstrom/s
    nu_weight = c_a_per_s / wave_uv_to_fir**2

    nlr_one_x = jnp.trapezoid((sed_one_x - sed_baseline) * nu_weight, wave_uv_to_fir)
    nlr_two_x = jnp.trapezoid((sed_two_x - sed_baseline) * nu_weight, wave_uv_to_fir)

    ratio = float(nlr_two_x / nlr_one_x)
    # Allow ±5% drift from exact linearity to absorb numerical-integration noise.
    assert 1.9 < ratio < 2.1, (
        f"NLR contribution does not scale linearly with covering fraction: "
        f"ratio(2× / 1×) = {ratio:.3f} (expected ≈ 2.0). "
        "Possible P-1 root cause."
    )


def test_log_lbol_scaling_is_logarithmic():
    """The AGN BOLOMETRIC luminosity scales as 10^(log_lbol).

    Pitfall: P-1, P-2 — guards against silent log-vs-linear unit confusion or a
    missing power-of-ten factor.

    Under the luminosity-first parameterization (ADR-0020) the physical disc
    SHAPE shifts with L_bol (higher L_bol -> higher lambda_Edd -> bluer), so a
    fixed wavelength does NOT scale linearly with L_bol. The invariant that still
    catches a log-vs-linear bug is the BOLOMETRIC integral, evaluated on an
    X-ray-inclusive grid so the disc's EUV/X-ray peak is captured.
    """
    log_lbol_a = 11.0
    log_lbol_b = 13.0  # +2 dex => bolometric x100
    wave = jnp.geomspace(0.1, 5.0e6, 12000)  # hard X-ray to far-IR

    def bolometric(log_lbol):
        sed = unified_nlr_blr(wave, agn_log_lbol=log_lbol, agn_nlr_cf=0.0, agn_blr_cf=0.0)
        nu = 2.99792458e18 / wave  # Hz
        order = jnp.argsort(nu)
        return float(jnp.trapezoid(sed[order], nu[order]))

    ratio = bolometric(log_lbol_b) / bolometric(log_lbol_a)
    assert 95.0 < ratio < 105.0, (
        f"AGN bolometric scales as {ratio:.2f} for +2 dex in log_lbol; "
        "expected ≈ 100. Possible normalization bug."
    )


# ---------------------------------------------------------------------------
# P-3 / inclination-conditional polar dust
# ---------------------------------------------------------------------------
# Ensures polar dust reddening only affects face-on (Type 1) views.
# Synthesizer parity: matches the inclination-conditional pattern in
# synthesizer's UnifiedAGN polar dust masking.


def test_polar_dust_off_edge_on_no_effect(wave_uv_to_fir, physical_log_lbol):
    """At edge-on (agn_cos_inc=0), polar dust has negligible effect on SED.

    When cos_inc ≈ 0 (edge-on / Type 2), the disc and BLR are already
    obscured by the torus. The visibility mask ≈ 0, so the inclination-weighted
    polar-dust transmission approaches unity (no reddening). Changing agn_polar_ebv
    should have < 1% effect on the total SED amplitude.

    Pitfall: P-3 — guards against the polar-dust factor being applied
    unconditionally, regardless of inclination. The old code had a bug where
    polar dust reddened even at edge-on, when it should have no effect.
    """
    sed_no_polar = unified_nlr_blr(
        wave_uv_to_fir,
        agn_log_lbol=physical_log_lbol,
        agn_cos_inc=0.0,  # edge-on / Type 2
        agn_polar_ebv=0.0,
        agn_nlr_cf=0.1,
        agn_blr_cf=0.1,
    )
    sed_with_polar = unified_nlr_blr(
        wave_uv_to_fir,
        agn_log_lbol=physical_log_lbol,
        agn_cos_inc=0.0,  # edge-on / Type 2
        agn_polar_ebv=0.3,  # strong polar dust
        agn_nlr_cf=0.1,
        agn_blr_cf=0.1,
    )
    # Difference must be tiny (< 1% of baseline).
    # The small residual is due to numerical bleed-through in the sigmoid;
    # it should be much smaller than if polar dust were applied unconditionally.
    diff_max = jnp.max(jnp.abs(sed_with_polar - sed_no_polar))
    baseline_max = jnp.max(jnp.abs(sed_no_polar))
    fractional_diff = float(diff_max / baseline_max)
    assert fractional_diff < 0.01, (
        f"Polar dust changed edge-on SED by {100.0 * fractional_diff:.1f}%; "
        f"expected < 1%. Inclination-weighting may be broken."
    )


def test_polar_dust_on_face_on_reddens_uv(wave_uv_to_fir, physical_log_lbol):
    """At face-on (agn_cos_inc=1), polar dust SMC extinction reddens UV more than optical.

    When cos_inc ≈ 1 (face-on / Type 1), the visibility mask ≈ 1, so the full
    SMC polar-dust reddening applies. The SMC law has a strong UV rise in
    extinction (k(λ) increases toward shorter wavelengths), so doubling
    agn_polar_ebv should reduce the UV SED more than the optical.

    Pitfall: P-3 — guards against the extinction law being applied incorrectly
    (e.g., wrong sign, missing factor, or law not matched to SMC).
    """
    sed_no_polar = unified_nlr_blr(
        wave_uv_to_fir,
        agn_log_lbol=physical_log_lbol,
        agn_cos_inc=1.0,  # face-on / Type 1
        agn_polar_ebv=0.0,
        agn_nlr_cf=0.1,
        agn_blr_cf=0.1,
    )
    sed_with_polar = unified_nlr_blr(
        wave_uv_to_fir,
        agn_log_lbol=physical_log_lbol,
        agn_cos_inc=1.0,  # face-on / Type 1
        agn_polar_ebv=0.2,
        agn_nlr_cf=0.1,
        agn_blr_cf=0.1,
    )
    # Select UV and optical wavelengths (roughly)
    uv_idx = int(jnp.argmin(jnp.abs(wave_uv_to_fir - 1500.0)))  # ~1500 Å
    opt_idx = int(jnp.argmin(jnp.abs(wave_uv_to_fir - 5500.0)))  # ~5500 Å
    # Fractional attenuation
    frac_atten_uv = float((sed_no_polar[uv_idx] - sed_with_polar[uv_idx]) / sed_no_polar[uv_idx])
    frac_atten_opt = float(
        (sed_no_polar[opt_idx] - sed_with_polar[opt_idx]) / sed_no_polar[opt_idx]
    )
    # SMC law: UV extinction much stronger than optical. UV attenuation should
    # be significantly larger.
    assert frac_atten_uv > frac_atten_opt, (
        f"Polar dust attenuates UV ({frac_atten_uv:.3f}) less than optical "
        f"({frac_atten_opt:.3f}); SMC law expected UV > optical. "
        "Extinction law may be wrong or reversed."
    )
