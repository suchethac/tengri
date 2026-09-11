# SPDX-License-Identifier: BSD-3-Clause
"""Physics tests for the Yang+22 / Lehmer+2016 X-ray scalings.

Verifies the CIGALE ``yang20`` upgrade:

* ``regression_paper`` — HMXB and LMXB integrated 2–10 keV luminosities
  match the Lehmer+19 / Lehmer+14 polynomial fits at canonical (Z, age).
* ``regression_paper`` — diffuse hot-gas 0.5–2 keV integral matches
  Mineo+2012 / Yang+22 normalization ``8.3 × 10³⁸ · SFR`` erg/s.
* ``regression_bug`` — ``xray_anisotropy`` denominator is now present
  (CIGALE ``yang20.py`` lines 231–234; previously absent, biasing
  face-on flux by ~ 7 %).
* ``bounds`` / ``gradient`` — XRB scaling positive everywhere and
  smoothly differentiable in metallicity.
"""

from __future__ import annotations

import chex
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.xray.xray import (
    xray_anisotropy,
    xray_hotgas,
    xray_total,
    xray_xrb,
)
from tests._grad_parity import assert_grad_matches_fd


# ---------------------------------------------------------------- helpers
def _band_integral(
    wave_aa: jnp.ndarray, L_nu: jnp.ndarray, e_lo_kev: float, e_hi_kev: float
) -> float:
    """Integrate ∫ L_ν dν over a keV band (returns erg/s)."""
    kev_per_aa = 12.398  # h·c / keV in Å
    # restrict to the band
    e_kev = kev_per_aa / np.asarray(wave_aa)
    mask = (e_kev >= e_lo_kev) & (e_kev <= e_hi_kev)
    nu_band = (e_kev[mask] / kev_per_aa) * 2.998e18  # Hz (c in Å/s)
    order = np.argsort(nu_band)
    return float(np.trapezoid(np.asarray(L_nu)[mask][order], nu_band[order]))


# ---------------------------------------------------------------- HMXB / LMXB
@pytest.mark.regression_paper
def test_hmxb_band_integral_matches_lehmer19_solar() -> None:
    """HMXB 2–10 keV integral matches Yang+22 / Lehmer+19 at Z=Z_sun, SFR=1.

    Reference (Yang+22 yang20.py:207–214, Lehmer+19 polynomial):
        log L_HMXB(2–10) [W] = 33.28 − 62.12 Z + 569.44 Z² − 1833.8 Z³
                              + 1968.33 Z⁴ + log SFR
    At Z = 0.02, SFR = 1: log L = 32.250 W → L ≈ 1.78×10³⁹ erg/s.
    Tolerance 5 % covers band-integration grid choice.
    """
    wave = jnp.logspace(np.log10(0.01), np.log10(124.0), 4000)  # 0.1 – 1.2e5 keV
    L = xray_xrb(
        wave,
        sfr=1.0,
        stellar_mass=0.0,  # zero LMXB to isolate HMXB
        metallicity_z=0.02,
        stellar_age_gyr=5.0,
    )
    L_band = _band_integral(wave, L, 2.0, 10.0)
    expected = 10**32.250 * 1e7  # W → erg/s
    chex.assert_tree_all_finite(L)
    assert 0.95 * expected < L_band < 1.05 * expected, f"got {L_band:.3e}, expected {expected:.3e}"


@pytest.mark.regression_paper
def test_lmxb_band_integral_matches_lehmer14_age10gyr() -> None:
    """LMXB 2–10 keV integral matches Yang+22 / Lehmer+14 at age=10 Gyr.

    Reference (Yang+22 yang20.py:217–224, Lehmer+14 quartic in logT):
        log L_LMXB(2–10) [W/(M*/1e10)] = 33.276 − 1.503 logT − 0.423 logT²
                                        + 0.425 logT³ + 0.136 logT⁴
    At logT = 1 (age 10 Gyr): log L = 31.911 W per 1e10 Msun
    → ~ 8.15×10³⁸ erg/s for M* = 1×10¹⁰ Msun.
    """
    wave = jnp.logspace(np.log10(0.01), np.log10(124.0), 4000)
    L = xray_xrb(
        wave,
        sfr=0.0,
        stellar_mass=1e10,  # zero SFR to isolate LMXB
        metallicity_z=0.02,
        stellar_age_gyr=10.0,
    )
    L_band = _band_integral(wave, L, 2.0, 10.0)
    expected = 10**31.911 * 1e7  # W → erg/s
    chex.assert_tree_all_finite(L)
    assert 0.95 * expected < L_band < 1.05 * expected


@pytest.mark.bounds
def test_hmxb_metallicity_dependence_positive_everywhere() -> None:
    """HMXB normalization stays > 0 across the Z = 0 → 0.05 range.

    Physical bound: a luminosity per SFR can't be negative; the quartic
    is empirical and must stay in the physically meaningful regime
    over the validity range.
    """
    wave = jnp.array([4.0])  # ≈ 3 keV
    for Z in (0.0001, 0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05):
        L = xray_xrb(wave, sfr=1.0, stellar_mass=0.0, metallicity_z=Z)
        assert float(L[0]) > 0.0, f"Z={Z}: L = {float(L[0])}"


@pytest.mark.gradient
def test_hmxb_smooth_in_metallicity() -> None:
    """d L_HMXB / dZ finite and well-behaved at Z = Z_sun.

    Lehmer+19 quartic is C∞ in Z; ensures upstream stellar metallicity
    can be optimized against X-ray data without gradient pathologies.
    """
    wave = jnp.array([4.0])

    def hmxb_at_Z(Z: float) -> float:
        return xray_xrb(wave, sfr=1.0, stellar_mass=0.0, metallicity_z=Z)[0]

    grad = assert_grad_matches_fd(hmxb_at_Z, jnp.array(0.02))
    assert jnp.isfinite(grad)
    assert jnp.any(grad != 0.0), (
        "`grad` is identically zero — finite is not enough, "
        "a value that has collapsed to zero is as unusable as a NaN one (#2100)"
    )
    # At Z=Z_sun the quartic has negative slope (−62.12 + ...). Numerical:
    # d log L / dZ = (−62.12 + 1138.88·0.02 − 5501.4·0.0004 + 7873.3·8e-6)
    #             = −62.12 + 22.78 − 2.20 + 0.063 ≈ −41.5
    # So d L / dZ = L · ln(10) · (−41.5) < 0
    assert float(grad) < 0.0


# ---------------------------------------------------------------- hot gas
@pytest.mark.regression_paper
def test_hotgas_band_integral_matches_mineo2012() -> None:
    """Hot-gas 0.5–2 keV integral matches Mineo+2012 / Yang+22:

        L_hotgas(0.5–2 keV) = 8.3×10³⁸ · SFR  [erg/s]

    Tolerance 5 % for band-integration grid resolution.
    """
    wave = jnp.logspace(np.log10(1.0), np.log10(124.0), 4000)
    L = xray_hotgas(wave, sfr=1.0)
    L_band = _band_integral(wave, L, 0.5, 2.0)
    expected = 8.3e38
    chex.assert_tree_all_finite(L)
    assert 0.95 * expected < L_band < 1.05 * expected, f"got {L_band:.3e}, expected {expected:.3e}"


@pytest.mark.bounds
def test_hotgas_cut_off_above_10kev() -> None:
    """Hot-gas L_ν at 30 keV is < 0.1 % of L_ν at 2 keV.

    Physical bound: the Yang+22 template
    L_ν(λ) ∝ λ⁻² exp(−λ_1keV/λ) ≡ (ν/ν_1keV)² exp(−ν/ν_1keV) is a
    Wien-tail-like shape — rising as ν² inside the soft band and
    exponentially cut above ~ a few keV. Beyond 10 keV the XRB power
    law should be many orders of magnitude brighter than diffuse gas.
    """
    wave = jnp.array([12.4 / 2.0, 12.4 / 30.0])  # 2 keV, 30 keV
    L = xray_hotgas(wave, sfr=1.0)
    assert float(L[1] / L[0]) < 1e-3, (
        f"L_ν(30 keV) / L_ν(2 keV) = {float(L[1] / L[0]):.3g} not exponentially small"
    )


# ---------------------------------------------------------------- anisotropy
@pytest.mark.regression_bug
def test_anisotropy_normalized_to_30deg() -> None:
    """Anisotropy factor returns 1 at θ = 30° (yang20.py:231–234).

    Reason: L_2500 (which sets the X-ray normalization via α_ox) is
    a θ = 30° quantity in SKIRTOR. The denominator
    1 − 0.13397 a1 − 0.25 a2 normalizes so f(cos 30°) = 1.
    Previously the denominator was absent and face-on flux was biased
    high by ~ 7 %.
    """
    cos_30 = jnp.cos(jnp.deg2rad(30.0))  # ≈ 0.866
    L = jnp.array([1.0])
    result = xray_anisotropy(L, cos_inc=cos_30, a1=0.5, a2=0.0)
    chex.assert_trees_all_close(result, jnp.array([1.0]), rtol=1e-5)


@pytest.mark.regression_bug
def test_anisotropy_face_on_brighter_than_30deg_default() -> None:
    """Default a1=0.5: face-on is ~ 7 % brighter than θ = 30°.

    f(1)/f(cos30) = 1.0 / (1 - 0.13397·0.5) = 1.0717. Locks the
    physical sign of the Yang+22 correction.
    """
    face_on = xray_anisotropy(jnp.array([1.0]), cos_inc=1.0)
    at_30 = xray_anisotropy(jnp.array([1.0]), cos_inc=jnp.cos(jnp.deg2rad(30.0)))
    chex.assert_trees_all_close(face_on / at_30, jnp.array([1.0717]), rtol=1e-3)


# ---------------------------------------------------------------- xray_total
@pytest.mark.regression_paper
def test_xray_total_sums_three_components() -> None:
    """``xray_total`` equals XRB + hot-gas + AGN corona.

    Conservation check: the integrated 0.5–10 keV luminosity from
    xray_total must equal the sum of the three components computed
    independently, to numerical tolerance.

    Note: xray_total currently passes its E_cut (set by the AGN
    corona, default 300 keV) to xray_xrb, so the standalone XRB
    call must match that to compare exactly. A follow-up could split
    XRB and AGN E_cuts.
    """
    from tengri.components.xray.xray import xray_agn_corona

    wave = jnp.logspace(np.log10(0.1), np.log10(124.0), 2000)
    L_2500 = 1e44 / (5.15 * 1.199e15)  # Hopkins+2007 BC_2500 → erg/s/Hz

    # Match xray_total's internal kwargs exactly. xray_total currently:
    #   * passes its own E_cut (300 keV, AGN default) to xray_xrb — so the
    #     XRB cutoff differs from xray_xrb's standalone default of 100 keV;
    #   * calls xray_hotgas(..., gamma=1.0, E_cut=1.0).
    # Splitting XRB and AGN E_cuts is tracked in the plan file.
    total = xray_total(
        wave,
        sfr=2.0,
        stellar_mass=3e10,
        metallicity_z=0.02,
        stellar_age_gyr=5.0,
        l_2500_30deg=L_2500,
        E_cut=300.0,
        apply_anisotropy=False,
    )
    xrb = xray_xrb(
        wave,
        sfr=2.0,
        stellar_mass=3e10,
        metallicity_z=0.02,
        stellar_age_gyr=5.0,
        E_cut=300.0,  # ← xray_total passes its own (AGN) E_cut here
    )
    hot = xray_hotgas(wave, sfr=2.0, gamma=1.0, E_cut=1.0)
    agn = xray_agn_corona(wave, l_2500_30deg_erg_hz=L_2500, apply_anisotropy=False)
    chex.assert_trees_all_close(total, xrb + hot + agn, rtol=1e-10, atol=1e-30)


@pytest.mark.limit
def test_xray_total_no_agn_is_finite_and_strictly_positive() -> None:
    """l_2500_30deg = 0 ⇒ no AGN corona; total stays finite + non-negative.

    Physical limit: with no AGN upstream the corona term collapses to
    zero (NaN guard in xray_agn_corona_from_disc), leaving XRBs + hot
    gas as the only contributors. This locks the NaN guard against
    accidental removal.
    """
    wave = jnp.logspace(np.log10(0.1), np.log10(124.0), 2000)
    total = xray_total(wave, sfr=1.0, stellar_mass=1e10, l_2500_30deg=0.0)
    chex.assert_tree_all_finite(total)
    assert bool(jnp.all(total >= 0.0))
    # XRBs alone for SFR=1, M*=1e10 inject ~ 10^39 erg/s in 2-10 keV,
    # so peak L_ν should be ≳ 1e22 erg/s/Hz in the soft band.
    assert float(jnp.max(total)) > 1e22
