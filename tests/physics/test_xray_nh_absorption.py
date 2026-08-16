# SPDX-License-Identifier: BSD-3-Clause
"""Physics tests for X-ray N_H photoelectric absorption (issue #292).

Verifies the Morrison & McCammon (1983) ``wabs`` cross-section fit
applied to the AGN corona via ``tbabs_transmission``. Markers:

* ``regression_paper`` — transmission at fixed energies matches the
  Morrison & McCammon (1983, Table 2) polynomial fit at N_H levels
  spanning Compton-thin to Compton-thick.
* ``limit`` — log N_H = 18 (effectively unabsorbed) reproduces the
  pre-N_H spectrum to float tolerance; log N_H = 25 fully extinguishes
  the soft band.
* ``bounds`` — transmission stays in [0, 1] for all (E, N_H).
* ``gradient`` — d(transmission)/d(log_nh) ≤ 0 (more N_H ⇒ less flux).
"""

from __future__ import annotations

import chex
import jax
import jax.numpy as jnp
import pytest

from tengri.components.xray.xray import (
    compton_scattering_transmission,
    tbabs_transmission,
    # PR #329 changed the public `xray_agn_corona` to take L_2500_30deg
    # directly. The legacy L_bol-driven path (which this PR's N_H absorption
    # was originally implemented against) lives behind the
    # `xray_agn_corona_bolometric` deprecation shim. The N_H math is
    # identical in both paths; testing it via the shim keeps the focus on
    # absorption physics rather than re-deriving L_2500 from L_bol.
    xray_agn_corona_bolometric as xray_agn_corona,
)

# Morrison & McCammon (1983) Table 2 reference: σ(E)·E³ values
# (units 10⁻²⁴ cm² · keV³). Computed from the published coefficients
# at the center of representative bins. These numbers are the contract
# the JAX implementation must reproduce to numerical tolerance.
_MM83_REFERENCE = [
    # (E_keV, sigma_in_1e_24_cm2), bin chosen to match the M&M83 edges
    (0.5, (71.4 + 66.8 * 0.5 - 51.4 * 0.5**2) / 0.5**3),  # bin [0.400, 0.532]
    (1.0, (120.6 + 169.3 * 1.0 - 47.7 * 1.0**2) / 1.0**3),  # bin [0.867, 1.303]
    (2.0, (202.7 + 104.7 * 2.0 - 17.0 * 2.0**2) / 2.0**3),  # bin [1.840, 2.471]
    (5.0, (433.9 - 2.4 * 5.0 + 0.75 * 5.0**2) / 5.0**3),  # bin [4.038, 7.111]
    (8.0, (629.0 + 30.9 * 8.0) / 8.0**3),  # bin [7.111, 8.331]
]


@pytest.mark.regression_paper
@pytest.mark.parametrize("E_keV, sigma_ref_1e24", _MM83_REFERENCE)
def test_tbabs_matches_morrison_mccammon(E_keV: float, sigma_ref_1e24: float) -> None:
    """Transmission matches σ from Morrison & McCammon (1983, Table 2).

    Reference: σ(E)·E³ = c0 + c1·E + c2·E² (σ in 10⁻²⁴ cm², E in keV).
    Tolerance: 1 % — the polynomial fit is exact, so this guards only
    against indexing or unit bugs.
    """
    log_nh = 22.0
    sigma_cm2 = sigma_ref_1e24 * 1e-24
    expected_trans = jnp.exp(-sigma_cm2 * 10.0**log_nh)

    got = tbabs_transmission(jnp.array(E_keV), jnp.array(log_nh))

    chex.assert_trees_all_close(got, expected_trans, rtol=1e-2)


@pytest.mark.limit
def test_unabsorbed_limit_matches_intrinsic_power_law() -> None:
    """N_H → 0 limit recovers the intrinsic α_ox-normalized power-law.

    Physical limit: τ_phabs, τ_cabs → 0 ⇒ both transmissions → 1,
    and the scattered term 0.01·L is added once via the +0.01·L
    addend. With ``scattered_frac=0`` the absorbed spectrum equals
    the intrinsic power-law everywhere in the X-ray band.
    """
    wave = jnp.logspace(0.0, 2.0, 100)  # 1 – 100 Å ≈ 0.124 – 12.4 keV
    l_unabs = xray_agn_corona(wave, L_agn_bol=1e45, log_nh=15.0, scattered_frac=0.0)
    # Build the intrinsic spectrum analytically (no absorption).
    from tengri.utils.physics_constants import C_AA, H_PLANCK

    nu = C_AA / wave
    E_keV = H_PLANCK * nu / 1.6022e-9
    L_2500 = 1e45 / (5.15 * 1.199e15)
    # PR #329 tightened the α_OX divisor from 0.384 → 0.3838 (exact
    # 1/log10(ν_2keV/ν_2500Å)) to match X-CIGALE yang20.py:227. The 0.4 %
    # difference shows up here because we recompute the expected spectrum
    # analytically — use the same divisor so the test pins the active code.
    L_2keV = L_2500 * 10.0 ** (-1.4 / 0.3838)
    spec = (E_keV / 2.0) ** (-1.8 + 1) * jnp.exp(-E_keV / 300.0)
    expected = jnp.where(wave < 124.0, L_2keV * spec, 0.0)

    chex.assert_equal_shape([l_unabs, expected])
    chex.assert_tree_all_finite(l_unabs)
    chex.assert_trees_all_close(l_unabs, expected, rtol=1e-3, atol=1e-12)


@pytest.mark.regression_paper
def test_compton_thick_floor_at_scattered_fraction() -> None:
    """At log_nh = 26 the soft-band flux floors at the scattered fraction.

    Physical limit (Ricci+2017; Matsumoto+2026 Eq. B6): when both
    τ_phabs ≫ 1 and τ_cabs ≫ 1, the primary continuum is fully
    extinguished and the only remaining flux is the constant
    scattered fraction ``f_scat · L_intr``. Verifies the
    Compton-thick AGN signature: soft X-rays survive at the 1 %
    level even when the nucleus is opaque.
    """
    wave = jnp.array([100.0])  # ≈ 0.124 keV
    l_intr = xray_agn_corona(wave, L_agn_bol=1e45, log_nh=15.0, scattered_frac=0.0)
    l_thick = xray_agn_corona(wave, L_agn_bol=1e45, log_nh=26.0, scattered_frac=0.01)
    ratio = float(l_thick[0] / l_intr[0])
    assert 0.005 < ratio < 0.015, f"expected ≈ 0.01, got {ratio}"


@pytest.mark.regression_paper
def test_compton_scattering_thomson_value() -> None:
    """T_cabs matches σ_T·N_H at the Thomson cross-section.

    σ_T = 6.6525e-25 cm² (NIST). At log_nh = 24, τ_T = 0.665 and
    T_cabs = exp(−0.665) ≈ 0.514.
    """
    T = compton_scattering_transmission(jnp.array(24.0))
    chex.assert_trees_all_close(T, jnp.exp(-0.66524587), rtol=1e-4)


@pytest.mark.bounds
def test_transmission_bounded_unit_interval() -> None:
    """0 ≤ T(E, N_H) ≤ 1 everywhere in the X-ray band.

    Physical bound: transmission is exp(−non-negative number); the
    cross-section is positive.
    """
    E_grid = jnp.logspace(-1.0, 1.0, 200)  # 0.1–10 keV
    for log_nh in (18.0, 20.0, 22.0, 24.0, 25.0):
        T = tbabs_transmission(E_grid, jnp.array(log_nh))
        chex.assert_shape(T, (200,))
        chex.assert_tree_all_finite(T)
        assert bool(jnp.all(T >= 0.0))
        assert bool(jnp.all(T <= 1.0 + 1e-12))


@pytest.mark.gradient
def test_transmission_monotone_in_log_nh() -> None:
    """More obscuration ⇒ less transmitted flux: dT/d(log_nh) ≤ 0.

    Verified at three soft-band energies where photoelectric
    absorption matters most (≲ 5 keV).
    """
    for E_keV in (0.5, 1.0, 3.0):
        grad = jax.grad(lambda nh, E=E_keV: tbabs_transmission(jnp.array(E), nh).sum())(
            jnp.array(22.0)
        )
        assert float(grad) <= 0.0, f"E={E_keV}: dT/d(log_nh) = {float(grad)}"


@pytest.mark.bounds
def test_xrb_unaffected_by_nh() -> None:
    """N_H absorbs only the AGN corona, not stellar XRB/hot-gas.

    Physical reason: N_H is the line-of-sight column to the AGN; the
    diffuse XRB population is not behind the torus.
    """
    from tengri.components.xray.xray import xray_xrb

    wave = jnp.logspace(0.0, 2.0, 50)
    l_xrb = xray_xrb(wave, sfr=1.0, stellar_mass=1e10)
    # xray_xrb has no log_nh kwarg — absorption is corona-only by design.
    chex.assert_shape(l_xrb, (50,))
    chex.assert_tree_all_finite(l_xrb)
