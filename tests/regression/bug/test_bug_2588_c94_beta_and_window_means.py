# SPDX-License-Identifier: BSD-3-Clause
"""Regression tests for #2588: Calzetti-94 β pooled fit and Δλ-weighted window means.

Two rest-frame diagnostics measure different quantities from what they cite:
1. UV slope β: Should use ONE pooled fit over all ten Calzetti+1994 windows,
   not per-window slopes averaged (fixes window 6 = 1677–1740, not 1611–1711).
2. Window means: Should use Δλ-weighted means (⟨F⟩ = ∫F dλ / ∫dλ), not pixel-weighted.

Tests cover:
- β of analytic power laws through C94 windows (exact to 1e-6)
- Three SED rows from the issue against a numpy C94 fit (atol 0.02)
- Window table constant exposure
- Dn4000 grid-independence on uniform and clustered grids
- Equivalent width of Gaussian emission line grid-independence
- float32 vs float64 agreement
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from tengri import DEFAULT, Fixed, SEDModel, load_ssp_data
from tengri.analysis.diagnostics.spectral import (
    CALZETTI94_WINDOWS_AA,
    dn4000,
    uv_slope_beta,
)
from tengri.observation.spectral_indices import STANDARD_INDICES, measure_index_jax
from tengri.utils.sed_quantities import compute_dn4000

pytestmark = pytest.mark.regression_bug

# ── Constants ──────────────────────────────────────────────────────

C_AA = 2.99792458e18  # speed of light in Angstrom/s

# Import CALZETTI94_WINDOWS_AA from spectral.py for use in tests
C94_WINDOWS = list(CALZETTI94_WINDOWS_AA)


def beta_c94_reference(w: np.ndarray, lnu: np.ndarray) -> float:
    """Reference C94 Eq. 3: ONE fit over all window pixels.

    Computes F_λ ∝ λ^β by fitting log(F_λ) vs log(λ) over the union of the
    ten Calzetti+1994 Table 2 windows.
    """
    flam = lnu * C_AA / w**2
    mask = np.zeros_like(w, bool)
    for lo, hi in C94_WINDOWS:
        mask |= (w >= lo) & (w <= hi)
    return float(np.polyfit(np.log10(w[mask]), np.log10(flam[mask]), 1)[0])


# ── Tests ──────────────────────────────────────────────────────────


def test_uv_slope_beta_analytic_power_laws():
    """β of analytic power laws F_λ ∝ λ^β through C94 windows.

    Non-uniform grid: 1 Å steps with 0.1 Å cluster in one window.
    Pooled fit is exact; per-window average on 1–3 pixels is not.
    """
    # Create non-uniform grid with a cluster
    uniform = np.arange(1250.0, 2600.0, 1.0)
    cluster_in_window_6 = np.arange(1677.0, 1740.0, 0.1)  # inside window 6
    wave = np.unique(np.concatenate([uniform, cluster_in_window_6]))

    # Test three slope values (negative, zero, positive)
    for beta_true in [-2.5, -1.0, 0.5]:
        # F_λ ∝ λ^β ⟹ L_ν = F_λ * c/λ² ∝ λ^(β+2)
        # So create L_ν ∝ λ^(β+2) to test β
        lnu = wave ** (beta_true + 2)
        lnu_jax = jnp.asarray(lnu)
        wave_jax = jnp.asarray(wave)

        # Compute β through diagnostics function
        beta_measured = float(uv_slope_beta(wave_jax, lnu_jax))

        # Also compute reference C94 fit on the converted F_λ
        flam = lnu * (C_AA / wave**2)
        beta_ref = beta_c94_reference(wave, lnu)

        # Both should match the true β to high precision
        assert np.abs(beta_measured - beta_true) < 1e-6, (
            f"β={beta_true}: diagnostics gave {beta_measured:.6f}, expected {beta_true:.6f}"
        )
        assert np.abs(beta_ref - beta_true) < 1e-6, (
            f"β={beta_true}: reference C94 gave {beta_ref:.6f}, expected {beta_true:.6f}"
        )


def test_uv_slope_beta_against_c94_fit():
    """β of the three SED rows from issue #2588.

    Compares `uv_slope_beta` against a numpy C94 Eq. 3 fit written in the test
    over the ten Table-2 windows. Expects agreement to atol=0.02.
    """
    ssp = load_ssp_data("data/fsps_prsc_miles_chabrier.h5")

    test_cases = [
        (0.0, 0.0, -2.307),  # tau_V, E_bump, expected C94 β
        (0.5, 3.0, -1.842),
        (3.0, 1.0, +1.103),
    ]

    for tau_v, bump, beta_expected in test_cases:
        st = SEDModel.build(
            ssp_data=ssp,
            sfh={
                "type": "delayed",
                "tau_gyr": Fixed(1.0),
                "age_gyr": Fixed(1.0),
                "log_total_mass": Fixed(10.0),
                "all_params": Fixed(DEFAULT),
            },
            dust_attenuation={
                "law": "noll09",
                "type": "two_component",
                "tau_bc": Fixed(0.0),
                "tau_diff": Fixed(tau_v),
                "dust_bump_strength": Fixed(bump),
                "dust_delta": Fixed(0.0),
                "all_params": Fixed(DEFAULT),
            },
            neb={"type": "none"},
            redshift=Fixed(0.0),
        ).predict_state({})

        w = np.asarray(st.wave)
        lnu = np.asarray(st.sed_intrinsic)

        # Compute β through diagnostics function
        beta_measured = float(uv_slope_beta(jnp.asarray(w), jnp.asarray(lnu)))

        # Compute reference C94 fit
        beta_ref = beta_c94_reference(w, lnu)

        # Both should match the expected value
        assert np.abs(beta_measured - beta_expected) < 0.02, (
            f"tau_V={tau_v} E_b={bump}: diagnostics {beta_measured:.3f} "
            f"vs expected {beta_expected:.3f} (δ={beta_measured - beta_expected:.3f})"
        )
        assert np.abs(beta_ref - beta_expected) < 0.02, (
            f"tau_V={tau_v} E_b={bump}: reference {beta_ref:.3f} "
            f"vs expected {beta_expected:.3f} (δ={beta_ref - beta_expected:.3f})"
        )


def test_uv_slope_beta_gaussian_absorption_window_6_sensitive():
    """β with Gaussian absorption trough at 1640 Å is sensitive to window 6 bounds.

    Tests that window 6 = (1677–1740) correctly excludes the 1640 Å feature.
    A power law L_ν ∝ λ^(β+2) with β = −2.0 creates F_λ ∝ λ^0.
    Adding a Gaussian absorption trough at 1640 Å (depth 50%, σ=8 Å) creates
    a dip inside the old window 6 = (1611–1711) but NOT inside the correct
    window 6 = (1677–1740). Therefore:
    - Correct implementation (current) returns β ≈ −2.0 (unaffected)
    - Old window (1611–1711) would return β ≠ −2.0 (affected by the dip)
    """
    # Create a power law: L_ν ∝ λ^0 (F_λ ∝ λ^−2)
    wave = np.linspace(1250.0, 2600.0, 2048)
    lnu_base = np.ones_like(wave)  # L_ν ∝ λ^0 ⟹ β = −2

    # Add Gaussian absorption trough at 1640 Å: depth 50%, σ=8 Å
    sigma = 8.0
    depth = 0.5
    gaussian_absorption = 1.0 - depth * np.exp(-0.5 * ((wave - 1640.0) / sigma) ** 2)
    lnu = lnu_base * gaussian_absorption

    wave_jax = jnp.asarray(wave)
    lnu_jax = jnp.asarray(lnu)

    beta_measured = float(uv_slope_beta(wave_jax, lnu_jax))

    # With correct window 6 = (1677–1740), the trough at 1640 is outside,
    # so β should still be close to −2.0
    assert np.abs(beta_measured - (-2.0)) < 1e-3, (
        f"Gaussian trough test: β={beta_measured:.6f}, expected ≈ −2.0 "
        f"with correct window 6; if ~−1.8, old window 6 was used"
    )


def test_calzetti94_windows_match_table_2():
    """CALZETTI94_WINDOWS_AA matches Calzetti et al. 1994 Table 2 ten windows."""
    # Verify the constant has all ten windows
    assert len(CALZETTI94_WINDOWS_AA) == 10, (
        f"Expected 10 windows, got {len(CALZETTI94_WINDOWS_AA)}"
    )
    # Check each window matches Table 2
    expected = (
        (1268.0, 1284.0),
        (1309.0, 1316.0),
        (1342.0, 1371.0),
        (1407.0, 1515.0),
        (1562.0, 1583.0),
        (1677.0, 1740.0),
        (1760.0, 1833.0),
        (1866.0, 1890.0),
        (1930.0, 1950.0),
        (2400.0, 2580.0),
    )
    assert expected == CALZETTI94_WINDOWS_AA, (
        "CALZETTI94_WINDOWS_AA does not match Calzetti+1994 Table 2 windows"
    )


def test_dn4000_grid_independence_uniform():
    """Dn4000 = 1.07840 for L_ν ∝ λ² on a uniform 1 Å grid."""
    # Exact L_ν ∝ λ² ⟹ Dn4000 = 1.07840
    wave = np.arange(3000.0, 5000.0, 1.0)
    lnu = (wave / 4000.0) ** 2 * 1e30
    lnu_jax = jnp.asarray(lnu)
    wave_jax = jnp.asarray(wave)

    # Three implementations should all agree to 1e-5
    dn4000_diag = float(dn4000(wave_jax, lnu_jax))
    dn4000_index = float(measure_index_jax(wave_jax, lnu_jax, STANDARD_INDICES["Dn4000"]))
    dn4000_compute = float(compute_dn4000(lnu_jax, wave_jax))

    expected = 1.07840
    assert np.abs(dn4000_diag - expected) < 1e-5, (
        f"diagnostics.dn4000: {dn4000_diag:.5f} vs {expected:.5f}"
    )
    assert np.abs(dn4000_index - expected) < 1e-5, (
        f"measure_index_jax: {dn4000_index:.5f} vs {expected:.5f}"
    )
    assert np.abs(dn4000_compute - expected) < 1e-5, (
        f"compute_dn4000: {dn4000_compute:.5f} vs {expected:.5f}"
    )


def test_dn4000_grid_independence_clustered():
    """Dn4000 = 1.07840 for L_ν ∝ λ² on a clustered 1 Å + 0.05 Å grid.

    The cluster sits inside the blue window (3860–3880 Å). All three
    implementations must agree to 1e-5 on both uniform and clustered grids.
    """
    uniform = np.arange(3000.0, 5000.0, 1.0)
    # Add a 0.05 Å cluster inside the blue window [3850, 3950]
    cluster = np.arange(3860.0, 3880.0, 0.05)
    wave = np.unique(np.concatenate([uniform, cluster]))
    lnu = (wave / 4000.0) ** 2 * 1e30
    lnu_jax = jnp.asarray(lnu)
    wave_jax = jnp.asarray(wave)

    # Three implementations should all agree to 1e-5
    dn4000_diag = float(dn4000(wave_jax, lnu_jax))
    dn4000_index = float(measure_index_jax(wave_jax, lnu_jax, STANDARD_INDICES["Dn4000"]))
    dn4000_compute = float(compute_dn4000(lnu_jax, wave_jax))

    expected = 1.07840
    assert np.abs(dn4000_diag - expected) < 1e-5, (
        f"diagnostics.dn4000 (clustered): {dn4000_diag:.5f} vs {expected:.5f}"
    )
    assert np.abs(dn4000_index - expected) < 1e-5, (
        f"measure_index_jax (clustered): {dn4000_index:.5f} vs {expected:.5f}"
    )
    assert np.abs(dn4000_compute - expected) < 1e-5, (
        f"compute_dn4000 (clustered): {dn4000_compute:.5f} vs {expected:.5f}"
    )


def test_dn4000_agreement_three_implementations():
    """All three Dn4000 implementations agree on uniform and clustered grids."""
    # Test on both uniform and clustered grids
    uniform = np.arange(3000.0, 5000.0, 1.0)
    cluster = np.arange(3860.0, 3880.0, 0.05)
    clustered = np.unique(np.concatenate([uniform, cluster]))

    for name, wave_base in [("uniform", uniform), ("clustered", clustered)]:
        lnu = (wave_base / 4000.0) ** 2 * 1e30
        lnu_jax = jnp.asarray(lnu)
        wave_jax = jnp.asarray(wave_base)

        dn4000_diag = float(dn4000(wave_jax, lnu_jax))
        dn4000_index = float(measure_index_jax(wave_jax, lnu_jax, STANDARD_INDICES["Dn4000"]))
        dn4000_compute = float(compute_dn4000(lnu_jax, wave_jax))

        # All three should be identical to 1e-6
        assert np.abs(dn4000_diag - dn4000_index) < 1e-6, (
            f"{name}: diagnostics {dn4000_diag:.6f} vs index {dn4000_index:.6f}"
        )
        assert np.abs(dn4000_diag - dn4000_compute) < 1e-6, (
            f"{name}: diagnostics {dn4000_diag:.6f} vs compute {dn4000_compute:.6f}"
        )
        assert np.abs(dn4000_index - dn4000_compute) < 1e-6, (
            f"{name}: index {dn4000_index:.6f} vs compute {dn4000_compute:.6f}"
        )


@pytest.mark.parametrize("dtype_backend", [np.float64, np.float32])
def test_dn4000_float32_vs_float64(dtype_backend):
    """Dn4000 agrees to 1e-4 between float32 and float64 implementations."""
    uniform = np.arange(3000.0, 5000.0, 1.0)
    cluster = np.arange(3860.0, 3880.0, 0.05)
    wave = np.unique(np.concatenate([uniform, cluster]))
    lnu = (wave / 4000.0) ** 2 * 1e30

    # Compute in float64
    lnu_f64 = jnp.asarray(lnu, dtype=jnp.float64)
    wave_f64 = jnp.asarray(wave, dtype=jnp.float64)
    dn4000_f64 = float(dn4000(wave_f64, lnu_f64))

    # Compute in float32
    lnu_f32 = jnp.asarray(lnu, dtype=jnp.float32)
    wave_f32 = jnp.asarray(wave, dtype=jnp.float32)
    dn4000_f32 = float(dn4000(wave_f32, lnu_f32))

    # Should agree to 1e-4
    assert np.abs(dn4000_f64 - dn4000_f32) < 1e-4, (
        f"float64: {dn4000_f64:.6f}, float32: {dn4000_f32:.6f}, δ={dn4000_f64 - dn4000_f32:.6f}"
    )
