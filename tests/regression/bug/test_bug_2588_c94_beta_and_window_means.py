# SPDX-License-Identifier: BSD-3-Clause
"""Regression tests for #2588: C94 pooled hard-window β and Δλ-weighted window means.

- β is ONE least-squares fit of log F_λ against log λ over the pixels inside the
  ten Calzetti et al. (1994) Table 2 windows (Eq. 3); the window bounds are hard.
- Window means are wavelength integrals, ⟨F⟩ = ∫F dλ / ∫dλ, on any grid:
  Dn4000 (Balogh et al. 1999), the equivalent width pseudo-continuum
  (Vollmann & Eversberg 2006) and the index / line-flux window LUT.

Expected values are analytic (power laws, a Gaussian trough on a linear
continuum, F_ν ∝ λ²) or from a numpy fit of the published definition.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import DEFAULT, Fixed, SEDModel, load_ssp_data
from tengri.analysis.diagnostics.spectral import (
    CALZETTI94_WINDOWS_AA,
    dn4000,
    equivalent_width,
    uv_slope_beta,
)
from tengri.observation.spectral_indices import (
    STANDARD_INDICES,
    _window_mean_flux,
    measure_index_jax,
    precompute_index_windows,
    window_ssp_integral,
)
from tengri.utils.sed_quantities import compute_dn4000

pytestmark = pytest.mark.regression_bug

# ── Constants ──────────────────────────────────────────────────────

C_AA = 2.99792458e18  # speed of light in Angstrom/s

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
    over the ten Table-2 windows. Expects agreement to atol=5e-3.
    """
    ssp = load_ssp_data("data/fsps_prsc_miles_chabrier.h5")

    test_cases = [
        (0.0, 0.0, -2.3066),  # tau_V, E_bump, expected C94 β
        (0.5, 3.0, -1.8419),
        (3.0, 1.0, +1.1030),
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

        # The hard-window fit IS the numpy reference (Eq. 3 over the same pixels)
        assert np.abs(beta_measured - beta_ref) < 1e-3, (
            f"tau_V={tau_v} E_b={bump}: {beta_measured:.5f} vs numpy fit {beta_ref:.5f}"
        )

        # Both should match the expected value
        assert np.abs(beta_measured - beta_expected) < 5e-3, (
            f"tau_V={tau_v} E_b={bump}: diagnostics {beta_measured:.3f} "
            f"vs expected {beta_expected:.3f} (δ={beta_measured - beta_expected:.3f})"
        )
        assert np.abs(beta_ref - beta_expected) < 5e-3, (
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


def _clustered_dn4000_grid():
    uniform = np.arange(3000.0, 5000.0, 1.0)
    cluster = np.arange(3860.0, 3880.0, 0.05)
    wave = np.unique(np.concatenate([uniform, cluster]))
    return wave, (wave / 4000.0) ** 2 * 1e30


@pytest.mark.parametrize("dtype_backend", [np.float64, np.float32])
def test_dn4000_float32_vs_float64(dtype_backend):
    """Dn4000 in either dtype agrees with the float64 value to 1e-4."""
    wave, lnu = _clustered_dn4000_grid()
    reference = float(dn4000(jnp.asarray(wave, jnp.float64), jnp.asarray(lnu, jnp.float64)))
    if dtype_backend is np.float32:
        with jax.enable_x64(False):
            w32 = jnp.asarray(wave, dtype=jnp.float32)
            l32 = jnp.asarray(lnu, dtype=jnp.float32)
            assert w32.dtype == jnp.float32
            value = float(dn4000(w32, l32))
    else:
        value = reference
    assert abs(value - reference) < 1e-4, f"{dtype_backend.__name__}: {value} vs {reference}"
    assert abs(value - 1.07840) < 1e-4


def test_uv_slope_beta_float32_vs_float64():
    """β in float32 agrees with float64 to 1e-3 (pooled fit, centered abscissa)."""
    wave = np.arange(1250.0, 2600.0, 1.0)
    lnu = wave ** (-1.7 + 2.0) * (1.0 + 0.1 * np.sin(wave / 37.0))
    b64 = float(uv_slope_beta(jnp.asarray(wave), jnp.asarray(lnu)))
    with jax.enable_x64(False):
        w32 = jnp.asarray(wave, dtype=jnp.float32)
        l32 = jnp.asarray(lnu / lnu.max(), dtype=jnp.float32)
        assert w32.dtype == jnp.float32
        b32 = float(uv_slope_beta(w32, l32))
    assert abs(b32 - b64) < 1e-3, f"β32={b32}, β64={b64}"


def test_uv_slope_beta_hard_windows_ignore_pixels_outside():
    """A narrow trough 3 Å outside window 6 leaves β exactly unchanged (Eq. 3 bounds)."""
    wave = np.arange(1250.0, 2600.0, 0.5)
    lnu = np.ones_like(wave)
    trough = 1.0 - 0.8 * np.exp(-0.5 * ((wave - 1674.0) / 0.6) ** 2)
    b = float(uv_slope_beta(jnp.asarray(wave), jnp.asarray(lnu * trough)))
    assert abs(b + 2.0) < 1e-6, f"β={b}"


# ── Equivalent width: wavelength-integrated pseudo-continuum ──────

EW_CENTER, EW_HALF, EW_SIDE = 5000.0, 20.0, 50.0
EW_AMP, EW_SIGMA, EW_SLOPE = 0.5, 3.0, 0.5


def _ew_analytic() -> float:
    """∫ (F/F_c − 1) dλ over the line window for a linear continuum + Gaussian trough.

    The sidebands are symmetric, so F_c equals the continuum at the line center and
    the odd (slope) terms integrate to zero: EW = −A σ √(2π) erf(h / (σ √2)).
    """
    from math import erf, pi, sqrt

    return -EW_AMP * EW_SIGMA * sqrt(2 * pi) * erf(EW_HALF / (EW_SIGMA * sqrt(2)))


def _ew_grid(cluster_in: str) -> np.ndarray:
    wave = np.arange(4800.0, 5200.0, 1.0)
    if cluster_in == "blue":
        wave = np.concatenate([wave, np.arange(4940.0, 4950.0, 0.02)])
    elif cluster_in == "red":
        wave = np.concatenate([wave, np.arange(5040.0, 5050.0, 0.02)])
    return np.unique(wave)


@pytest.mark.parametrize("cluster_in", ["none", "blue", "red"])
def test_equivalent_width_grid_independent(cluster_in):
    """EW of a Gaussian trough on a sloped F_λ continuum equals its analytic value."""
    wave = _ew_grid(cluster_in)
    x = wave - EW_CENTER
    flam = (1.0 + EW_SLOPE * x / 100.0) * (1.0 - EW_AMP * np.exp(-0.5 * (x / EW_SIGMA) ** 2))
    lnu = flam * wave**2 / C_AA
    ew = float(equivalent_width(jnp.asarray(wave), jnp.asarray(lnu), EW_CENTER, EW_HALF, EW_SIDE))
    assert abs(ew - _ew_analytic()) < 1e-3, f"EW={ew:.5f} vs {_ew_analytic():.5f}"


# ── Index / line window LUT: Δλ-weighted integral and norm ────────


def _nonuniform_ssp(n_met=2, n_age=3):
    wave = np.unique(
        np.concatenate(
            [
                np.arange(3600.0, 6000.0, 5.0),
                np.arange(3860.0, 3900.0, 0.5),
                np.arange(5160.0, 5190.0, 0.25),
            ]
        )
    )
    rng = np.random.default_rng(7)
    base = 1.0 + 0.3 * np.sin(wave / 41.0)
    amp = rng.uniform(0.5, 2.0, size=(n_met, n_age, 1))
    return wave, amp * base[None, None, :] * (
        1.0 + 1e-3 * rng.normal(size=(n_met, n_age, wave.size))
    )


_WINDOWS = [(3850.0, 3950.0), (4000.0, 4100.0), (5160.0, 5192.0), (4847.0, 4876.0)]


@pytest.mark.parametrize("lo,hi", _WINDOWS)
def test_soft_window_integral_matches_exact_window_mean(lo, hi):
    """LUT mean (integral / norm) equals `_window_mean_flux` on a non-uniform grid."""
    wave, flux = _nonuniform_ssp()
    integral, norm = window_ssp_integral(jnp.asarray(wave), jnp.asarray(flux), lo, hi)
    for i in range(flux.shape[0]):
        for j in range(flux.shape[1]):
            exact = float(_window_mean_flux(jnp.asarray(wave), jnp.asarray(flux[i, j]), lo, hi))
            lut = float(integral[i, j] / norm)
            assert abs(lut / exact - 1.0) < 1e-6, f"[{lo},{hi}] ({i},{j}): {lut} vs {exact}"


@pytest.mark.parametrize("name", ["Dn4000", "D4000", "HdA", "Hbeta", "Mgb", "Fe5270"])
def test_precompute_index_windows_matches_exact_means(name):
    """Every window slot of `precompute_index_windows` reproduces the exact window mean."""
    wave, flux = _nonuniform_ssp()
    idx = STANDARD_INDICES[name]
    pre = precompute_index_windows(jnp.asarray(wave), jnp.asarray(flux), [idx])
    windows = list(idx.continuum) + ([idx.feature] if idx.index_type == "EW" else [])
    centers = np.asarray(pre.window_centers)
    for lo, hi in windows:
        k = int(np.argmin(np.abs(centers - 0.5 * (lo + hi))))
        lut = float(pre.window_integrals[0, 0, k] / pre.window_norms[k])
        exact = float(_window_mean_flux(jnp.asarray(wave), jnp.asarray(flux[0, 0]), lo, hi))
        assert abs(lut / exact - 1.0) < 1e-6, f"{name} [{lo},{hi}]: {lut} vs {exact}"


@pytest.mark.parametrize("lo,hi", [(3850.0, 3950.0), (4000.0, 4100.0), (4847.0, 4876.0)])
def test_soft_window_integral_matches_exact_on_miles_grid(lo, hi):
    """On the native MILES SSP grid the LUT and exact window means agree to 1e-8."""
    ssp = load_ssp_data("data/fsps_prsc_miles_chabrier.h5")
    wave = jnp.asarray(ssp.ssp_wave)
    flux = jnp.asarray(ssp.ssp_flux[:2, :3])
    integral, norm = window_ssp_integral(wave, flux, lo, hi)
    for i in range(2):
        for j in range(3):
            exact = float(_window_mean_flux(wave, flux[i, j], lo, hi))
            assert abs(float(integral[i, j] / norm) / exact - 1.0) < 1e-8


def test_dn4000_native_ssp_grid_edge_inclusive():
    """F_ν ∝ λ² sampled on the native SSP grid gives the analytic Dn4000 = 1.07840 to 1e-5.

    ⟨λ²⟩ over [a, b] is (b³ − a³) / (3 (b − a)); the ratio of the red (4000–4100 Å)
    to blue (3850–3950 Å) means is 1.07840.
    """
    ssp = load_ssp_data("data/fsps_prsc_miles_chabrier.h5")
    wave = np.asarray(ssp.ssp_wave)
    lnu = (wave / 4000.0) ** 2 * 1e30
    red = (4100.0**3 - 4000.0**3) / (3 * 100.0)
    blue = (3950.0**3 - 3850.0**3) / (3 * 100.0)
    assert abs(red / blue - 1.07840) < 1e-5
    d = float(dn4000(jnp.asarray(wave), jnp.asarray(lnu)))
    assert abs(d - red / blue) < 1e-5, f"{d:.6f} vs {red / blue:.6f}"


# ── Line-flux window means on the model's own wNE spectrum ─────────

_PC_CM = 3.0856775814913673e18  # parsec [cm]
_LINE_WAVES = np.array([4862.68, 5008.24, 6564.61, 6585.28])
_LINE_NAMES = ("Hbeta", "OIII_5007", "Halpha", "NII_6584")


def _soft_window_mean_numpy(w: np.ndarray, f: np.ndarray, lo: float, hi: float) -> float:
    """⟨F⟩ = ∫F W dλ / ∫W dλ with the 1 Å sigmoid-edge W, trapezoid on the pixel edges."""

    from scipy.special import expit

    weight = expit(w - lo) * expit(hi - w)
    return float(np.trapezoid(f * weight, w) / np.trapezoid(weight, w))


def _line_flux_numpy(w, lnu, line, dl_cm) -> float:
    """Catalog-style line flux of the documented operator from the numpy window means."""
    (blo, bhi), (rlo, rhi) = line.continuum
    flo, fhi = line.feature
    f_blue = _soft_window_mean_numpy(w, lnu, blo, bhi)
    f_red = _soft_window_mean_numpy(w, lnu, rlo, rhi)
    f_feat = _soft_window_mean_numpy(w, lnu, flo, fhi)
    lam_c = 0.5 * (flo + fhi)
    x_b, x_r = 0.5 * (blo + bhi), 0.5 * (rlo + rhi)
    cont = f_blue + (f_red - f_blue) * (lam_c - x_b) / (x_r - x_b)
    return (f_feat - cont) * C_AA / lam_c**2 * (fhi - flo) / (4.0 * np.pi * dl_cm**2)


@pytest.mark.parametrize(
    "backend",
    [
        ("data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5", {"type": "none"}),
        ("data/fsps_prsc_miles_chabrier.h5", {"type": "cue", "all_params": Fixed(DEFAULT)}),
    ],
    ids=["wNE-baked-in", "cue-additive"],
)
def test_measure_line_fluxes_exact_path_is_the_wavelength_integral_mean(backend):
    """Exact-path line fluxes of the wNE model equal the Δλ-weighted window integral.

    The reference is the documented operator written in numpy on the model's own
    rest spectrum: window means ∫F W dλ / ∫W dλ (trapezoid on the wavelength
    differences, 1 Å sigmoid edges), a linear side-band continuum at the feature
    center, and (L_ν − L_ν^cont) (c / λ_c²) Δλ / (4π d_L²) at the 10 pc distance of
    z = 0. Measured agreement is 1e-13 or better; the test allows 1e-10. The
    pixel-count mean differs from the integral by +7.2e-6 (Hβ), −3.2e-6 (OIII 5007),
    +1.1e-5 (Hα) and −1.7e-3 (NII 6584) on the wNE model, and by +4.7e-2, +8.2e-3,
    +5.7e-2 and −2.9e-2 on the Cue model (narrow additive lines on a non-uniform
    grid), so 1e-10 separates the two forms by at least four orders of magnitude.
    """
    from tengri.observation.line_measurement import default_line_defs

    ssp_path, neb = backend
    with jax.enable_x64(True):
        ssp = load_ssp_data(ssp_path)
        model = SEDModel.build(
            ssp_data=ssp,
            sfh={
                "type": "delayed",
                "all_params": Fixed(DEFAULT),
                "log_total_mass": Fixed(10.0),
                "tau_gyr": Fixed(1.0),
                "age_gyr": Fixed(5.0),
            },
            dust_attenuation={
                "type": "two_component",
                "law": "calzetti",
                "all_params": Fixed(DEFAULT),
                "tau_diff": Fixed(0.75),
                "tau_bc": Fixed(0.0),
            },
            neb=neb,
            redshift=Fixed(0.0),
        )
        line_defs = default_line_defs(_LINE_WAVES, _LINE_NAMES)
        measured = np.asarray(model.measure_line_fluxes({}, line_defs, approx=False))
        rest = model._predict_rest_sed({})
        wave = np.asarray(rest.wavelength, dtype=np.float64)
        lnu = np.asarray(rest.sed, dtype=np.float64)

    expected = np.array([_line_flux_numpy(wave, lnu, ld, 10.0 * _PC_CM) for ld in line_defs])
    np.testing.assert_allclose(measured, expected, rtol=1e-10, atol=0.0)
