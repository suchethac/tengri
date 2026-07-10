# SPDX-License-Identifier: BSD-3-Clause
"""Regression tests for #1004 / #1005 — Casey (2012) closure and analytic-emitter grid.

#1004: the casey2012 closure was not Casey 2012 Eq. 1–2: the mid-IR power
law carried a spurious Wien ``exp(-hν/kT)`` factor (e⁻⁴¹ at 10 µm, T=35 K —
component annihilated), its slope was inverted (``ν^+α`` instead of ``λ^α``),
the graybody used the optically-thin ``ν^(3+β)`` limit instead of the paper's
general-opacity form with λ₀ = 200 µm (FIR peak 96 µm instead of ~120 µm at
T = 35 K, β = 1.6), and the turnover misread Casey's b₂ = 6.246 (dimensionless
α coefficient) as 6.246e-3 µm/K. Isolated by the dust-emission parity sweep
against CIGALE 2025.1, whose casey2012 implements the paper verbatim; tengri
DL2007/DL2014 matched CIGALE to 1.000 in every band in the same sweep, so the
defect was local to this closure.

#1005: analytic dust emitters declared no native wavelength grid, so the
master union grid stopped at the SSP edge (160 µm for BC03) — submm flux
silently zero while energy balance re-normalized on the truncated grid.
"""

import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.regression_bug

_C_CGS = 2.99792458e10  # cm/s
_H = 6.62607015e-27  # erg s
_KB = 1.380649e-16  # erg/K


def _casey_reference(wave_aa: np.ndarray, T: float, beta: float, alpha: float) -> np.ndarray:
    """Independent numpy Casey (2012) Eq. 1–2 shape in L_ν.

    Coefficients from the paper (Eqs. 11–12); the same reading as CIGALE's
    ``casey2012.py``: general-opacity graybody with λ₀ = 200 µm, power law
    amplitude tied to the graybody at the turnover λ_c, Gaussian long-λ
    cutoff ``exp(-(λ/λ_c)²)``.
    """
    lam = wave_aa * 1e-8  # cm
    lam0 = 200.0e-4  # 200 µm [cm]
    denom = (26.68 + 6.246 * alpha) ** -2.0 + (1.905e-4 + 7.243e-5 * alpha) * T
    lam_c = (0.75e3 / denom) * 1e-7  # nm → cm
    x = _H * _C_CGS / (lam * _KB * T)
    gray = (1.0 - np.exp(-((lam0 / lam) ** beta))) * (_C_CGS / lam) ** 3 / np.expm1(x)
    x_c = _H * _C_CGS / (lam_c * _KB * T)
    n_pl = (1.0 - np.exp(-((lam0 / lam_c) ** beta))) * (_C_CGS / lam_c) ** 3 / np.expm1(x_c)
    pl = n_pl * (lam / lam_c) ** alpha * np.exp(-((lam / lam_c) ** 2))
    return gray + pl


def _norm_nu(wave_aa: np.ndarray, L: np.ndarray) -> np.ndarray:
    nu = 2.99792458e18 / wave_aa
    order = np.argsort(nu)
    return L / np.trapezoid(L[order], nu[order])


class TestCaseyEq1:
    def test_shape_matches_paper_form(self):
        from tengri.components.dust.emission.analytic._closures import casey2012

        wave = np.geomspace(2.0e4, 1.0e7, 900)  # 2 µm – 1 mm
        L = np.asarray(
            casey2012(jnp.asarray(wave), 1.0, dust_T=35.0, dust_beta_ir=1.6, dust_alpha_mir=2.0)
        )
        ref = _casey_reference(wave, 35.0, 1.6, 2.0)
        L_n, ref_n = _norm_nu(wave, L), _norm_nu(wave, ref)
        # Restrict to λ ≤ 300 µm: tengri additionally applies the da Cunha
        # (2013) CMB contrast, which even at z = 0 suppresses the far
        # Rayleigh–Jeans tail by up to ~0.3% (B_ν(T_CMB)/B_ν(T_dust));
        # the paper-only reference has no CMB term.
        good = (ref_n > ref_n.max() * 1e-6) & (wave <= 3.0e6)
        np.testing.assert_allclose(L_n[good], ref_n[good], rtol=1e-5)

    def test_midir_powerlaw_alive(self):
        """Pre-fix S(10 µm)/S(peak) was ~4e-13; Casey Eq. 1 gives percent level."""
        from tengri.components.dust.emission.analytic._closures import casey2012

        wave = np.geomspace(2.0e4, 1.0e7, 900)
        L = np.asarray(
            casey2012(jnp.asarray(wave), 1.0, dust_T=35.0, dust_beta_ir=1.6, dust_alpha_mir=2.0)
        )
        i10 = int(np.argmin(np.abs(wave - 1.0e5)))
        assert L[i10] / L.max() > 1e-3

    def test_fir_peak_position(self):
        """General-opacity graybody peaks near 120 µm at T=35 K, β=1.6 (was 96)."""
        from tengri.components.dust.emission.analytic._closures import casey2012

        wave = np.geomspace(2.0e4, 1.0e7, 3000)
        L = np.asarray(
            casey2012(jnp.asarray(wave), 1.0, dust_T=35.0, dust_beta_ir=1.6, dust_alpha_mir=2.0)
        )
        peak_um = wave[int(np.argmax(L))] / 1e4
        assert 108.0 < peak_um < 132.0

    def test_optically_thin_variant_switches_graybody_not_powerlaw(self):
        """optically_thin selects the thin graybody limit; it must NOT zero
        the mid-IR power law (the old behavior)."""
        from tengri.components.dust.emission.analytic._closures import casey2012

        wave = np.geomspace(2.0e4, 1.0e7, 900)
        L = np.asarray(
            casey2012(
                jnp.asarray(wave),
                1.0,
                dust_T=35.0,
                dust_beta_ir=1.6,
                dust_alpha_mir=2.0,
                optically_thin=True,
            )
        )
        i10 = int(np.argmin(np.abs(wave - 1.0e5)))
        assert L[i10] / L.max() > 1e-3


class TestAnalyticEmitterGrid:
    """#1005 — analytic emitters must extend the master grid into the submm."""

    def test_native_grid_declared(self):
        from tengri.forward.wavelength_extension import native_wave_dust_emission

        for name in ("casey2012", "modified_blackbody", "pah_drude", "schreiber2016"):
            grid = native_wave_dust_emission(name)
            assert grid is not None, f"{name}: no native grid declared"
            assert grid.max() >= 1.0e7, f"{name}: grid stops at {grid.max():.3g} Å"
            assert grid.min() <= 3.0e4, f"{name}: grid starts at {grid.min():.3g} Å"

    def test_template_models_unchanged(self):
        from tengri.forward.wavelength_extension import native_wave_dust_emission

        assert native_wave_dust_emission(None) is None
        # the bookkeeping pseudo-model stays grid-less
        assert native_wave_dust_emission("energy_balance_split") is None
