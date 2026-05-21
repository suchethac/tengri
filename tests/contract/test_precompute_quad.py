"""Tests for the Taylor-corrected photometric precomputation.

Validates:
- ssp_phot_moment (Ψ tensor): shape, finiteness, zero for flat SSP.
- Taylor correction reduces factorization error vs Zacharegkas (n=1).
- Backward compatibility: taylor_correction=False gives Ψ=None.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from tengri.components.stellar.sps.precompute import precompute_photometry

# ── Minimal fake SSP data ─────────────────────────────────────────

pytestmark = pytest.mark.bounds


class _SSPData:
    """Minimal SSP stand-in."""

    def __init__(self, n_met=3, n_age=5, n_wave=200, flat=False):
        self.ssp_wave = np.linspace(3000.0, 10000.0, n_wave)
        if flat:
            self.ssp_flux = np.ones((n_met, n_age, n_wave))
        else:
            # λ^-2 spectrum (varies within filters → nonzero Ψ)
            spec = (self.ssp_wave / 5500.0) ** (-2.0)
            self.ssp_flux = np.broadcast_to(spec, (n_met, n_age, n_wave)).copy()
            # Slight metallicity variation
            self.ssp_flux *= 1.0 + 0.1 * np.arange(n_met)[:, None, None]
        self.ssp_lgmet = np.array([-2.0, -1.0, 0.0])[:n_met]
        self.ssp_lg_age_gyr = np.linspace(-3, 1, n_age)


def _simple_filters(n_filt=3):
    """Synthetic top-hat filters."""
    centers = [4500.0, 6000.0, 8000.0][:n_filt]
    width = 500.0
    waves, trans = [], []
    for c in centers:
        w = np.linspace(c - width, c + width, 60)
        t = np.ones_like(w)
        t[:5] = np.linspace(0, 1, 5)
        t[-5:] = np.linspace(1, 0, 5)
        waves.append(w)
        trans.append(t)
    return waves, trans


# ── Tests ─────────────────────────────────────────────────────────


class TestTaylorMomentTensor:
    """Ψ tensor shape, finiteness, and properties."""

    def test_moment_shape_matches_phot(self):
        """Ψ has the same shape as Φ."""
        ssp = _SSPData()
        fw, ft = _simple_filters()
        pc = precompute_photometry(ssp, fw, ft, 0.1, 1e28, taylor_correction=True)
        assert pc.ssp_phot_moment is not None
        assert pc.ssp_phot_moment.shape == pc.ssp_phot.shape

    def test_moment_finite(self):
        """Ψ contains no NaN or Inf."""
        ssp = _SSPData()
        fw, ft = _simple_filters()
        pc = precompute_photometry(ssp, fw, ft, 0.1, 1e28)
        assert np.all(np.isfinite(np.asarray(pc.ssp_phot_moment)))

    def test_moment_zero_for_flat_ssp(self):
        """For a flat SSP (constant across λ), Ψ = SSP · <λ - λ_eff> = 0."""
        ssp = _SSPData(flat=True)
        fw, ft = _simple_filters()
        pc = precompute_photometry(ssp, fw, ft, 0.1, 1e28)
        # <(λ - λ_eff)> weighted by T·λ is zero by definition of λ_eff
        # ... but only if λ_eff is computed from the same T·λ² / T·λ formula.
        # For top-hat filters this is exact to machine precision.
        assert_allclose(np.asarray(pc.ssp_phot_moment), 0.0, atol=1e-8)

    def test_moment_nonzero_for_steep_ssp(self):
        """For a steep λ^-2 SSP, Ψ ≠ 0."""
        ssp = _SSPData(flat=False)
        fw, ft = _simple_filters()
        pc = precompute_photometry(ssp, fw, ft, 0.1, 1e28)
        moment = np.asarray(pc.ssp_phot_moment)
        assert np.any(np.abs(moment) > 1e-6)

    def test_taylor_disabled_gives_none(self):
        """taylor_correction=False → ssp_phot_moment is None."""
        ssp = _SSPData()
        fw, ft = _simple_filters()
        pc = precompute_photometry(ssp, fw, ft, 0.1, 1e28, taylor_correction=False)
        assert pc.ssp_phot_moment is None


class TestTaylorCorrectionAccuracy:
    """Taylor correction f ≈ A·Φ + A'·Ψ reduces factorization error."""

    @staticmethod
    def _compute_errors(ssp, fw_list, ft_list, redshift=0.1, dl_cm=1e28):
        """Compute Zacharegkas (n=1) and Taylor-corrected errors per filter."""
        pc = precompute_photometry(ssp, fw_list, ft_list, redshift, dl_cm)
        ssp_flux_np = np.asarray(ssp.ssp_flux)
        wave_obs = np.asarray(ssp.ssp_wave) * (1.0 + redshift)

        tau_bc = 1.0

        def dust_fn(lam):
            return np.exp(-tau_bc * (lam / 5500.0) ** -0.7)

        def dust_deriv(lam):
            return dust_fn(lam) * (-tau_bc * (-0.7) / lam) * (lam / 5500.0) ** (-0.7)

        phi = np.asarray(pc.ssp_phot)  # (n_met, n_age, n_filt)
        psi = np.asarray(pc.ssp_phot_moment)
        eff = np.asarray(pc.effective_wavelengths)

        errs_n1 = []
        errs_taylor = []
        for f_idx, (fw, ft) in enumerate(zip(fw_list, ft_list)):
            fw_np, ft_np = np.asarray(fw), np.asarray(ft)
            # Exact: ∫ SSP · A · T · λ dλ / ∫ T · λ dλ
            t_on_ssp = np.interp(wave_obs, fw_np, ft_np, left=0, right=0)
            dust_on_ssp = dust_fn(wave_obs)
            denom = np.trapezoid(t_on_ssp * wave_obs, wave_obs)
            exact = np.trapezoid(
                ssp_flux_np[0, 0] * dust_on_ssp * t_on_ssp * wave_obs, wave_obs
            ) / max(denom, 1e-30)

            lam_eff = float(eff[f_idx])
            A_eff = dust_fn(np.array([lam_eff]))[0]
            Ap_eff = dust_deriv(np.array([lam_eff]))[0]

            approx_n1 = A_eff * phi[0, 0, f_idx]
            approx_taylor = A_eff * phi[0, 0, f_idx] + Ap_eff * psi[0, 0, f_idx]

            errs_n1.append(abs(approx_n1 - exact) / abs(exact) * 100)
            errs_taylor.append(abs(approx_taylor - exact) / abs(exact) * 100)

        return errs_n1, errs_taylor

    def test_taylor_reduces_mean_error(self):
        """Taylor correction has lower mean error than Zacharegkas."""
        ssp = _SSPData()
        fw, ft = _simple_filters()
        errs_n1, errs_taylor = self._compute_errors(ssp, fw, ft)
        assert np.mean(errs_taylor) < np.mean(errs_n1)

    def test_taylor_reduces_max_error(self):
        """Taylor correction has lower max error than Zacharegkas."""
        ssp = _SSPData()
        fw, ft = _simple_filters()
        errs_n1, errs_taylor = self._compute_errors(ssp, fw, ft)
        assert max(errs_taylor) < max(errs_n1)

    def test_taylor_improvement_factor(self):
        """Taylor correction improves by at least 2× on mean error."""
        ssp = _SSPData()
        fw, ft = _simple_filters()
        errs_n1, errs_taylor = self._compute_errors(ssp, fw, ft)
        improvement = np.mean(errs_n1) / max(np.mean(errs_taylor), 1e-10)
        assert improvement > 2.0, f"Only {improvement:.1f}× improvement"
