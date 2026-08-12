# SPDX-License-Identifier: BSD-3-Clause
"""Faithful Mahadevan 1997 ADAF model — validated against the primary source.

Every scaling relation and coefficient here is checked against Mahadevan 1997
(ApJ 477, 585; astro-ph/9609107). Rewrite tracking: #898.

Foundation layer (this batch): the differentiable modified Bessel ``K_2`` and the
relativistic Maxwellian factor ``g(theta_e)`` (Eq. 11), verified against SciPy.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import kve as scipy_kve

from tests._grad_parity import assert_grad_matches_fd

jax.config.update("jax_enable_x64", True)

pytestmark = pytest.mark.regression_paper


class TestBesselK2:
    """The differentiable K_2 must match SciPy across the ADAF temperature range."""

    def test_k2e_matches_scipy(self):
        from tengri.components.agn.adaf import _bessel_k2e

        # x = 1/theta_e; ADAF T_e ~ 5e8-1e10 K -> theta_e ~ 0.08-1.7 -> x ~ 0.6-12.
        x = np.array([0.6, 1.0, 2.0, 2.97, 4.0, 6.0, 10.0, 12.0])
        got = np.asarray(_bessel_k2e(x))  # exponentially scaled: K_2(x) * e^x
        ref = scipy_kve(2.0, x)
        rel = np.max(np.abs(got - ref) / ref)
        assert rel < 1e-4, f"K_2 e^x rel error {rel:.2e} vs scipy"

    def test_k2e_is_differentiable(self):
        from tengri.components.agn.adaf import _bessel_k2e

        g = assert_grad_matches_fd(lambda t: _bessel_k2e(t), 3.0)
        assert np.isfinite(float(g))


class TestGTheta:
    """g(theta_e) (Mahadevan Eq. 11) — the relativistic Maxwellian heating factor."""

    def _g_ref(self, t_e):
        # Direct scipy evaluation of Eq. 11 for cross-check.
        from tengri.components.agn._phys import C_LIGHT
        from tengri.utils.physics_constants import K_BOLTZ, M_ELECTRON

        theta = K_BOLTZ * t_e / (M_ELECTRON * C_LIGHT**2)
        x = 1.0 / theta
        k2 = scipy_kve(2.0, x) * np.exp(-x)  # unscaled K_2
        return (2.0 + 2.0 * theta + 1.0 / theta) * np.exp(-x) / k2

    def test_g_theta_matches_eq11(self):
        from tengri.components.agn.adaf import _adaf_g_theta

        for t_e in (5e8, 1e9, 1.5e9, 2e9, 5e9):
            got = float(_adaf_g_theta(t_e))
            ref = self._g_ref(t_e)
            assert abs(got - ref) / ref < 1e-4, f"g(theta_e) at T_e={t_e:.0e}: {got} vs {ref}"

    def test_g_theta_paper_value(self):
        """Paper (p.19): at very high mdot, T_e ~ 1.5e9 K gives g(theta_e) ~ 7."""
        from tengri.components.agn.adaf import _adaf_g_theta

        assert 6.5 < float(_adaf_g_theta(1.5e9)) < 7.5


class TestSelfSimilar:
    """Self-similar flow quantities at r_min (Mahadevan Eq. 5)."""

    def test_ne_b_at_rmin(self):
        """n_e, B at r_min, Mahadevan 1997 Eq. 5 (p.4), at the Fig. 1 parameters
        m=5e9, mdot=1e-3, alpha=0.3, beta=0.5.

        Verification trail (values are Eq. 5 evaluated by hand at r_min=3, c1=0.5,
        c3=0.3 — the paper gives no worked table, so we pin the closed form):

            b1  = 3.16e19 * alpha^-1 * c1^-1                       = 2.107e20
            n_e = b1 * m^-1 * mdot * r_min^-3/2                    = 8.11e6  cm^-3
            s1  = 1.42e9 * alpha^-1/2 (1-beta)^1/2 c1^-1/2 c3^1/2  = 1.420e9
            B   = s1 * m^-1/2 * mdot^1/2 * r_min^-5/4              = 160.8   G

        CONVENTION NOTE: these are the *Mahadevan 1997* (spherical-accretion)
        constants, which the paper states explicitly "differ from Narayan & Yi
        (1995b)" — Eq. 1's footnote carries a factor 1/3 for a 3-D tangled field,
        so the NY95b B-normalization (~6.55e8 with a c3^1/4) does NOT apply here.
        We implement Mahadevan 1997 Eq. 5, not NY95b.
        """
        from tengri.components.agn.adaf import _adaf_ne_b_rmin

        n_e, B = _adaf_ne_b_rmin(m=5e9, mdot=1e-3, alpha=0.3, beta=0.5)
        assert abs(float(n_e) - 8.11e6) / 8.11e6 < 0.02, f"n_e={float(n_e):.3e}"
        assert abs(float(B) - 160.8) / 160.8 < 0.02, f"B={float(B):.3e}"


class TestTauEsAlphaC:
    """Electron-scattering optical depth (Eq. 31) and Compton slope (Eq. 34)."""

    def test_tau_es_fiducial(self):
        """Mahadevan 1997 Eq. 31 (p.12): tau_es = 6.2 alpha^-1 c1^-1 mdot r_min^-1/2,
        which reduces to 23.87*mdot at alpha=0.3 (c1=0.5, r_min=3) — the paper's
        own quoted normalization. This is HALF the total electron-scattering depth
        of Narayan & Yi (1995b): the paper takes the mean photon to see half the
        total depth ("we therefore take the optical depth ... to be half of that
        as given in Narayan & Yi 1995b")."""
        from tengri.components.agn.adaf import _adaf_tau_es

        assert abs(float(_adaf_tau_es(mdot=1e-2, alpha=0.3)) - 0.2387) / 0.2387 < 1e-3
        # scales linearly in mdot and as 1/alpha
        assert abs(float(_adaf_tau_es(mdot=1e-2, alpha=0.6)) - 0.1194) / 0.1194 < 1e-3

    def test_alpha_c_eq34(self):
        """Eq. 34: alpha_c = -ln(tau_es)/ln(A), A = 1 + 4 theta_e + 16 theta_e^2."""
        from tengri.components.agn.adaf import _adaf_alpha_c

        # T_e=2e9 -> theta_e~0.337 -> A~4.17 -> ln A~1.43; tau_es=0.024 -> -ln~3.73
        ac = float(_adaf_alpha_c(tau_es=0.024, t_e=2e9))
        assert 2.4 < ac < 2.9, f"alpha_c={ac}"
        # Larger tau_es (higher mdot) -> smaller alpha_c (Compton cooling grows)
        assert float(_adaf_alpha_c(tau_es=0.24, t_e=2e9)) < ac


class TestXmSolver:
    """Synchrotron self-absorption parameter x_M (Mahadevan Eq. 20)."""

    def test_x_m_root(self):
        """Solve Eq. 20 for m=5e9, mdot=1e-3, alpha=0.3, beta=0.5 at T_e=2e9.

        Hand analysis (4 pi n_e R / B ~ 2.8e21, theta_e~0.34) gives x_M ~ 2000.
        """
        from tengri.components.agn.adaf import _adaf_x_m

        x_m = float(_adaf_x_m(t_e=2e9, m=5e9, mdot=1e-3, alpha=0.3, beta=0.5))
        assert 1000.0 < x_m < 4000.0, f"x_M={x_m:.1f}"

    def test_x_m_satisfies_eq20(self):
        """The returned x_M must actually satisfy the transcendental Eq. 20."""
        import numpy as np
        from scipy.special import kve

        from tengri.components.agn.adaf import _adaf_ne_b_rmin, _adaf_x_m

        m, mdot, alpha, beta, t_e = 5e9, 1e-3, 0.3, 0.5, 2e9
        x_m = float(_adaf_x_m(t_e=t_e, m=m, mdot=mdot, alpha=alpha, beta=beta))
        n_e, B = (float(v) for v in _adaf_ne_b_rmin(m, mdot, alpha, beta))
        from tengri.components.agn._phys import C_LIGHT
        from tengri.utils.physics_constants import K_BOLTZ, M_ELECTRON

        theta = K_BOLTZ * t_e / (M_ELECTRON * C_LIGHT**2)
        R = 3.0 * 2.95e5 * m
        k2 = kve(2.0, 1.0 / theta) * np.exp(-1.0 / theta)
        lhs = np.exp(1.8899 * x_m ** (1.0 / 3.0))
        rhs = (
            2.49e-10
            * (4 * np.pi * n_e * R / B)
            / (theta**3 * k2)
            * (x_m ** (-7.0 / 6.0) + 0.40 * x_m ** (-17.0 / 12.0) + 0.5316 * x_m ** (-5.0 / 3.0))
        )
        assert abs(lhs - rhs) / rhs < 1e-3, f"Eq20 residual: lhs={lhs:.3e} rhs={rhs:.3e}"


class TestElectronTemperature:
    """Self-consistent equilibrium electron temperature (Mahadevan Eqs. 40/43)."""

    def test_te_high_mdot_paper_anchor(self):
        """Paper p.15/17: at high mdot (~1e-2) the equilibrium T_e ~ 2e9 K,
        independent of black-hole mass."""
        from tengri.components.agn.adaf import _adaf_electron_temperature

        t_e = float(_adaf_electron_temperature(m=1e8, mdot=1e-2, alpha=0.3, beta=0.5, delta=1e-3))
        assert 1.5e9 < t_e < 3.5e9, f"T_e(mdot=1e-2)={t_e:.3e}"

    def test_te_in_physical_range_over_mdot(self):
        """T_e stays in ~[1e9, 1e10] K across the ADAF mdot range (paper Fig. 2)."""
        from tengri.components.agn.adaf import _adaf_electron_temperature

        for mdot in (1e-4, 1e-3, 3e-3, 1e-2):
            t_e = float(
                _adaf_electron_temperature(m=5e9, mdot=mdot, alpha=0.3, beta=0.5, delta=1e-3)
            )
            assert 5e8 < t_e < 2e10, f"T_e(mdot={mdot})={t_e:.3e}"

    def test_te_mass_insensitivity(self):
        """Paper: at high mdot T_e is 'fairly insensitive to the mass'."""
        from tengri.components.agn.adaf import _adaf_electron_temperature

        t_lo = float(_adaf_electron_temperature(m=1e6, mdot=1e-2, alpha=0.3, beta=0.5, delta=1e-3))
        t_hi = float(
            _adaf_electron_temperature(m=1e10, mdot=1e-2, alpha=0.3, beta=0.5, delta=1e-3)
        )
        assert abs(t_lo - t_hi) / t_hi < 0.5, (
            f"T_e mass-sensitivity too high: {t_lo:.2e} vs {t_hi:.2e}"
        )


def _wide_grid():
    """Radio-to-hard-X-ray rest-frame wavelength grid [Angstrom]."""
    return np.logspace(-3.0, 10.0, 6000)


class TestSpectrumAssembly:
    """End-to-end L_nu spectrum (Mahadevan Eqs. 21-49 assembled).

    Structural anchor to the paper's Fig. 1 (m=5e9, alpha=0.3, beta=0.5): a
    rising nu^{2/5} synchrotron branch to a sub-mm self-absorption peak, a
    Comptonized nu^{-alpha_c} decline, and a bremsstrahlung X-ray tail.
    """

    def test_normalizes_to_lbol(self):
        """The canonical contract: int L_nu dnu = agn_lum_ratio * L_bol (Eq. 49 closes it)."""
        from scipy.integrate import trapezoid

        from tengri.components.agn._phys import wavelength_to_nu
        from tengri.components.agn.adaf import adaf_spectrum
        from tengri.utils.physics_constants import L_SUN

        wave = _wide_grid()
        nu = np.asarray(wavelength_to_nu(jnp.asarray(wave)))
        for log_lbol in (8.5, 9.0, 9.5):
            l_nu = np.asarray(
                adaf_spectrum(jnp.asarray(wave), agn_log_lbol=log_lbol, agn_log_mbh=9.7)
            )
            integ = trapezoid(l_nu[::-1], nu[::-1])
            assert abs(integ / (10**log_lbol * L_SUN) - 1.0) < 0.02, (
                f"int L_nu / L_bol = {integ / (10**log_lbol * L_SUN):.3f}"
            )

    def test_synchrotron_slope_two_fifths(self):
        """Below nu_p: L_nu ~ nu^{2/5} (Mahadevan Eq. 25)."""
        from tengri.components.agn._phys import wavelength_to_nu
        from tengri.components.agn.adaf import (
            _adaf_electron_temperature,
            _adaf_mdot_from_lbol,
            _adaf_nu_peak,
            _adaf_x_m,
            adaf_spectrum,
        )
        from tengri.utils.physics_constants import L_SUN

        m, log_lbol = 5e9, 8.9
        mdot = float(_adaf_mdot_from_lbol(10**log_lbol * L_SUN, m, 0.3, 0.5, 0.1))
        t_e = float(_adaf_electron_temperature(m, mdot, 0.3, 0.5, 0.1))
        x_m = float(_adaf_x_m(t_e, m, mdot, 0.3, 0.5))
        nu_p = float(_adaf_nu_peak(t_e, x_m, m, mdot, 0.3, 0.5))

        wave = _wide_grid()
        nu = np.asarray(wavelength_to_nu(jnp.asarray(wave)))
        l_nu = np.asarray(adaf_spectrum(jnp.asarray(wave), agn_log_lbol=log_lbol, agn_log_mbh=9.7))
        mask = (nu > nu_p / 100) & (nu < nu_p / 3) & (l_nu > 0)
        slope = np.polyfit(np.log10(nu[mask]), np.log10(l_nu[mask]), 1)[0]
        assert abs(slope - 0.40) < 0.05, f"synch slope {slope:.3f} != 2/5"

    def test_compton_slope_minus_alpha_c(self):
        """Above nu_p: L_nu ~ nu^{-alpha_c} (Mahadevan Eq. 38)."""
        from tengri.components.agn._phys import wavelength_to_nu
        from tengri.components.agn.adaf import (
            _adaf_alpha_c,
            _adaf_electron_temperature,
            _adaf_mdot_from_lbol,
            _adaf_nu_peak,
            _adaf_tau_es,
            _adaf_x_m,
            adaf_spectrum,
        )
        from tengri.utils.physics_constants import L_SUN

        m, log_lbol = 5e9, 8.9
        mdot = float(_adaf_mdot_from_lbol(10**log_lbol * L_SUN, m, 0.3, 0.5, 0.1))
        t_e = float(_adaf_electron_temperature(m, mdot, 0.3, 0.5, 0.1))
        x_m = float(_adaf_x_m(t_e, m, mdot, 0.3, 0.5))
        nu_p = float(_adaf_nu_peak(t_e, x_m, m, mdot, 0.3, 0.5))
        alpha_c = float(_adaf_alpha_c(_adaf_tau_es(mdot, 0.3), t_e))

        wave = _wide_grid()
        nu = np.asarray(wavelength_to_nu(jnp.asarray(wave)))
        l_nu = np.asarray(adaf_spectrum(jnp.asarray(wave), agn_log_lbol=log_lbol, agn_log_mbh=9.7))
        mask = (nu > 3 * nu_p) & (nu < 100 * nu_p) & (l_nu > 0)
        slope = np.polyfit(np.log10(nu[mask]), np.log10(l_nu[mask]), 1)[0]
        assert abs(slope - (-alpha_c)) < 0.1, (
            f"Compton slope {slope:.3f} != -alpha_c={-alpha_c:.3f}"
        )

    def test_nu_p_in_submm(self):
        """The synchrotron self-absorption peak sits in the sub-mm/mm (Fig. 1)."""
        from tengri.components.agn.adaf import (
            _adaf_electron_temperature,
            _adaf_mdot_from_lbol,
            _adaf_nu_peak,
            _adaf_x_m,
        )
        from tengri.utils.physics_constants import L_SUN

        m, log_lbol = 5e9, 8.9
        mdot = float(_adaf_mdot_from_lbol(10**log_lbol * L_SUN, m, 0.3, 0.5, 0.1))
        t_e = float(_adaf_electron_temperature(m, mdot, 0.3, 0.5, 0.1))
        x_m = float(_adaf_x_m(t_e, m, mdot, 0.3, 0.5))
        nu_p = float(_adaf_nu_peak(t_e, x_m, m, mdot, 0.3, 0.5))
        assert 5e10 < nu_p < 5e12, f"nu_p={nu_p:.2e} Hz not in sub-mm/mm"

    def test_mdot_derived_from_lbol_and_clipped(self):
        """mdot rises with L_bol (Eq. 49) and is clipped to mdot_crit=0.28 alpha^2 (Eq. 52)."""
        from tengri.components.agn.adaf import _adaf_mdot_from_lbol
        from tengri.utils.physics_constants import L_SUN

        m, alpha = 5e9, 0.3
        md_lo = float(_adaf_mdot_from_lbol(10**8.0 * L_SUN, m, alpha, 0.5, 0.1))
        md_hi = float(_adaf_mdot_from_lbol(10**9.5 * L_SUN, m, alpha, 0.5, 0.1))
        assert md_hi > md_lo, "mdot must increase with L_bol"
        # Absurdly high L_bol -> clipped at the critical rate, not runaway.
        md_max = float(_adaf_mdot_from_lbol(10**20 * L_SUN, m, alpha, 0.5, 0.1))
        assert md_max <= 0.28 * alpha**2 + 1e-9, f"mdot={md_max} exceeds mdot_crit"

    def test_finite_positive(self):
        from tengri.components.agn.adaf import adaf_spectrum

        l_nu = np.asarray(
            adaf_spectrum(jnp.asarray(_wide_grid()), agn_log_lbol=9.0, agn_log_mbh=9.7)
        )
        assert np.all(np.isfinite(l_nu)) and np.all(l_nu >= 0.0)

    def test_jit_and_grad(self):
        """JIT-compiles and is differentiable wrt the canonical luminosity."""
        from tengri.components.agn.adaf import adaf_spectrum

        wave = jnp.asarray(np.logspace(1.0, 8.0, 200))
        fn = jax.jit(lambda ll: adaf_spectrum(wave, agn_log_lbol=ll, agn_log_mbh=9.7).sum())
        val = float(fn(9.0))
        grad = float(jax.grad(fn)(9.0))
        assert np.isfinite(val) and np.isfinite(grad) and grad != 0.0
