# SPDX-License-Identifier: BSD-3-Clause
"""Faithful Mahadevan 1997 ADAF model — validated against the primary source.

Every scaling relation and coefficient here is checked against Mahadevan 1997
(ApJ 477, 585; astro-ph/9609107). Rewrite tracking: #898.

Foundation layer (this batch): the differentiable modified Bessel ``K_2`` and the
relativistic Maxwellian factor ``g(theta_e)`` (Eq. 11), verified against SciPy.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest
from scipy.special import kve as scipy_kve

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

        g = jax.grad(lambda t: _bessel_k2e(t))(3.0)
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
        """n_e, B at r_min for m=5e9, mdot=1e-3, alpha=0.3, beta=0.5 (Fig. 1 params)."""
        from tengri.components.agn.adaf import _adaf_ne_b_rmin

        n_e, B = _adaf_ne_b_rmin(m=5e9, mdot=1e-3, alpha=0.3, beta=0.5)
        # Hand-evaluated from Eq. 5 with c1=0.5, c3=0.3, r_min=3.
        assert abs(float(n_e) - 8.11e6) / 8.11e6 < 0.02, f"n_e={float(n_e):.3e}"
        assert abs(float(B) - 160.8) / 160.8 < 0.02, f"B={float(B):.3e}"


class TestTauEsAlphaC:
    """Electron-scattering optical depth (Eq. 31) and Compton slope (Eq. 34)."""

    def test_tau_es_fiducial(self):
        """Eq. 31: tau_es = 23.87 mdot at alpha=0.3 (c1=0.5, r_min=3)."""
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
