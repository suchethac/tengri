# SPDX-License-Identifier: BSD-3-Clause
"""Regression tests for #2583: corona anchor energy and LMXB photon index.

The cutoff power-law shape of the AGN corona is ``(E/E_ref)^(1-Gamma) *
exp(-(E-E_ref)/E_cut)``, which equals 1 at ``E_ref = 2 keV`` with the cutoff
included: ``L_nu(2 keV)`` is then the monochromatic luminosity of Yang et al. 2020
Eq. 2 for every ``E_cut`` and ``Gamma``. The cells pin that anchor (plus the 1 %
scattered fraction at ``log N_H = 0``) on the ``yang20`` corona, its deprecated
bolometric sibling, and the ``lopez24`` corona's 2-10 keV band luminosity, and pin
the LMXB photon index default at Gamma = 1.56 (Yang et al. 2020 Sect. 2.2.2;
Fabbiano 2006).
"""

import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.xray.xray import (
    _xray_agn_corona_bolometric,
    xray_agn_corona,
    xray_agn_corona_lopez24,
    xray_xrb_terms,
)
from tengri.utils.physics_constants import (
    C_AA as _C_AA,
    H_PLANCK as _H_PLANCK,
    KEV_TO_ERG as _KEV_TO_ERG,
    KEV_TO_HZ as _KEV_TO_HZ,
)

pytestmark = pytest.mark.regression_bug

#: The wavelength the code maps to exactly 2 keV (its own constants, not rounded ones).
_W_2KEV = _C_AA * _H_PLANCK / (2.0 * _KEV_TO_ERG)


class TestCoronaAnchorAtReferenceEnergy:
    """The shape is normalized to unity at E_ref after the cutoff."""

    @pytest.mark.parametrize("gamma", [1.4, 1.8, 2.4])
    @pytest.mark.parametrize("e_cut", [50.0, 100.0, 300.0, 500.0, 1000.0])
    def test_yang20_anchor_at_2kev_with_scatter_term(self, e_cut, gamma):
        """L_nu(2 keV) / (L_2500 * 10**(alpha_ox/0.3838)) = 1.01 at log N_H = 0.

        The anisotropy factor is switched off: its denominator carries
        ``0.13397`` (cos 30 deg to five digits), so at the 30 deg anchor it is
        1 - 2.8e-6, not 1, which is a separate convention from the anchor shape.
        """
        l2500 = 1e29
        alpha_ox = -0.137 * np.log10(l2500) + 2.638  # Just+2007 Eq. 3
        anchor = l2500 * 10.0 ** (alpha_ox / 0.3838)
        lnu2 = float(
            xray_agn_corona(
                jnp.asarray([_W_2KEV]),
                l2500,
                gamma=gamma,
                E_cut=e_cut,
                log_nh=0.0,
                apply_anisotropy=False,
            )[0]
        )
        assert lnu2 / anchor == pytest.approx(1.01, abs=1e-6)

    @pytest.mark.parametrize("gamma", [1.4, 1.8, 2.4])
    @pytest.mark.parametrize("e_cut", [50.0, 100.0, 300.0, 500.0, 1000.0])
    def test_lopez24_band_luminosity_is_1p01_times_nu_lnu_over_alpha_irx(self, e_cut, gamma):
        """The lopez24 corona has no 2 keV anchor: its 2-10 keV integral is
        1.01 * nu L_nu(12 um) / 10**alpha_IRX at log N_H = 0, for every E_cut and Gamma.

        Integrated on the 200-point keV trapezoid grid that defines the amplitude
        (``_cutoff_powerlaw_band_norm``). A 20001-point integral of the same
        spectrum differs by 4e-6 (Gamma = 1.4) to 4e-5 (Gamma = 2.4): the
        trapezoid discretization of the shared band norm, not an anchor error.
        """
        log_l12, alpha_irx = 44.0, 0.3
        energy = np.linspace(2.0, 10.0, 200)  # the emitter's own band-norm grid
        lam = _C_AA * _H_PLANCK / (energy * _KEV_TO_ERG)
        lnu = np.asarray(
            xray_agn_corona_lopez24(
                jnp.asarray(lam),
                log_l12,
                alpha_irx=alpha_irx,
                gamma=gamma,
                E_cut=e_cut,
                log_nh=0.0,
                apply_anisotropy=False,
            )
        )
        band = np.trapezoid(lnu, energy * _KEV_TO_HZ)
        expected = 1.01 * 10.0 ** (log_l12 - alpha_irx)
        assert band / expected == pytest.approx(1.0, abs=1e-6)

    @pytest.mark.parametrize("gamma", [1.4, 1.8, 2.4])
    @pytest.mark.parametrize("e_cut", [50.0, 100.0, 300.0, 500.0, 1000.0])
    def test_lopez24_physical_band_integral_matches_to_discretization(self, e_cut, gamma):
        """The physical 2-10 keV integral (20001-point grid) agrees to 1e-4.

        The residual against the exact integral is the 200-point trapezoid of the
        shared band norm: 4.5e-6 (Gamma = 1.4) to 4e-5 (Gamma = 2.4).
        """
        log_l12, alpha_irx = 44.0, 0.3
        energy = np.linspace(2.0, 10.0, 20001)
        lam = _C_AA * _H_PLANCK / (energy * _KEV_TO_ERG)
        lnu = np.asarray(
            xray_agn_corona_lopez24(
                jnp.asarray(lam),
                log_l12,
                alpha_irx=alpha_irx,
                gamma=gamma,
                E_cut=e_cut,
                log_nh=0.0,
                apply_anisotropy=False,
            )
        )
        band = np.trapezoid(lnu, energy * _KEV_TO_HZ)
        expected = 1.01 * 10.0 ** (log_l12 - alpha_irx)
        assert band / expected == pytest.approx(1.0, abs=1e-4)

    def test_deprecated_bolometric_corona_anchor_at_2kev(self):
        """The deprecated _xray_agn_corona_bolometric has the same anchor shape."""
        l_agn_bol = 1e45  # erg/s
        HC = 12.398419843
        w_2kev = HC / 2.0
        # At log_nh=0 (unobscured), L_ν(2 keV) should equal L_2keV × (1 + scattered_frac).
        # L_2keV is computed from L_bol via Hopkins+2007 bolometric correction.
        lnu2 = float(
            _xray_agn_corona_bolometric(
                jnp.asarray([w_2kev]),
                l_agn_bol,
                gamma=1.8,
                E_cut=300.0,
                log_nh=0.0,
                scattered_frac=0.01,
            )[0]
        )
        # Recompute L_2keV the same way as the function
        _NU_2500 = 1.199e15
        _BC_2500 = 5.15
        l_2500 = l_agn_bol / (_BC_2500 * _NU_2500)
        alpha_ox = -1.4
        l_2kev = l_2500 * 10.0 ** (alpha_ox / 0.3838)
        expected = l_2kev * (1.0 + 0.01)
        # The legacy function converts lambda -> E with 1.6022e-9 erg/keV, which
        # differs from HC by 1.4e-5, hence rel=1e-4 (as in the active-path cell).
        assert lnu2 == pytest.approx(expected, rel=1e-4)


class TestLMXBPhotonIndexDefault:
    """The LMXB Γ default must be 1.56, matching Yang+2020 Sect. 2.2.2."""

    def test_lmxb_band_ratio_with_gamma_1p56_fixed(self):
        """The LMXB 0.5–2 / 2–10 keV band ratio for Γ=1.56 is ~0.46."""
        # The exact formula depends on the normalization of xray_xrb_terms.
        # We verify empirically that with the fixed default Γ=1.56, the ratio is
        # reproduced at the known value from Yang+2020 calibration (0.4618–0.4625).
        gamma = 1.56
        HC = 12.398419843
        w = HC / np.linspace(0.3, 12.0, 4000)  # wavelength grid
        lmxb_lnu = np.asarray(
            xray_xrb_terms(
                jnp.asarray(w),
                sfr=1.0,
                stellar_mass=1e10,
                metallicity_z=0.02,
                stellar_age_gyr=3.0,
                gamma_lmxb=gamma,
            )["lmxb"]
        )

        def band_integral(lnu, e1, e2, w=w):
            E_keV = HC / w
            mask = (E_keV >= e1) & (E_keV <= e2)
            nu = 2.99792458e18 / w[mask]
            order = np.argsort(nu)
            return np.trapezoid(lnu[mask][order], nu[order])

        L_soft = band_integral(lmxb_lnu, 0.5, 2.0)
        L_hard = band_integral(lmxb_lnu, 2.0, 10.0)
        measured_ratio = L_soft / L_hard

        # Yang+2020 / issue reproducer gives 0.4625 for Γ=1.56
        assert measured_ratio == pytest.approx(0.4625, abs=1e-3)

    def test_default_gamma_lmxb_is_1p56(self):
        """The declared default for xray_gamma_lmxb is 1.56."""
        import inspect

        from tengri.components.xray.xray import xray_xrb_terms

        sig = inspect.signature(xray_xrb_terms)
        default_gamma = sig.parameters["gamma_lmxb"].default
        assert default_gamma == 1.56
