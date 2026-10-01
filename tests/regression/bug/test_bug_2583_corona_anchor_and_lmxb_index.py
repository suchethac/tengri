# SPDX-License-Identifier: BSD-3-Clause
"""Regression tests for #2583 — corona anchor offset and LMXB Γ default.

Bug 1: The cutoff powerlaw shape spec=(E/E_ref)^(-gamma+1) * exp(-E/E_cut) is
normalised at the wrong anchor point. The shape equals 1 at E_ref=2 keV BEFORE
the cutoff factor, but Yang+2020 Eq. 2 defines L_ν(2 keV) as the monochromatic
luminosity AFTER the exponential cutoff. The spec must be normalised to 1 after
applying the cutoff: spec = (E/E_ref)^(-gamma+1) * exp(-(E-E_ref)/E_cut).

Bug 2: The LMXB photon index defaults to 1.6, but Yang et al. 2020 Sect. 2.2.2
adopt Γ = 1.56 (Fabbiano 2006). PCigale and the literature use 1.56.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.xray.xray import (
    _xray_agn_corona_bolometric,
    xray_agn_corona,
    xray_xrb_terms,
)

pytestmark = pytest.mark.regression_bug


class TestCoronaAnchorAtReferenceEnergy:
    """The shape is normalised to unity at E_ref after the cutoff."""

    def test_anchor_at_2kev_with_scatter_term(self):
        """L_ν(2 keV)/anchor must equal 1.01 (scatter) at log_nh=0."""
        l2500 = 1e29
        # Just+2007 Eq. 3, Yang+2020 Eq. 6
        alpha_ox = -0.137 * np.log10(l2500) + 2.638
        anchor = l2500 * 10.0 ** (alpha_ox / 0.3838)

        HC = 12.398419843  # keV·Angstrom
        w_2kev = HC / 2.0
        for ecut_kev in (50.0, 100.0, 300.0, 500.0, 1000.0):
            lnu2 = float(
                xray_agn_corona(
                    jnp.asarray([w_2kev]), l2500, gamma=1.8, E_cut=ecut_kev, log_nh=0.0
                )[0]
            )
            ratio = lnu2 / anchor
            # With scatter ON (default), ratio should be ~1.01.
            # The shape alone gives 1.0 (fixed after cutoff); scatter adds 1%.
            assert ratio == pytest.approx(1.01, abs=1e-4)

    def test_anchor_independent_of_gamma(self):
        """The anchor point is independent of the photon index Γ."""
        l2500 = 1e29
        alpha_ox = -0.137 * np.log10(l2500) + 2.638
        anchor = l2500 * 10.0 ** (alpha_ox / 0.3838)

        HC = 12.398419843
        w_2kev = HC / 2.0
        for gamma in (1.4, 1.6, 1.8, 2.0, 2.4):
            lnu2 = float(
                xray_agn_corona(
                    jnp.asarray([w_2kev]), l2500, gamma=gamma, E_cut=300.0, log_nh=0.0
                )[0]
            )
            ratio = lnu2 / anchor
            assert ratio == pytest.approx(1.01, abs=1e-4)

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
