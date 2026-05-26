# SPDX-License-Identifier: BSD-3-Clause
"""Systematic tests of dust attenuation curve normalizations and features.

Every registered dust attenuation curve must satisfy k(5500 Å) ≈ 1.
"""

from __future__ import annotations

from typing import ClassVar

import chex
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.bounds


class TestDustAttenuation:
    """All attenuation curves are normalised so k(5500 Å) = 1."""

    WL = jnp.array([5500.0, 2175.0, 1500.0, 8000.0])  # V, UV bump, FUV, I-band

    def test_calzetti_values(self):
        """Calzetti+2000: k(2175)/k(V)≈2.09, k(1500)/k(V)≈2.55, k(8000)/k(V)≈0.63."""
        from tengri.components.dust.attenuation import calzetti

        k = np.array(calzetti(self.WL))
        assert abs(k[0] - 1.0) < 0.01
        assert 2.0 < k[1] < 2.2  # mild feature, no strong UV bump
        assert 2.4 < k[2] < 2.7
        assert 0.55 < k[3] < 0.70

    def test_cardelli_has_uv_bump(self):
        """Cardelli+1989 MW: strong 2175 Å feature, k(2175)/k(V) ≈ 3.2 for R_V=3.1."""
        from tengri.components.dust.attenuation import cardelli

        k = np.array(cardelli(self.WL, dust_Rv=3.1))
        assert 3.0 < k[1] < 3.4, f"MW UV bump k(2175)={k[1]:.2f}, expected 3.2"
        # Bump must exceed linear interpolation between neighbour wavelengths
        wl2 = jnp.array([1900.0, 2175.0, 2450.0])
        k2 = np.array(cardelli(wl2, dust_Rv=3.1))
        baseline = 0.5 * (k2[0] + k2[2])
        assert k2[1] > baseline * 1.05, "Cardelli UV bump missing/too weak"

    def test_smc_steep_uv_no_bump(self):
        """SMC: k(1500)/k(V)≈4.6, 2175 Å bump absent (feature <10%)."""
        from tengri.components.dust.attenuation import smc

        k = np.array(smc(self.WL))
        assert k[2] > 4.0, f"SMC k(1500) = {k[2]:.2f}, expected ≈4.6"
        # Absence of bump: k(2175) should be close to linear interp
        wl2 = jnp.array([1900.0, 2175.0, 2450.0])
        k2 = np.array(smc(wl2))
        baseline = 0.5 * (k2[0] + k2[2])
        assert abs(k2[1] - baseline) / baseline < 0.1, "SMC should not have UV bump"

    def test_salim_collapses_to_calzetti_at_delta_zero(self):
        """Salim+2018: δ=0 and bump=1.0 recovers Calzetti within 1%."""
        from tengri.components.dust.attenuation import calzetti, salim

        ks = np.array(salim(self.WL, dust_delta=0.0, dust_uv_bump=1.0))
        kc = np.array(calzetti(self.WL))
        np.testing.assert_allclose(ks, kc, rtol=0.02)


class TestDustLawCombinations:
    """Systematic test of every registered dust attenuation curve.

    Convention: all curves should satisfy k(5500 Å) ≈ 1 so that A(λ) = k(λ)·A_V.
    k(FUV)/k(V) must be in [1, 15], k(I)/k(V) in [0.2, 2] for any realistic law.
    """

    WL: ClassVar = jnp.array([1500.0, 2175.0, 3000.0, 5500.0, 9000.0])
    _REQUIRES_RV: ClassVar[set[str]] = {"cardelli", "conroy2010", "d03_mwrv31"}

    @pytest.mark.parametrize(
        "name",
        [
            "calzetti",
            "cardelli",
            "conroy2010",
            "d03_mwrv31",
            "hd23_mwrv31",
            "kriek_conroy",
            "leitherer02",
            "li08",
            "lmc",
            "narayanan_z",
            "noll09",
            "power_law",
            "salim",
            "salim_sbl18",
            "smc",
            "tea",
            "vw07_bc",
            "vw07_diff",
            "wd01_mwrv31",
            "wd01_smcbar",
        ],
    )
    def test_dust_law_normalized_at_V_band(self, name):
        """All registered laws (except prevot_smc) normalise to k(V)=1 within 5%."""
        from tengri.components.dust.attenuation import resolve_dust_law

        fn = resolve_dust_law(name)
        kwargs = {"dust_Rv": 3.1} if name in self._REQUIRES_RV else {}
        k = np.array(fn(self.WL, **kwargs))
        chex.assert_tree_all_finite(k)
        assert np.all(k >= 0.0), f"{name}: negative k values"
        assert 0.95 < k[3] < 1.05, f"{name}: k(5500Å) = {k[3]:.3f}"
        assert 1.5 < k[0] < 15.0, f"{name}: k(FUV) = {k[0]:.2f}"
        assert 0.2 < k[4] < 2.0, f"{name}: k(I) = {k[4]:.2f}"

    def test_prevot_smc_normalization(self):
        """prevot_smc should return k(V)=1 per tengri convention."""
        from tengri.components.dust.attenuation import prevot_smc

        k = np.array(prevot_smc(self.WL))
        assert 0.95 < k[3] < 1.05, f"prevot_smc k(V) = {k[3]:.3f}"

    @pytest.mark.parametrize("name", ["cardelli", "conroy2010", "d03_mwrv31", "hd23_mwrv31"])
    def test_milky_way_laws_have_uv_bump(self, name):
        """MW-type curves must show a 2175 Å feature: k(2175) > avg(k(1900), k(2450))."""
        from tengri.components.dust.attenuation import resolve_dust_law

        fn = resolve_dust_law(name)
        wl2 = jnp.array([1900.0, 2175.0, 2450.0])
        kwargs = {"dust_Rv": 3.1} if name in self._REQUIRES_RV else {}
        k = np.array(fn(wl2, **kwargs))
        baseline = 0.5 * (k[0] + k[2])
        assert k[1] > baseline, f"{name}: no UV bump (k(2175)={k[1]:.2f})"
