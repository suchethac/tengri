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

from tests._dust_laws import every_dust_law, requires_dust_extinction

pytestmark = pytest.mark.bounds

#: Excluded from the k(V)=1 sweep on purpose: prevot_smc does not follow that
#: normalization convention and has its own test below. Stated here rather than
#: left as a gap in a hand-written list, which is how reddy15 went untested.
_NOT_V_NORMALIZED = frozenset({"prevot_smc"})


class TestDustAttenuation:
    """All attenuation curves are normalized so k(5500 Å) = 1."""

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
        # Bump must exceed linear interpolation between neighbor wavelengths
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

    @pytest.mark.parametrize("name", every_dust_law(exclude=_NOT_V_NORMALIZED))
    def test_dust_law_normalized_at_V_band(self, name):
        """All registered laws (except prevot_smc) normalize to k(V)=1 within 5%.

        The list used to be written out here by hand and had 20 entries against
        a 22-entry registry: ``prevot_smc`` on purpose (see
        ``_NOT_V_NORMALIZED``) and ``reddy15`` by omission, so reddy15 was
        checked by nothing. Derived from the registry now, so a law added
        tomorrow is swept without editing this file.
        """
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

    @pytest.mark.parametrize(
        "name",
        [
            "cardelli",
            "conroy2010",
            pytest.param("d03_mwrv31", marks=[requires_dust_extinction]),
            pytest.param("hd23_mwrv31", marks=[requires_dust_extinction]),
        ],
    )
    def test_milky_way_laws_have_uv_bump(self, name):
        """MW-type curves must show a 2175 Å feature: k(2175) > avg(k(1900), k(2450))."""
        from tengri.components.dust.attenuation import resolve_dust_law

        fn = resolve_dust_law(name)
        wl2 = jnp.array([1900.0, 2175.0, 2450.0])
        kwargs = {"dust_Rv": 3.1} if name in self._REQUIRES_RV else {}
        k = np.array(fn(wl2, **kwargs))
        baseline = 0.5 * (k[0] + k[2])
        assert k[1] > baseline, f"{name}: no UV bump (k(2175)={k[1]:.2f})"


class TestKriekConroyMatchesFSPS:
    """``kriek_conroy`` must reproduce FSPS ``dust_type=4`` (Prospector's path).

    Prospector applies the Kriek & Conroy (2013) law through FSPS, whose
    ``attn_curve.f90`` (``dust_type=4`` branch) ties the 2175 Å bump
    amplitude to the slope via KC13 Eqn 3 and divides the Drude term by
    R_V = 4.05::

        eb    = 0.85 - 1.9 * dust_index            ! KC13 Eqn 3
        drude = eb*(lam*dlam)**2 / ((lam**2 - lamuvb**2)**2 + (lam*dlam)**2)
        attn  = tauv*(cal00 + drude/4.05)*(lam/lamv)**dust_index

    These tests pin tengri's curve to that reference, normalized to
    k(5500 Å) = 1, so a regression in the bump normalization or the
    slope-bump coupling trips here.
    """

    pytestmark = pytest.mark.regression_paper

    WL: ClassVar = np.logspace(np.log10(1200.0), np.log10(22000.0), 3000)

    @staticmethod
    def _fsps_dust_type4(wl: np.ndarray, delta: float) -> np.ndarray:
        """Matches FSPS ``attn_curve.f90`` ``dust_type=4`` (tauv=1)."""
        x = 1e4 / wl  # 1/micron
        below = wl <= 6300.0
        cal = np.where(
            below,
            1.17 * (-2.156 + 1.509 * x - 0.198 * x**2 + 0.011 * x**3) + 1.78,
            1.17 * (-1.857 + 1.04 * x) + 1.78,
        )
        cal = np.clip(cal, 0.0, None) / 0.44 / 4.05
        eb = 0.85 - 1.9 * delta  # KC13 Eqn 3
        dlam, lamuvb = 350.0, 2175.0
        drude = eb * (wl * dlam) ** 2 / ((wl**2 - lamuvb**2) ** 2 + (wl * dlam) ** 2)
        return (cal + drude / 4.05) * (wl / 5500.0) ** delta

    @staticmethod
    def _bump_excess(wl: np.ndarray, k_norm: np.ndarray) -> float:
        i = int(np.argmin(np.abs(wl - 2175.0)))
        lo = k_norm[int(np.argmin(np.abs(wl - 1950.0)))]
        hi = k_norm[int(np.argmin(np.abs(wl - 2500.0)))]
        return float(k_norm[i] - 0.5 * (lo + hi))

    def _norm(self, k: np.ndarray) -> np.ndarray:
        return k / k[int(np.argmin(np.abs(self.WL - 5500.0)))]

    @pytest.mark.parametrize("delta", [0.0, -0.4, 0.2])
    def test_curve_matches_fsps_dust_type4(self, delta):
        """tengri ``kriek_conroy`` ≈ FSPS ``dust_type=4`` at default coupling."""
        from tengri.components.dust.attenuation import kriek_conroy

        k_t = self._norm(np.asarray(kriek_conroy(self.WL, dust_delta=delta)))
        k_f = self._norm(self._fsps_dust_type4(self.WL, delta))
        # Curves should agree to a few percent across the UV–NIR.
        rel = np.abs(k_t - k_f) / np.maximum(np.abs(k_f), 1e-3)
        assert np.median(rel) < 0.03, (
            f"delta={delta}: median rel diff {np.median(rel):.3f} vs FSPS dust_type=4"
        )

    def test_bump_excess_matches_fsps_at_delta_zero(self):
        """At δ=0 the 2175 Å bump excess (A_λ/A_V) matches FSPS (Eb=0.85)."""
        from tengri.components.dust.attenuation import kriek_conroy

        k_t = self._norm(np.asarray(kriek_conroy(self.WL, dust_delta=0.0)))
        k_f = self._norm(self._fsps_dust_type4(self.WL, 0.0))
        e_t = self._bump_excess(self.WL, k_t)
        e_f = self._bump_excess(self.WL, k_f)
        assert abs(e_t - e_f) < 0.05, (
            f"bump excess tengri={e_t:.3f} vs FSPS={e_f:.3f} (expected ~0.17)"
        )

    def test_bump_couples_to_slope(self):
        """Steeper (more negative δ) ⇒ stronger bump, per KC13 Eqn 3."""
        from tengri.components.dust.attenuation import kriek_conroy

        steep = self._bump_excess(
            self.WL, self._norm(np.asarray(kriek_conroy(self.WL, dust_delta=-0.4)))
        )
        flat = self._bump_excess(
            self.WL, self._norm(np.asarray(kriek_conroy(self.WL, dust_delta=0.2)))
        )
        assert steep > flat, f"bump should grow as δ decreases: steep={steep:.3f}, flat={flat:.3f}"
