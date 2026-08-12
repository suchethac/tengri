# SPDX-License-Identifier: BSD-3-Clause
"""Tests for kubota_done_full_agn: full 3-zone K&D disc + two-temperature torus."""

from __future__ import annotations

import chex
import jax
import jax.numpy as jnp
import pytest

from tests._bounds import assert_non_negative

pytestmark = pytest.mark.bounds


@pytest.fixture()
def wavelength():
    """Broad wavelength grid from radio (1 cm) to hard X-ray (1 A)."""
    return jnp.logspace(0, 8, 500)  # 1 A to 10^8 A (= 1 cm)


class TestKubotaDoneFullAgn:
    """Tests for kubota_done_full_agn (K&D 3-zone disc + 2T torus)."""

    def test_finite_nonneg(self, wavelength):
        """kubota_done_full_agn produces finite, non-negative SED."""
        from tengri.components.agn.unified import kubota_done_full_agn

        l_nu = kubota_done_full_agn(wavelength, agn_log_lbol=44.0)
        chex.assert_tree_all_finite(l_nu)
        assert_non_negative(l_nu, name="l_nu")
        chex.assert_equal_shape([l_nu, wavelength])

    def test_registered_as_kubota_done_full(self):
        """'kubota_done_full' resolves via resolve_agn_model."""
        import warnings

        from tengri.components.agn.unified import resolve_agn_model

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            fn = resolve_agn_model("kubota_done_full")
        assert callable(fn)

    def test_agn_frac_scales_linearly(self, wavelength):
        """agn_lum_ratio multiplies the whole SED linearly."""
        from tengri.components.agn.unified import kubota_done_full_agn

        l1 = kubota_done_full_agn(wavelength, agn_log_lbol=44.0, agn_lum_ratio=0.1)
        l2 = kubota_done_full_agn(wavelength, agn_log_lbol=44.0, agn_lum_ratio=0.2)
        significant = l1 > 1e-50
        assert jnp.any(significant), "No significant SED values found"
        ratio = l2[significant] / l1[significant]
        assert jnp.allclose(ratio, 2.0, rtol=0.01)

    def test_has_xray_emission(self, wavelength):
        """Full 3-zone disc produces X-ray emission from the hot corona.

        kubota_done_full includes a hard X-ray power law (hot corona).
        At λ < 100 Å, the full model should have non-negligible flux
        while a torus-only comparison has zero disc contribution.
        """
        from tengri.components.agn.unified import kubota_done_full_agn

        xray_mask = (wavelength > 1.0) & (wavelength < 100.0)
        if not jnp.any(xray_mask):
            pytest.skip("wavelength grid does not cover X-ray regime")

        l_nu = kubota_done_full_agn(
            wavelength,
            agn_log_lbol=44.0,
            agn_lum_ratio=1.0,
            agn_f_hard=0.1,
            agn_torus_frac=0.0,
        )
        assert float(jnp.sum(l_nu[xray_mask])) > 0.0

    def test_f_hard_changes_sed_shape(self, wavelength):
        """Changing agn_f_hard from 0 to 0.1 alters the SED.

        With fixed L_bol, increasing f_hard routes more power to the
        corona power law and less to the disc. The two SEDs must differ.
        """
        from tengri.components.agn.unified import kubota_done_full_agn

        # Physical L_bol (log10 L_sun): L_Edd(1e8 M_sun) ≈ 10^12.5 L_sun, so
        # agn_log_lbol=12 is sub-Eddington. Previously 44 (super-Eddington),
        # which now clips at the Eddington limit under the luminosity-first
        # parameterization (ADR-0020) and saturates the f_hard effect.
        l_no_corona = kubota_done_full_agn(
            wavelength, agn_log_lbol=12.0, agn_lum_ratio=1.0, agn_f_hard=0.0, agn_torus_frac=0.0
        )
        l_corona = kubota_done_full_agn(
            wavelength, agn_log_lbol=12.0, agn_lum_ratio=1.0, agn_f_hard=0.1, agn_torus_frac=0.0
        )
        # The SEDs must differ somewhere (not identical arrays)
        assert not jnp.allclose(l_corona, l_no_corona, rtol=1e-6)

    def test_jit_compatible(self, wavelength):
        """kubota_done_full_agn is JIT-compilable."""
        from tengri.components.agn.unified import kubota_done_full_agn

        @jax.jit
        def _run(wave):
            return kubota_done_full_agn(wave, agn_log_lbol=44.0)

        chex.assert_tree_all_finite(_run(wavelength))


@pytest.mark.regression_bug
class TestHotCoronaSeedRollover:
    """Regression: the K&D hot corona must not leak into the infrared/radio.

    The optically-thin hot corona is a thermal-Comptonization spectrum,
    L_nu ~ nu^(1-Gamma) * exp(-h*nu/kT_e) * exp(-nu_seed/nu), defined only
    between its seed-photon energy and the electron temperature (Kubota & Done
    2018, MNRAS 480, 1247, Section 2.2). Without the low-energy seed rollover
    the bare nu^(1-Gamma) tail (Gamma ~ 1.8) rose monotonically toward low
    frequency, so the disc climbed ~400x from the UV into the radio instead of
    falling as Rayleigh-Jeans.
    """

    def test_disc_falls_from_optical_into_radio(self, wavelength):
        """L_nu must decrease monotonically from the optical to the radio."""
        from tengri.components.agn.disc import kubota_done_disc

        wave = jnp.asarray([5000.0, 1e4, 1e5, 1e6, 1e7, 1e8])  # 5000 A -> 1 cm
        l_nu = kubota_done_disc(wave, agn_log_lbol=12.22)
        chex.assert_tree_all_finite(l_nu)
        # Strictly decreasing from the optical peak down into the radio.
        assert jnp.all(jnp.diff(l_nu) < 0.0), f"disc does not fall toward radio: {l_nu}"
        # The radio must sit far below the optical (was ~450x ABOVE before the fix).
        assert float(l_nu[-1] / l_nu[0]) < 1e-3

    def test_far_tail_is_rayleigh_jeans(self):
        """The far-IR/radio tail follows the multicolor-disc Rayleigh-Jeans law.

        For L_nu ∝ nu^2 (∝ lambda^-2), d(log L_nu)/d(log lambda) = -2.
        """
        from tengri.components.agn.disc import kubota_done_disc

        wave = jnp.asarray([1e7, 1e8])  # 1 mm -> 1 cm
        l_nu = kubota_done_disc(wave, agn_log_lbol=12.22)
        slope = float(jnp.log(l_nu[1] / l_nu[0]) / jnp.log(wave[1] / wave[0]))
        assert -2.3 < slope < -1.7, f"far tail slope {slope:.2f} is not Rayleigh-Jeans (~-2)"

    def test_xray_corona_preserved(self, wavelength):
        """The seed rollover must not touch the validated X-ray spectrum.

        The rollover exp(-nu_seed/nu) -> 1 for nu >> nu_seed, so hard X-rays are
        unchanged: the corona must remain the brightest part of the disc SED in
        the X-ray relative to the radio.
        """
        from tengri.components.agn.disc import kubota_done_disc

        wave = jnp.asarray([1.0, 1e8])  # 1 A (hard X-ray) vs 1 cm (radio)
        l_nu = kubota_done_disc(wave, agn_log_lbol=12.22)
        chex.assert_tree_all_finite(l_nu)
        assert float(l_nu[0]) > float(l_nu[1]), "X-ray corona flux lost relative to radio"

    def test_preintegrated_matches_full_wavelength_uv(self):
        """Preintegrated photometry must track the (now-corrected) full path.

        The seed rollover is wavelength-dependent within a band, so it is baked
        into the corona lookup table along a seed-temperature axis. A UV filter
        straddling the rollover knee is the sensitive case.
        """
        import numpy as np

        from tengri.components.agn.disc import kubota_done_disc
        from tengri.components.agn.kd_precompute import (
            kubota_done_disc_preintegrated,
            preintegrate_kd_components,
        )

        # A narrow UV bandpass near the seed-rollover knee (~2500 A).
        center = 2500.0
        fw = [np.linspace(center - 200.0, center + 200.0, 64)]
        ft = [np.exp(-0.5 * ((fw[0] - center) / 80.0) ** 2)]
        z = 0.0
        kd = preintegrate_kd_components(fw, ft, redshift=z)

        kw = dict(
            agn_log_lbol=12.22,
            agn_log_mbh=8.0,
            agn_log_ledd=-1.0,
            agn_a_spin=0.0,
            agn_cos_inc=0.5,
        )
        preint = kubota_done_disc_preintegrated(kd, **kw)[0]

        # Full-wavelength band flux through the same filter.
        wave = jnp.asarray(fw[0])
        l_nu_full = kubota_done_disc(wave, **kw)
        trans = jnp.asarray(ft[0])
        full = float(
            jnp.trapezoid(l_nu_full * trans / wave, wave) / jnp.trapezoid(trans / wave, wave)
        )

        rel_err = abs(float(preint) - full) / full
        assert rel_err < 0.12, f"preint {float(preint):.3e} vs full {full:.3e}, err {rel_err:.1%}"


@pytest.mark.regression_bug
class TestSelfGravityRadiusQsosedConvention:
    """The outer disc radius must follow the canonical qsosed normalization.

    The Laor & Netzer (1989) self-gravity radius in qsosed (Quera-Bofarull,
    ``Sed.gravity_radius``) is

        r_sg / R_g = 2150 * (M_BH / 10^9 M_sun)^{-2/9}
                          * mdot^{4/9} * (alpha/0.1)^{2/9},

    with the mass normalized to 10^9 M_sun. A prior tengri version normalized
    by 10^8 M_sun, making r_sg a factor 10^{2/9} ~ 1.67 too small at every
    mass: it truncated the coolest outer annuli and produced a near-IR disc
    tail ~30% below the AGNfitter-rX KD18 reference. This pins the convention.
    """

    def test_matches_qsosed_gravity_radius_formula(self):
        """``_self_gravity_radius`` reproduces the qsosed 10^9-M_sun form."""
        import numpy as np

        from tengri.components.agn.disc import _self_gravity_radius

        for log_mbh, lam, alpha in [
            (8.0, 0.5, 0.1),
            (9.0, 0.1, 0.1),
            (7.0, 1.0, 0.05),
            (8.5, 0.3, 0.2),
        ]:
            mass = 10.0**log_mbh / 1.0e9  # qsosed convention: M / 10^9 Msun
            expected = (
                2150.0 * mass ** (-2.0 / 9.0) * lam ** (4.0 / 9.0) * (alpha / 0.1) ** (2.0 / 9.0)
            )
            got = float(_self_gravity_radius(log_mbh, lam, alpha))
            assert np.isclose(got, expected, rtol=1e-6), (
                f"log_mbh={log_mbh} lam={lam} alpha={alpha}: "
                f"got {got:.4f} R_g, expected {expected:.4f} R_g"
            )

    def test_not_the_buggy_1e8_normalization(self):
        """Guard against regressing to the 10^8-M_sun normalization."""
        import numpy as np

        from tengri.components.agn.disc import _self_gravity_radius

        got = float(_self_gravity_radius(8.0, 0.5, 0.1))
        buggy_1e8 = 2150.0 * (10.0**8.0 / 1.0e8) ** (-2.0 / 9.0) * 0.5 ** (4.0 / 9.0)
        # The correct value is a factor 10^{2/9} ~ 1.67 larger than the bug.
        assert got > buggy_1e8 * 1.6, (
            f"r_sg {got:.1f} R_g looks like the 1e8 bug ({buggy_1e8:.1f})"
        )
        assert np.isclose(got / buggy_1e8, 10.0 ** (2.0 / 9.0), rtol=1e-6)
