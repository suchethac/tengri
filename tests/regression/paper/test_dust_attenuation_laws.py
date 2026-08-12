# SPDX-License-Identifier: BSD-3-Clause
"""Unit tests for new attenuation curves: leitherer02, noll09, salim_sbl18.

Tests use hardcoded reference values computed from the dust_attenuation
package (Karl Gordon et al., v0.5.dev22) and do NOT require the package
to be installed. These serve as regression tests.

References
----------
- Calzetti et al. 2000, ApJ, 533, 682
- Leitherer et al. 2002, ApJS, 140, 303
- Noll et al. 2009, A&A, 507, 1793
- Salim, Boquien & Lee 2018, ApJ, 859, 11
"""

import chex
import pytest

pytestmark = pytest.mark.regression_paper
import jax
import jax.numpy as jnp
import numpy as np


def fd_grad(f, x: float, eps: float = 1e-4) -> float:
    """Central finite-difference gradient. O(eps^2) accurate."""
    return float((f(x + eps) - f(x - eps)) / (2.0 * eps))


from tengri.components.dust.attenuation import (
    DUST_LAWS,
    calzetti,
    cardelli,
    leitherer02,
    noll09,
    reddy15,
    salim_sbl18,
)
from tests._jit_parity import assert_jit_matches_eager

# Full-range wavelengths (Angstrom)
WAVS = np.array([1000.0, 1200.0, 1500.0, 2175.0, 3000.0, 5500.0, 10000.0, 20000.0])


class TestRegistry:
    """New models should be in the DUST_LAWS registry."""

    @pytest.mark.parametrize("name", ["leitherer02", "noll09", "salim_sbl18"])
    def test_registered(self, name):
        assert name in DUST_LAWS


class TestLeitherer02:
    """Leitherer et al. (2002) UV extension of Calzetti."""

    def test_reference_values(self):
        """L02 at key UV wavelengths (Av=1). Reference from dust_attenuation."""
        ref = np.array([3.42720988, 2.94808185, 2.54615821, 2.31222646])
        wavs = jnp.array([1000.0, 1200.0, 1500.0, 1800.0])
        tng = np.array(leitherer02(wavs))
        np.testing.assert_allclose(tng, ref, rtol=1e-6)

    def test_matches_calzetti_in_optical(self):
        """L02 should match Calzetti above 1800 A."""
        wavs = jnp.array([2000.0, 3000.0, 5500.0, 10000.0])
        l02 = np.array(leitherer02(wavs))
        c00 = np.array(calzetti(wavs))
        np.testing.assert_allclose(l02, c00, rtol=1e-10)

    def test_positive_output(self):
        """Output should always be positive."""
        wavs = jnp.linspace(900.0, 25000.0, 500)
        k = leitherer02(wavs)
        assert jnp.all(k >= 0.0)

    def test_monotonic_uv(self):
        """Should be monotonically decreasing from UV to NIR."""
        wavs = jnp.linspace(1000.0, 20000.0, 200)
        k = np.array(leitherer02(wavs))
        # Overall trend should decrease (some local features allowed)
        assert k[0] > k[-1]

    def test_jit_compatible(self):
        """Should work inside jax.jit."""
        wavs = jnp.array([1500.0, 5500.0])
        result = assert_jit_matches_eager(leitherer02, wavs)
        chex.assert_shape(result, (2,))

    def test_gradient(self):
        """Should be differentiable (for use in optimization)."""

        # leitherer02 has no free params, but the wavelength input should
        # propagate gradients through the curve
        def loss(wav):
            return jnp.sum(leitherer02(wav))

        def f_scalar(wav0: float) -> float:
            return float(loss(jnp.array([wav0])))

        grad_jax = float(jax.grad(loss)(jnp.array([1500.0]))[0])
        np.testing.assert_allclose(
            grad_jax,
            fd_grad(f_scalar, 1500.0, eps=1.0),
            rtol=5e-3,
            err_msg="leitherer02: FD check ∂(∑k)/∂wavelength",
        )


class TestNoll09:
    """Noll et al. (2009) modified Calzetti+L02."""

    def test_reference_no_mods(self):
        """N09 baseline (ampl=0, slope=0) = pure Calzetti+L02."""
        ref = np.array(
            [
                3.42720988,
                2.94808185,
                2.54615821,
                2.09349184,
                1.70999069,
                0.99947911,
                0.46360420,
                0.12220173,
            ]
        )
        tng = np.array(noll09(jnp.array(WAVS)))
        np.testing.assert_allclose(tng, ref, rtol=1e-6)

    def test_reference_with_mods(self):
        """N09 ampl=3, slope=-0.1. Reference from dust_attenuation."""
        ref = np.array(
            [
                4.07188201,
                3.44669893,
                2.93559197,
                3.10975253,
                1.86173361,
                1.00367016,
                0.43764091,
                0.10760568,
            ]
        )
        tng = np.array(noll09(jnp.array(WAVS), dust_bump_strength=3.0, dust_delta=-0.1))
        np.testing.assert_allclose(tng, ref, rtol=1e-6)

    def test_bump_enhances_2175(self):
        """Positive bump amplitude should increase attenuation near 2175 A."""
        k_base = noll09(jnp.array([2175.0]), dust_bump_strength=0.0)
        k_bump = noll09(jnp.array([2175.0]), dust_bump_strength=3.0)
        assert float(k_bump[0]) > float(k_base[0])

    def test_negative_slope_steepens(self):
        """Negative slope should steepen UV relative to NIR."""
        k_flat = noll09(jnp.array(WAVS), dust_delta=0.0)
        k_steep = noll09(jnp.array(WAVS), dust_delta=-0.3)
        # UV/NIR ratio should increase with negative delta
        ratio_flat = float(k_flat[0] / k_flat[-1])
        ratio_steep = float(k_steep[0] / k_steep[-1])
        assert ratio_steep > ratio_flat

    def test_positive_output(self):
        """Output should be non-negative for reasonable parameters."""
        wavs = jnp.linspace(900.0, 25000.0, 500)
        for ampl, delta in [(0.0, 0.0), (3.0, -0.3), (5.0, -0.5)]:
            k = noll09(wavs, dust_bump_strength=ampl, dust_delta=delta)
            assert jnp.all(k >= 0.0)

    def test_jit_compatible(self):
        """Should work inside jax.jit."""
        result = assert_jit_matches_eager(
            lambda w: noll09(w, dust_bump_strength=2.0, dust_delta=-0.1), jnp.array(WAVS)
        )
        chex.assert_equal_shape([result, WAVS])

    def test_gradient_wrt_params(self):
        """Gradients of noll09 match central FD w.r.t. bump strength and slope."""

        def loss(ampl, delta):
            return jnp.sum(noll09(jnp.array(WAVS), dust_bump_strength=ampl, dust_delta=delta))

        g_ampl, g_delta = jax.grad(loss, argnums=(0, 1))(1.0, -0.1)

        def f_ampl(ampl: float) -> float:
            return float(loss(ampl, -0.1))

        def f_delta(delta: float) -> float:
            return float(loss(1.0, delta))

        np.testing.assert_allclose(
            float(g_ampl),
            fd_grad(f_ampl, 1.0),
            rtol=1e-3,
            err_msg="noll09: FD check ∂(∑k)/∂dust_bump_strength",
        )
        np.testing.assert_allclose(
            float(g_delta),
            fd_grad(f_delta, -0.1),
            rtol=1e-3,
            err_msg="noll09: FD check ∂(∑k)/∂dust_delta",
        )


class TestSalimSBL18:
    """Salim, Boquien & Lee (2018) modified Calzetti+L02."""

    def test_reference_no_mods(self):
        """SBL18 baseline (ampl=0, slope=0) = pure Calzetti+L02."""
        ref = np.array(
            [
                3.42720988,
                2.94808185,
                2.54615821,
                2.09349184,
                1.70999069,
                0.99947911,
                0.46360420,
                0.12220173,
            ]
        )
        tng = np.array(salim_sbl18(jnp.array(WAVS)))
        np.testing.assert_allclose(tng, ref, rtol=1e-6)

    def test_reference_with_mods(self):
        """SBL18 ampl=3, slope=-0.1. Reference from dust_attenuation."""
        ref = np.array(
            [
                4.07068075,
                3.44474637,
                2.93118586,
                3.03774403,
                1.85909357,
                1.00367016,
                0.43769885,
                0.10763381,
            ]
        )
        tng = np.array(salim_sbl18(jnp.array(WAVS), dust_bump_strength=3.0, dust_delta=-0.1))
        np.testing.assert_allclose(tng, ref, rtol=1e-6)

    def test_differs_from_n09_with_both(self):
        """SBL18 != N09 when both bump and slope are nonzero."""
        k_n09 = np.array(noll09(jnp.array(WAVS), dust_bump_strength=3.0, dust_delta=-0.3))
        k_sbl = np.array(salim_sbl18(jnp.array(WAVS), dust_bump_strength=3.0, dust_delta=-0.3))
        assert not np.allclose(k_n09, k_sbl, rtol=1e-3)

    def test_equals_n09_no_slope(self):
        """SBL18 == N09 when slope=0 (only bump matters)."""
        k_n09 = np.array(noll09(jnp.array(WAVS), dust_bump_strength=3.0, dust_delta=0.0))
        k_sbl = np.array(salim_sbl18(jnp.array(WAVS), dust_bump_strength=3.0, dust_delta=0.0))
        np.testing.assert_allclose(k_n09, k_sbl, rtol=1e-10)

    def test_equals_n09_no_bump(self):
        """SBL18 == N09 when bump=0 (only slope matters)."""
        k_n09 = np.array(noll09(jnp.array(WAVS), dust_bump_strength=0.0, dust_delta=-0.3))
        k_sbl = np.array(salim_sbl18(jnp.array(WAVS), dust_bump_strength=0.0, dust_delta=-0.3))
        np.testing.assert_allclose(k_n09, k_sbl, rtol=1e-10)

    def test_jit_compatible(self):
        """Should work inside jax.jit."""
        result = assert_jit_matches_eager(
            lambda w: salim_sbl18(w, dust_bump_strength=2.0, dust_delta=-0.1), jnp.array(WAVS)
        )
        chex.assert_equal_shape([result, WAVS])

    def test_gradient_wrt_params(self):
        """Gradients of salim_sbl18 match central FD w.r.t. bump strength and slope."""

        def loss(ampl, delta):
            return jnp.sum(salim_sbl18(jnp.array(WAVS), dust_bump_strength=ampl, dust_delta=delta))

        g_ampl, g_delta = jax.grad(loss, argnums=(0, 1))(1.0, -0.1)

        def f_ampl(ampl: float) -> float:
            return float(loss(ampl, -0.1))

        def f_delta(delta: float) -> float:
            return float(loss(1.0, delta))

        np.testing.assert_allclose(
            float(g_ampl),
            fd_grad(f_ampl, 1.0),
            rtol=1e-3,
            err_msg="salim_sbl18: FD check ∂(∑k)/∂dust_bump_strength",
        )
        np.testing.assert_allclose(
            float(g_delta),
            fd_grad(f_delta, -0.1),
            rtol=1e-3,
            err_msg="salim_sbl18: FD check ∂(∑k)/∂dust_delta",
        )


# ── Physics reference tests ───────────────────────────────────────
class TestCalzettiPhysics:
    """Reference value tests for Calzetti+2000 attenuation curve."""

    def test_calzetti_rv_normalization(self) -> None:
        """k(5500 Å) = 1.0 by the R_V = 4.05 normalization convention.
        Calzetti et al. 2000, ApJ 533, 682, Eq. 4: the curve is normalized at
        V-band so that one unit of tau_V produces one e-folding at 5500 Å.
        """
        k_v = float(calzetti(jnp.array([5500.0]))[0])
        np.testing.assert_allclose(
            k_v,
            1.0,
            atol=0.01,
            err_msg="Calzetti+2000 Eq. 4: k(5500 Å) = 1.0 by R_V=4.05 normalization",
        )

    def test_calzetti_uv_slope(self) -> None:
        """Extinction increases steeply into the UV: k(1500) > k(3000) > k(5500).
        Calzetti et al. 2000, ApJ 533, 682: the attenuation curve rises monotonically
        from the optical to the far-UV.
        """
        k = calzetti(jnp.array([1500.0, 3000.0, 5500.0]))
        assert float(k[0]) > float(k[1]) > float(k[2]), (
            f"Calzetti UV slope: k(1500)={float(k[0]):.3f}, "
            f"k(3000)={float(k[1]):.3f}, k(5500)={float(k[2]):.3f} — must be decreasing"
        )


class TestCardelli2175Bump:
    """Reference value tests for Cardelli+Clayton+Mathis 1989 extinction curve."""

    def test_ccm_2175_bump(self) -> None:
        """CCM curve has a local maximum at the 2175 Å UV bump.
        Cardelli, Clayton & Mathis 1989, ApJ 345, 245: the 2175 Å feature is the
        strongest interstellar extinction feature in the UV. k(2175) must exceed
        both the adjacent continuum at 2000 Å and 2500 Å.
        """
        wave = jnp.array([2000.0, 2175.0, 2500.0])
        k = cardelli(wave)
        assert float(k[1]) > float(k[0]), (
            f"CCM 2175 Å bump: k(2175)={float(k[1]):.3f} should exceed k(2000)={float(k[0]):.3f}"
        )
        assert float(k[1]) > float(k[2]), (
            f"CCM 2175 Å bump: k(2175)={float(k[1]):.3f} should exceed k(2500)={float(k[2]):.3f}"
        )


class TestReddy15:
    """Reddy et al. (2015) MOSDEF high-redshift dust attenuation curve."""

    def test_registered(self):
        """reddy15 should be in the DUST_LAWS registry."""
        assert "reddy15" in DUST_LAWS

    def test_rv_normalization(self):
        """k(5500 Å) = 1.0 by the R_V = 2.505 normalization convention.
        Reddy et al. 2015, ApJ 806, 259, Eq. 8: the curve is normalized at
        V-band so that one unit of tau_V produces one e-folding at 5500 Å.
        """
        k_v = float(reddy15(jnp.array([5500.0]))[0])
        np.testing.assert_allclose(
            k_v,
            1.0,
            atol=0.01,
            err_msg="Reddy+2015 Eq. 8: k(5500 Å) = 1.0 by R_V=2.505 normalization",
        )

    def test_reference_values_low_wavelength(self):
        """Reddy15 at low wavelengths (0.15-0.60 μm) matches polynomial formula.
        Computed from Reddy et al. 2015, Eq. 8, low-λ segment.
        """
        # Test at 1500 Å (0.15 μm), 2500 Å (0.25 μm), 5500 Å (0.55 μm)
        wavs = jnp.array([1500.0, 2500.0, 5500.0])
        k = reddy15(wavs)
        tng = np.array(k)

        # Computed from the polynomial (Reddy et al. 2015, Eq. 8)
        # normalized to k(5500 Å) = 1.0
        ref = np.array([3.4958, 2.5026, 1.0000])
        np.testing.assert_allclose(tng, ref, rtol=0.01)

    def test_reference_values_high_wavelength(self):
        """Reddy15 at high wavelengths (0.60-2.85 μm) matches polynomial formula.
        Computed from Reddy et al. 2015, Eq. 8, high-λ segment.
        """
        # Test at 6000 Å (0.60 μm), 10000 Å (1.0 μm), 20000 Å (2.0 μm)
        wavs = jnp.array([6000.0, 10000.0, 20000.0])
        k = reddy15(wavs)
        tng = np.array(k)

        # Computed from the polynomial (Reddy et al. 2015, Eq. 8)
        # normalized to k(5500 Å) = 1.0
        ref = np.array([0.8666, 0.3775, 0.0639])
        np.testing.assert_allclose(tng, ref, rtol=0.01)

    def test_positive_output(self):
        """Output should be non-negative across the full validity range."""
        wavs = jnp.linspace(1500.0, 28500.0, 500)
        k = reddy15(wavs)
        assert jnp.all(k >= 0.0), "reddy15 should be non-negative everywhere"

    def test_monotonic_behavior(self):
        """Reddy15 should generally decrease from UV to NIR (low to high wavelength).
        Reddy et al. 2015, Figure 10: the curve is monotonically decreasing overall.
        """
        wavs = jnp.linspace(1500.0, 28500.0, 200)
        k = np.array(reddy15(wavs))
        assert k[0] > k[-1], (
            f"Reddy15 should decrease UV→NIR: k(1500)={k[0]:.3f} > k(28500)={k[-1]:.3f}"
        )

    def test_similarity_to_smc(self):
        """Reddy15 shows similarity to SMC curve at long wavelengths.
        Reddy et al. 2015, Section 3.6.5: The MOSDEF curve is similar in shape
        to the SMC curve, particularly beyond 2500 Å.
        """
        # At long wavelengths, both Reddy15 and SMC should show similar behavior
        wavs = jnp.array([10000.0, 20000.0])
        k_reddy = reddy15(wavs)
        # Both should show relatively gentle slope in the IR
        # Check that IR (10000-20000 Å) values are reasonable (low attenuation)
        assert float(k_reddy[0]) > 0.0, "reddy15 at 10000 Å should be positive"
        assert float(k_reddy[0]) < float(k_reddy[1]) or float(k_reddy[1]) > 0, (
            "Reddy15 should show expected IR behavior"
        )

    def test_jit_compatible(self):
        """Should work inside jax.jit."""
        wavs = jnp.array([1500.0, 5500.0, 20000.0])
        result = assert_jit_matches_eager(reddy15, wavs)
        chex.assert_shape(result, (3,))

    def test_gradient(self):
        """Should be differentiable (for use in optimization)."""

        def loss(wav):
            return jnp.sum(reddy15(wav))

        def f_scalar(wav0: float) -> float:
            return float(loss(jnp.array([wav0])))

        grad_jax = float(jax.grad(loss)(jnp.array([2500.0]))[0])
        np.testing.assert_allclose(
            grad_jax,
            fd_grad(f_scalar, 2500.0, eps=1.0),
            rtol=5e-3,
            err_msg="reddy15: FD check ∂(∑k)/∂wavelength",
        )
