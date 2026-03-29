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

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from tengri.models.dust.attenuation import (
    DUST_LAWS,
    calzetti,
    leitherer02,
    noll09,
    salim_sbl18,
)

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
        f = jax.jit(leitherer02)
        wavs = jnp.array([1500.0, 5500.0])
        result = f(wavs)
        assert result.shape == (2,)

    def test_gradient(self):
        """Should be differentiable (for use in optimization)."""

        # leitherer02 has no free params, but the wavelength input should
        # propagate gradients through the curve
        def loss(wav):
            return jnp.sum(leitherer02(wav))

        grad_fn = jax.grad(loss)
        g = grad_fn(jnp.array([1500.0]))
        assert jnp.isfinite(g).all()


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
        f = jax.jit(lambda w: noll09(w, dust_bump_strength=2.0, dust_delta=-0.1))
        result = f(jnp.array(WAVS))
        assert result.shape == WAVS.shape

    def test_gradient_wrt_params(self):
        """Gradients w.r.t. bump and slope should be finite."""

        def loss(ampl, delta):
            return jnp.sum(noll09(jnp.array(WAVS), dust_bump_strength=ampl, dust_delta=delta))

        g_ampl, g_delta = jax.grad(loss, argnums=(0, 1))(1.0, -0.1)
        assert jnp.isfinite(g_ampl)
        assert jnp.isfinite(g_delta)


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
        f = jax.jit(lambda w: salim_sbl18(w, dust_bump_strength=2.0, dust_delta=-0.1))
        result = f(jnp.array(WAVS))
        assert result.shape == WAVS.shape

    def test_gradient_wrt_params(self):
        """Gradients w.r.t. bump and slope should be finite."""

        def loss(ampl, delta):
            return jnp.sum(salim_sbl18(jnp.array(WAVS), dust_bump_strength=ampl, dust_delta=delta))

        g_ampl, g_delta = jax.grad(loss, argnums=(0, 1))(1.0, -0.1)
        assert jnp.isfinite(g_ampl)
        assert jnp.isfinite(g_delta)
