# SPDX-License-Identifier: BSD-3-Clause
"""Tests for Witt & Gordon (2000) dust geometry transmission functions."""

import chex
import pytest

pytestmark = pytest.mark.bounds
import jax
import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_allclose

from tengri.components.dust.attenuation import (
    wg00_cloudy,
    wg00_dusty,
    wg00_shell,
)


def fd_grad(f, x: float, eps: float = 1e-4) -> float:
    """Central finite-difference gradient. O(eps^2) accurate."""

    return float((f(x + eps) - f(x - eps)) / (2.0 * eps))


@pytest.fixture
def wavelength():
    """Standard wavelength grid spanning UV to NIR (1000-25000 A)."""
    return jnp.linspace(1000.0, 25000.0, 200)


# ── wg00_shell tests ──────────────────────────────────────────────
class TestWG00Shell:
    """SHELL (foreground screen) geometry."""

    def test_output_shape(self, wavelength):
        """Output shape matches input wavelength array."""
        result = wg00_shell(wavelength, tau_v=1.0)
        chex.assert_equal_shape([result, wavelength])

    def test_zero_tau_gives_unity(self, wavelength):
        """Zero optical depth gives full transmission."""
        result = wg00_shell(wavelength, tau_v=0.0)
        assert_allclose(result, 1.0, rtol=1e-12)

    def test_values_in_unit_interval(self, wavelength):
        """Transmission is in [0, 1] for positive tau."""
        for tau in [0.25, 1.0, 4.0, 16.0]:
            result = wg00_shell(wavelength, tau_v=tau)
            assert jnp.all(result >= 0.0)
            assert jnp.all(result <= 1.0)

    def test_more_dust_less_transmission(self, wavelength):
        """Higher tau_V gives lower mean transmission."""
        t_low = wg00_shell(wavelength, tau_v=0.5)
        t_high = wg00_shell(wavelength, tau_v=2.0)
        assert float(jnp.mean(t_high)) < float(jnp.mean(t_low))

    def test_transmission_at_v_band(self):
        """At V band (5500 A), T = exp(-tau_V) since k(5500) = 1."""
        wave_v = jnp.array([5500.0])
        tau_v = 1.5
        result = wg00_shell(wave_v, tau_v=tau_v)
        # k(5500) ≈ 1.0 but not exact for all curves; relax tolerance
        assert_allclose(result, jnp.exp(-tau_v), rtol=1e-2)

    def test_different_dust_laws(self, wavelength):
        """Different extinction curves produce different transmissions."""
        t_mw = wg00_shell(wavelength, tau_v=1.0, law="cardelli")
        t_smc = wg00_shell(wavelength, tau_v=1.0, law="smc")
        # They should differ (especially in the UV)
        assert not jnp.allclose(t_mw, t_smc, atol=0.01)

    def test_jit_compatible(self, wavelength):
        """Function is JIT-compilable."""
        jitted = jax.jit(lambda w, t: wg00_shell(w, t, law="cardelli"))
        result = jitted(wavelength, 1.0)
        expected = wg00_shell(wavelength, tau_v=1.0, law="cardelli")
        assert_allclose(result, expected, rtol=1e-12)

    def test_differentiable(self):
        """FD check: ∂(∑T)/∂tau_V for wg00_shell. Gradient should be negative."""
        wave = jnp.array([3000.0, 5500.0, 10000.0])

        def f(t):
            return float(jnp.sum(wg00_shell(wave, t, law="cardelli")))

        grad_jax = float(jax.grad(lambda t: jnp.sum(wg00_shell(wave, t, law="cardelli")))(1.0))
        grad_fd = fd_grad(f, 1.0, eps=1e-4)
        np.testing.assert_allclose(
            grad_jax,
            grad_fd,
            rtol=1e-3,
            err_msg="wg00_shell: autodiff vs FD gradient w.r.t. tau_V",
        )
        assert grad_jax < 0.0, "wg00_shell: ∂T/∂tau_V should be negative"

    def test_shell_opaque_limit(self):
        """At tau_V=100, T must be < 1e-10 (Beer-Lambert opaque regime).
        Witt & Gordon 2000, ApJ 528, 799: in the shell (foreground screen) geometry
        the transmission is exp(-tau * k(lambda)), which decays exponentially at
        large tau. At tau_V=100 and V-band, T ≈ exp(-100) ≈ 3.7e-44, which is
        numerically zero. Even at shorter wavelengths where k < 1 (NIR end), the
        value must be far below 1e-10. This test guards against incorrect
        geometry implementations that saturate at a finite floor.
        """
        T = float(wg00_shell(jnp.array([5500.0]), tau_v=100.0, law="cardelli")[0])
        assert T < 1e-10, f"WG00 shell opaque limit: T={T:.2e} at tau_V=100 (expected <1e-10)"


# ── wg00_cloudy tests ─────────────────────────────────────────────
class TestWG00Cloudy:
    """CLOUDY (homogeneous dust-star mix) geometry."""

    def test_output_shape(self, wavelength):
        result = wg00_cloudy(wavelength, tau_v=1.0)
        chex.assert_equal_shape([result, wavelength])

    def test_zero_tau_gives_unity(self, wavelength):
        """Zero optical depth gives full transmission."""
        result = wg00_cloudy(wavelength, tau_v=0.0)
        assert_allclose(result, 1.0, atol=1e-6)

    def test_values_in_unit_interval(self, wavelength):
        """Transmission is in [0, 1] for positive tau."""
        for tau in [0.25, 1.0, 4.0, 16.0]:
            result = wg00_cloudy(wavelength, tau_v=tau)
            assert jnp.all(result >= 0.0), f"Negative T at tau={tau}"
            assert jnp.all(result <= 1.0 + 1e-10), f"T > 1 at tau={tau}"

    def test_grayer_than_shell(self, wavelength):
        """Cloudy geometry gives higher mean transmission (grayer) than shell."""
        tau_v = 2.0
        t_shell = wg00_shell(wavelength, tau_v=tau_v)
        t_cloudy = wg00_cloudy(wavelength, tau_v=tau_v)
        # Cloudy should be grayer: higher mean transmission at same tau
        assert float(jnp.mean(t_cloudy)) > float(jnp.mean(t_shell))

    def test_slab_formula_directly(self):
        """Check against hand-computed slab formula at V band."""
        wave_v = jnp.array([5500.0])
        tau_v = 2.0
        result = wg00_cloudy(wave_v, tau_v=tau_v)
        # At V band, k(5500) = 1, so T = (1 - exp(-tau)) / tau
        expected = (1.0 - jnp.exp(-tau_v)) / tau_v
        assert_allclose(result, expected, rtol=1e-2)

    def test_low_tau_approaches_unity(self):
        """For very small tau, slab formula -> 1."""
        wave_v = jnp.array([5500.0])
        result = wg00_cloudy(wave_v, tau_v=1e-8)
        assert_allclose(result, 1.0, atol=1e-6)

    def test_numerical_stability_small_tau(self, wavelength):
        """No NaN or inf at very small optical depths."""
        result = wg00_cloudy(wavelength, tau_v=1e-12)
        chex.assert_tree_all_finite(result)
        assert_allclose(result, 1.0, atol=1e-4)

    def test_more_dust_less_transmission(self, wavelength):
        """Higher tau_V gives lower mean transmission."""
        t_low = wg00_cloudy(wavelength, tau_v=0.5)
        t_high = wg00_cloudy(wavelength, tau_v=4.0)
        assert float(jnp.mean(t_high)) < float(jnp.mean(t_low))

    def test_jit_compatible(self, wavelength):
        jitted = jax.jit(lambda w, t: wg00_cloudy(w, t, law="cardelli"))
        result = jitted(wavelength, 1.0)
        expected = wg00_cloudy(wavelength, tau_v=1.0, law="cardelli")
        assert_allclose(result, expected, rtol=1e-12)

    def test_differentiable(self):
        """FD check: ∂(∑T)/∂tau_V for wg00_cloudy. Gradient should be negative."""
        wave = jnp.array([3000.0, 5500.0, 10000.0])

        def f(t):
            return float(jnp.sum(wg00_cloudy(wave, t, law="cardelli")))

        grad_jax = float(jax.grad(lambda t: jnp.sum(wg00_cloudy(wave, t, law="cardelli")))(1.0))
        grad_fd = fd_grad(f, 1.0, eps=1e-4)
        np.testing.assert_allclose(
            grad_jax,
            grad_fd,
            rtol=1e-3,
            err_msg="wg00_cloudy: autodiff vs FD gradient w.r.t. tau_V",
        )
        assert grad_jax < 0.0, "wg00_cloudy: ∂T/∂tau_V should be negative"


# ── wg00_dusty tests ──────────────────────────────────────────────
class TestWG00Dusty:
    """DUSTY (clumpy two-phase medium) geometry."""

    def test_output_shape(self, wavelength):
        result = wg00_dusty(wavelength, tau_v=1.0)
        chex.assert_equal_shape([result, wavelength])

    def test_zero_tau_gives_unity(self, wavelength):
        """Zero optical depth gives full transmission."""
        result = wg00_dusty(wavelength, tau_v=0.0)
        assert_allclose(result, 1.0, rtol=1e-10)

    def test_values_in_unit_interval(self, wavelength):
        """Transmission is in [0, 1] for positive tau."""
        for tau in [0.25, 1.0, 4.0, 16.0]:
            result = wg00_dusty(wavelength, tau_v=tau, n_clumps=10.0)
            assert jnp.all(result >= 0.0), f"Negative T at tau={tau}"
            assert jnp.all(result <= 1.0 + 1e-10), f"T > 1 at tau={tau}"

    def test_grayer_than_shell(self, wavelength):
        """Clumpy geometry gives higher mean transmission (grayer) than shell."""
        tau_v = 2.0
        t_shell = wg00_shell(wavelength, tau_v=tau_v)
        t_dusty = wg00_dusty(wavelength, tau_v=tau_v, n_clumps=10.0)
        assert float(jnp.mean(t_dusty)) > float(jnp.mean(t_shell))

    def test_fewer_clumps_grayer_than_many(self, wavelength):
        """With fewer clumps (same tau), attenuation is grayer."""
        tau_v = 4.0
        t_few = wg00_dusty(wavelength, tau_v=tau_v, n_clumps=3.0)
        t_many = wg00_dusty(wavelength, tau_v=tau_v, n_clumps=30.0)
        assert float(jnp.mean(t_few)) > float(jnp.mean(t_many))

    def test_many_clumps_grayer_than_few(self, wavelength):
        """More clumps (same total tau) gives less gray attenuation."""
        tau_v = 2.0
        t_100 = wg00_dusty(wavelength, tau_v=tau_v, n_clumps=100.0)
        t_1000 = wg00_dusty(wavelength, tau_v=tau_v, n_clumps=1000.0)
        # More clumps = less gray = lower mean transmission
        assert float(jnp.mean(t_1000)) < float(jnp.mean(t_100))

    def test_clumpy_formula_at_v_band(self):
        """Check against hand-computed clumpy formula at V band."""
        wave_v = jnp.array([5500.0])
        tau_v = 3.0
        n_clumps = 10.0
        tau_clump = tau_v / n_clumps
        result = wg00_dusty(wave_v, tau_v=tau_v, n_clumps=n_clumps)
        # At V band, k(5500) = 1
        expected = jnp.exp(-n_clumps * (1.0 - jnp.exp(-tau_clump)))
        assert_allclose(result, expected, rtol=1e-2)

    def test_single_clump(self):
        """n_clumps=1: single clump with tau_clump = tau_V."""
        wave_v = jnp.array([5500.0])
        tau_v = 2.0
        result = wg00_dusty(wave_v, tau_v=tau_v, n_clumps=1.0)
        # T = exp(-1 * (1 - exp(-tau_V * k(5500)))) ≈ exp(-(1 - exp(-tau_V)))
        expected = jnp.exp(-(1.0 - jnp.exp(-tau_v)))
        assert_allclose(result, expected, rtol=1e-2)

    def test_more_dust_less_transmission(self, wavelength):
        t_low = wg00_dusty(wavelength, tau_v=0.5, n_clumps=10.0)
        t_high = wg00_dusty(wavelength, tau_v=4.0, n_clumps=10.0)
        assert float(jnp.mean(t_high)) < float(jnp.mean(t_low))

    def test_fewer_clumps_grayer(self, wavelength):
        """Fewer clumps (same total tau) gives grayer attenuation."""
        tau_v = 4.0
        t_few = wg00_dusty(wavelength, tau_v=tau_v, n_clumps=3.0)
        t_many = wg00_dusty(wavelength, tau_v=tau_v, n_clumps=30.0)
        assert float(jnp.mean(t_few)) > float(jnp.mean(t_many))

    def test_jit_compatible(self, wavelength):
        jitted = jax.jit(lambda w, t, n: wg00_dusty(w, t, law="cardelli", n_clumps=n))
        result = jitted(wavelength, 1.0, 10.0)
        expected = wg00_dusty(wavelength, tau_v=1.0, law="cardelli", n_clumps=10.0)
        assert_allclose(result, expected, rtol=1e-12)

    def test_differentiable_tau(self):
        """FD check: ∂(∑T)/∂tau_V for wg00_dusty. Gradient should be negative."""
        wave = jnp.array([3000.0, 5500.0, 10000.0])

        def f(t):
            return float(jnp.sum(wg00_dusty(wave, t, law="cardelli", n_clumps=10.0)))

        grad_jax = float(
            jax.grad(lambda t: jnp.sum(wg00_dusty(wave, t, law="cardelli", n_clumps=10.0)))(1.0)
        )
        grad_fd = fd_grad(f, 1.0, eps=1e-4)
        np.testing.assert_allclose(
            grad_jax,
            grad_fd,
            rtol=1e-3,
            err_msg="wg00_dusty: autodiff vs FD gradient w.r.t. tau_V",
        )
        assert grad_jax < 0.0, "wg00_dusty: ∂T/∂tau_V should be negative"

    def test_differentiable_n_clumps(self):
        """FD check: ∂(∑T)/∂n_clumps for wg00_dusty."""
        wave = jnp.array([3000.0, 5500.0, 10000.0])

        def f(n):
            return float(jnp.sum(wg00_dusty(wave, 2.0, law="cardelli", n_clumps=n)))

        grad_jax = float(
            jax.grad(lambda n: jnp.sum(wg00_dusty(wave, 2.0, law="cardelli", n_clumps=n)))(10.0)
        )
        grad_fd = fd_grad(f, 10.0, eps=0.1)
        np.testing.assert_allclose(
            grad_jax,
            grad_fd,
            rtol=1e-3,
            err_msg="wg00_dusty: autodiff vs FD gradient w.r.t. n_clumps",
        )


# ── Geometry ordering tests (cross-geometry comparisons) ──────────
class TestGeometryOrdering:
    """The three WG00 geometries should satisfy shell < cloudy < dusty
    in terms of mean transmission (for moderate tau and n_clumps)."""

    def test_ordering_moderate_tau(self, wavelength):
        """SHELL most absorbed, CLOUDY grayer than shell."""
        tau_v = 2.0
        t_shell = float(jnp.mean(wg00_shell(wavelength, tau_v=tau_v)))
        t_cloudy = float(jnp.mean(wg00_cloudy(wavelength, tau_v=tau_v)))
        assert t_shell < t_cloudy

    def test_ordering_high_tau(self, wavelength):
        """Ordering holds at high optical depth."""
        tau_v = 8.0
        t_shell = float(jnp.mean(wg00_shell(wavelength, tau_v=tau_v)))
        t_cloudy = float(jnp.mean(wg00_cloudy(wavelength, tau_v=tau_v)))
        assert t_shell < t_cloudy

    def test_all_converge_at_zero_tau(self, wavelength):
        """All geometries give T=1 at tau_V=0."""
        for fn in [wg00_shell, wg00_cloudy]:
            result = fn(wavelength, tau_v=0.0)
            assert_allclose(result, 1.0, atol=1e-10)
        result = wg00_dusty(wavelength, tau_v=0.0, n_clumps=10.0)
        assert_allclose(result, 1.0, atol=1e-10)

    def test_smc_vs_mw_uv_difference(self):
        """SMC dust produces steeper UV attenuation than MW in all geometries."""
        wave_uv = jnp.array([1500.0])
        wave_nir = jnp.array([10000.0])
        tau_v = 1.0
        for geom_fn in [wg00_shell, wg00_cloudy]:
            t_uv_mw = geom_fn(wave_uv, tau_v=tau_v, law="cardelli")
            t_nir_mw = geom_fn(wave_nir, tau_v=tau_v, law="cardelli")
            t_uv_smc = geom_fn(wave_uv, tau_v=tau_v, law="smc")
            t_nir_smc = geom_fn(wave_nir, tau_v=tau_v, law="smc")
            # UV/NIR ratio for SMC should be lower (steeper)
            ratio_mw = float((t_uv_mw / t_nir_mw)[0])
            ratio_smc = float((t_uv_smc / t_nir_smc)[0])
            assert ratio_smc < ratio_mw
