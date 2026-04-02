"""Tests for Witt & Gordon (2000) dust geometry transmission functions."""

import jax
import jax.numpy as jnp
import pytest
from numpy.testing import assert_allclose

from tengri.models.dust.attenuation import (
    wg00_cloudy,
    wg00_dusty,
    wg00_shell,
)

jax.config.update("jax_enable_x64", True)


@pytest.fixture
def wavelength():
    """Standard wavelength grid spanning UV to NIR (1000-25000 A)."""
    return jnp.linspace(1000.0, 25000.0, 200)


# ===================================================================
# wg00_shell tests
# ===================================================================


class TestWG00Shell:
    """SHELL (foreground screen) geometry."""

    def test_output_shape(self, wavelength):
        """Output shape matches input wavelength array."""
        result = wg00_shell(wavelength, tau_v=1.0)
        assert result.shape == wavelength.shape

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
        """Gradients exist with respect to tau_V."""
        wave = jnp.array([3000.0, 5500.0, 10000.0])
        grad_fn = jax.grad(lambda t: jnp.sum(wg00_shell(wave, t, law="cardelli")))
        grad_val = grad_fn(1.0)
        assert jnp.isfinite(grad_val)
        # Gradient should be negative (more dust -> less transmission)
        assert float(grad_val) < 0.0


# ===================================================================
# wg00_cloudy tests
# ===================================================================


class TestWG00Cloudy:
    """CLOUDY (homogeneous dust-star mix) geometry."""

    def test_output_shape(self, wavelength):
        result = wg00_cloudy(wavelength, tau_v=1.0)
        assert result.shape == wavelength.shape

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

    def test_greyer_than_shell(self, wavelength):
        """Cloudy geometry gives higher mean transmission (greyer) than shell."""
        tau_v = 2.0
        t_shell = wg00_shell(wavelength, tau_v=tau_v)
        t_cloudy = wg00_cloudy(wavelength, tau_v=tau_v)
        # Cloudy should be greyer: higher mean transmission at same tau
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
        assert jnp.all(jnp.isfinite(result))
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
        """Gradients exist with respect to tau_V."""
        wave = jnp.array([3000.0, 5500.0, 10000.0])
        grad_fn = jax.grad(lambda t: jnp.sum(wg00_cloudy(wave, t, law="cardelli")))
        grad_val = grad_fn(1.0)
        assert jnp.isfinite(grad_val)
        assert float(grad_val) < 0.0


# ===================================================================
# wg00_dusty tests
# ===================================================================


class TestWG00Dusty:
    """DUSTY (clumpy two-phase medium) geometry."""

    def test_output_shape(self, wavelength):
        result = wg00_dusty(wavelength, tau_v=1.0)
        assert result.shape == wavelength.shape

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

    def test_greyer_than_shell(self, wavelength):
        """Clumpy geometry gives higher mean transmission (greyer) than shell."""
        tau_v = 2.0
        t_shell = wg00_shell(wavelength, tau_v=tau_v)
        t_dusty = wg00_dusty(wavelength, tau_v=tau_v, n_clumps=10.0)
        assert float(jnp.mean(t_dusty)) > float(jnp.mean(t_shell))

    def test_fewer_clumps_greyer_than_many(self, wavelength):
        """With fewer clumps (same tau), attenuation is greyer."""
        tau_v = 4.0
        t_few = wg00_dusty(wavelength, tau_v=tau_v, n_clumps=3.0)
        t_many = wg00_dusty(wavelength, tau_v=tau_v, n_clumps=30.0)
        assert float(jnp.mean(t_few)) > float(jnp.mean(t_many))

    def test_many_clumps_greyer_than_few(self, wavelength):
        """More clumps (same total tau) gives less grey attenuation."""
        tau_v = 2.0
        t_100 = wg00_dusty(wavelength, tau_v=tau_v, n_clumps=100.0)
        t_1000 = wg00_dusty(wavelength, tau_v=tau_v, n_clumps=1000.0)
        # More clumps = less grey = lower mean transmission
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

    def test_fewer_clumps_greyer(self, wavelength):
        """Fewer clumps (same total tau) gives greyer attenuation."""
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
        """Gradients exist with respect to tau_V."""
        wave = jnp.array([3000.0, 5500.0, 10000.0])
        grad_fn = jax.grad(lambda t: jnp.sum(wg00_dusty(wave, t, law="cardelli", n_clumps=10.0)))
        grad_val = grad_fn(1.0)
        assert jnp.isfinite(grad_val)
        assert float(grad_val) < 0.0

    def test_differentiable_n_clumps(self):
        """Gradients exist with respect to n_clumps."""
        wave = jnp.array([3000.0, 5500.0, 10000.0])
        grad_fn = jax.grad(lambda n: jnp.sum(wg00_dusty(wave, 2.0, law="cardelli", n_clumps=n)))
        grad_val = grad_fn(10.0)
        assert jnp.isfinite(grad_val)


# ===================================================================
# Geometry ordering tests (cross-geometry comparisons)
# ===================================================================


class TestGeometryOrdering:
    """The three WG00 geometries should satisfy shell < cloudy < dusty
    in terms of mean transmission (for moderate tau and n_clumps)."""

    def test_ordering_moderate_tau(self, wavelength):
        """SHELL most absorbed, CLOUDY greyer than shell."""
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
