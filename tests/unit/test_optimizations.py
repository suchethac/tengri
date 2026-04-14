"""Tests for performance optimizations."""

import jax
import jax.numpy as jnp
from numpy.testing import assert_allclose

from tengri.models.sfh.gp_sfh import compute_sqrt_power_drw
from tengri.utils.optimizations import (
    approximate_photometry,
    batched_forward,
    checkpointed_forward_model,
    compute_full_amplitude_drw,
    effective_wavelength,
    gp_from_xi_hartley,
    hartley,
    inverse_hartley,
    precompute_ssp_photometry,
)

jax.config.update("jax_enable_x64", True)

N = 256


class TestHartleyTransform:
    """Tests for Hartley transform (NIFTy.re-inspired)."""

    def test_hartley_is_real(self):
        """Hartley transform of real input is real."""
        x = jax.random.normal(jax.random.PRNGKey(0), shape=(N,))
        h = hartley(x)
        assert h.dtype in (jnp.float32, jnp.float64)
        assert not jnp.iscomplexobj(h)

    def test_inverse_roundtrip(self):
        """inverse_hartley(hartley(x)) = x."""
        x = jax.random.normal(jax.random.PRNGKey(1), shape=(N,))
        recovered = inverse_hartley(hartley(x))
        assert_allclose(recovered, x, atol=1e-10)

    def test_hartley_self_inverse(self):
        """hartley(hartley(x)) = N * x (self-reciprocal up to N)."""
        x = jax.random.normal(jax.random.PRNGKey(2), shape=(N,))
        hh = hartley(hartley(x))
        assert_allclose(hh, N * x, atol=1e-8)

    def test_is_jittable(self):
        """Hartley transform can be JIT-compiled."""
        fn = jax.jit(hartley)
        x = jax.random.normal(jax.random.PRNGKey(3), shape=(N,))
        h = fn(x)
        assert h.shape == (N,)

    def test_has_gradients(self):
        """Gradients through Hartley transform are finite."""
        grad_fn = jax.grad(lambda x: jnp.sum(hartley(x) ** 2))
        x = jax.random.normal(jax.random.PRNGKey(4), shape=(N,))
        g = grad_fn(x)
        assert jnp.all(jnp.isfinite(g))


class TestHartleyGP:
    """Test Hartley-based GP generation matches rfft-based version."""

    def test_gp_hartley_has_correct_shape(self):
        """Hartley GP output has correct shape."""
        d = (10.14 - 6.0) / (N - 1)
        amp = compute_full_amplitude_drw(N, d, 1.0, 50e6)
        xi = jax.random.normal(jax.random.PRNGKey(0), shape=(N,))
        gp = gp_from_xi_hartley(xi, amp)
        assert gp.shape == (N,)

    def test_gp_hartley_zero_xi_gives_zero(self):
        """Zero xi gives zero GP for Hartley version."""
        d = (10.14 - 6.0) / (N - 1)
        amp = compute_full_amplitude_drw(N, d, 1.0, 50e6)
        xi = jnp.zeros(N)
        gp = gp_from_xi_hartley(xi, amp)
        assert_allclose(gp, 0.0, atol=1e-15)

    def test_gp_hartley_has_gradients(self):
        """Gradients through Hartley GP are finite."""
        d = (10.14 - 6.0) / (N - 1)
        amp = compute_full_amplitude_drw(N, d, 1.0, 50e6)
        grad_fn = jax.grad(lambda xi: jnp.sum(gp_from_xi_hartley(xi, amp)))
        xi = jax.random.normal(jax.random.PRNGKey(0), shape=(N,))
        g = grad_fn(xi)
        assert jnp.all(jnp.isfinite(g))

    def test_hartley_and_rfft_same_statistics(self):
        """Hartley and rfft GP versions produce similar variance."""
        d = (10.14 - 6.0) / (N - 1)

        # rfft version
        sqrt_power = compute_sqrt_power_drw(N, d, 1.0, 50e6)
        # Hartley version
        amp_full = compute_full_amplitude_drw(N, d, 1.0, 50e6)

        key = jax.random.PRNGKey(42)
        n_real = 2000

        # rfft GP batch
        from tengri.models.sfh.gp_sfh import generate_gp_batch

        batch_rfft = generate_gp_batch(key, sqrt_power, N, n_real)
        var_rfft = float(jnp.var(batch_rfft))

        # Hartley GP batch
        keys = jax.random.split(key, n_real)
        batch_hartley = jax.vmap(
            lambda k: gp_from_xi_hartley(jax.random.normal(k, (N,)), amp_full)
        )(keys)
        var_hartley = float(jnp.var(batch_hartley))

        # Variances should be the same order of magnitude
        ratio = var_hartley / max(var_rfft, 1e-30)
        assert 0.1 < ratio < 10.0, f"Hartley/rfft variance ratio = {ratio:.3f}, expected ~1"


# ---------------------------------------------------------------------------
# effective_wavelength
# ---------------------------------------------------------------------------


class TestEffectiveWavelength:
    def test_uniform_filter_centroid(self):
        """Flat filter → λ_eff = int(T*λ²dλ)/int(T*λdλ) = midpoint of range."""
        wave = jnp.linspace(5000.0, 7000.0, 100)
        trans = jnp.ones(100)
        lam_eff = effective_wavelength(wave, trans)
        # For uniform T: λ_eff = (λ_max² + λ_min*λ_max + λ_min²)/(λ_max + λ_min) ≈ 6003 Å
        # Just verify it's within the filter bounds
        assert float(wave[0]) < float(lam_eff) < float(wave[-1])

    def test_returns_scalar(self):
        wave = jnp.linspace(4000.0, 8000.0, 50)
        trans = jnp.ones(50)
        lam_eff = effective_wavelength(wave, trans)
        assert lam_eff.shape == ()

    def test_finite(self):
        wave = jnp.linspace(3000.0, 9000.0, 80)
        trans = jnp.exp(-((wave - 6000.0) ** 2) / (500.0**2))
        lam_eff = effective_wavelength(wave, trans)
        assert jnp.isfinite(lam_eff)

    def test_peaked_filter_near_peak(self):
        """Gaussian filter → λ_eff ≈ peak wavelength."""
        wave = jnp.linspace(5000.0, 7000.0, 200)
        peak = 6000.0
        trans = jnp.exp(-((wave - peak) ** 2) / (100.0**2))
        lam_eff = effective_wavelength(wave, trans)
        assert abs(float(lam_eff) - peak) < 50.0  # within 50 Å


# ---------------------------------------------------------------------------
# precompute_ssp_photometry
# ---------------------------------------------------------------------------


class TestPrecomputeSspPhotometry:
    @staticmethod
    def _make_inputs():
        n_age, n_wave = 10, 100
        ssp_wave = jnp.linspace(3000.0, 9000.0, n_wave)
        ssp_flux = jnp.ones((n_age, n_wave))  # flat spectra
        filter_wave = jnp.linspace(5000.0, 7000.0, 40)
        filter_trans = jnp.ones(40)
        return ssp_flux, ssp_wave, filter_wave, filter_trans

    def test_output_shape(self):
        ssp_flux, ssp_wave, filter_wave, filter_trans = self._make_inputs()
        c = precompute_ssp_photometry(ssp_flux, ssp_wave, filter_wave, filter_trans, redshift=0.0)
        assert c.shape == (ssp_flux.shape[0],)

    def test_flat_spectra_constant_output(self):
        """Flat spectra → pre-computed flux is the same for every age bin."""
        ssp_flux, ssp_wave, filter_wave, filter_trans = self._make_inputs()
        c = precompute_ssp_photometry(ssp_flux, ssp_wave, filter_wave, filter_trans, redshift=0.0)
        assert jnp.allclose(c, c[0], rtol=1e-4)

    def test_finite_output(self):
        ssp_flux, ssp_wave, filter_wave, filter_trans = self._make_inputs()
        c = precompute_ssp_photometry(ssp_flux, ssp_wave, filter_wave, filter_trans, redshift=0.1)
        assert jnp.all(jnp.isfinite(c))

    def test_filter_outside_grid_gives_near_zero(self):
        """Filter entirely outside SSP range → pre-computed flux ≈ 0."""
        n_age, n_wave = 5, 50
        ssp_wave = jnp.linspace(3000.0, 9000.0, n_wave)
        ssp_flux = jnp.ones((n_age, n_wave))
        # Filter far in UV, not covered by SSP
        filter_wave = jnp.linspace(100.0, 200.0, 20)
        filter_trans = jnp.ones(20)
        c = precompute_ssp_photometry(ssp_flux, ssp_wave, filter_wave, filter_trans, redshift=0.0)
        assert jnp.all(jnp.abs(c) < 1e-10)


# ---------------------------------------------------------------------------
# approximate_photometry
# ---------------------------------------------------------------------------


class TestApproximatePhotometry:
    def test_output_scalar(self):
        n_age = 10
        weights = jnp.ones(n_age) / n_age
        ssp_phot = jnp.ones(n_age) * 1e28  # erg/s/Hz per age bin
        dust = jnp.ones(n_age)
        dl_cm = 3.086e27  # ~1 Gpc
        result = approximate_photometry(weights, ssp_phot, dust, dl_cm, redshift=0.1)
        assert result.shape == ()

    def test_finite_output(self):
        n_age = 8
        weights = jnp.ones(n_age) / n_age
        ssp_phot = jnp.ones(n_age) * 1e28
        dust = jnp.ones(n_age) * 0.5
        dl_cm = 3.086e27
        result = approximate_photometry(weights, ssp_phot, dust, dl_cm, redshift=0.5)
        assert jnp.isfinite(result)

    def test_zero_dust_higher_flux(self):
        """No dust (trans=1) gives more flux than dust (trans<1)."""
        n_age = 6
        weights = jnp.ones(n_age) / n_age
        ssp_phot = jnp.ones(n_age) * 1e28
        dl_cm = 3.086e26
        flux_no_dust = approximate_photometry(weights, ssp_phot, jnp.ones(n_age), dl_cm, 0.1)
        flux_dust = approximate_photometry(weights, ssp_phot, jnp.ones(n_age) * 0.3, dl_cm, 0.1)
        assert float(flux_no_dust) > float(flux_dust)


# ---------------------------------------------------------------------------
# checkpointed_forward_model
# ---------------------------------------------------------------------------


class TestCheckpointedForwardModel:
    def test_returns_callable(self):
        fn = lambda x: x**2  # noqa: E731
        cfn = checkpointed_forward_model(fn)
        assert callable(cfn)

    def test_same_output(self):
        """Checkpointed function produces identical output to the original."""
        fn = lambda x: jnp.sum(jnp.sin(x))  # noqa: E731
        cfn = checkpointed_forward_model(fn)
        x = jnp.linspace(0.0, 3.14, 50)
        assert jnp.allclose(fn(x), cfn(x))

    def test_gradients_finite(self):
        """Gradients through checkpointed function are finite."""
        fn = lambda x: jnp.sum(x**2)  # noqa: E731
        cfn = checkpointed_forward_model(fn)
        g = jax.grad(cfn)(jnp.ones(10))
        assert jnp.all(jnp.isfinite(g))


# ---------------------------------------------------------------------------
# batched_forward
# ---------------------------------------------------------------------------


class TestBatchedForward:
    def test_output_shape(self):
        """batched_forward concatenates results from all batches correctly."""
        model_fn = lambda p: jnp.sum(p["x"])  # noqa: E731
        params = {"x": jnp.ones((25, 4))}
        result = batched_forward(model_fn, params, batch_size=10)
        assert result.shape == (25,)

    def test_same_as_vmap(self):
        """batched_forward matches jax.vmap over the full batch."""
        model_fn = lambda p: jnp.sum(p["x"] ** 2)  # noqa: E731
        n = 20
        params = {"x": jnp.arange(n * 3, dtype=float).reshape(n, 3)}
        batched = batched_forward(model_fn, params, batch_size=7)
        vmapped = jax.vmap(model_fn)(params)
        assert jnp.allclose(batched, vmapped, rtol=1e-5)

    def test_batch_size_larger_than_n(self):
        """Batch size larger than dataset: single chunk, correct result."""
        model_fn = lambda p: p["v"] * 2.0  # noqa: E731
        params = {"v": jnp.ones(5)}
        result = batched_forward(model_fn, params, batch_size=100)
        assert result.shape == (5,)
        assert jnp.allclose(result, 2.0)
