"""Unit tests for top_hat and gaussian_burst parametric SFH models."""

import jax
import jax.numpy as jnp
import pytest

from tengri.components.stellar.sfh import (
    gaussian_burst,
    resolve_sfh,
    top_hat,
)


class TestTopHat:
    """Tests for top_hat parametric SFH model."""

    @pytest.mark.unit
    def test_top_hat_shape(self):
        """Test output shape matches input."""
        t = jnp.logspace(7, 10.14, 64)
        sfr = top_hat(t, amplitude=1.0, t_start=5e9, t_end=3e9)
        assert sfr.shape == t.shape

    @pytest.mark.unit
    def test_top_hat_nonzero_in_window(self):
        """Test SFR is nonzero inside the window."""
        t = jnp.logspace(7, 10.14, 256)
        amplitude = 2.5
        t_start = 5e9
        t_end = 3e9
        sfr = top_hat(t, amplitude=amplitude, t_start=t_start, t_end=t_end, smooth_width=1e8)

        # Find points clearly inside the window (away from edges)
        in_window = (t >= (t_end + 1e8)) & (t <= (t_start - 1e8))
        sfr_in = sfr[in_window]

        # SFR should be close to amplitude in the interior (allowing for sigmoid ramp)
        assert jnp.all(sfr_in > 0.8 * amplitude), "SFR too low in window interior"
        assert jnp.all(sfr_in <= amplitude * 1.01), "SFR exceeds amplitude in window"

    @pytest.mark.unit
    def test_top_hat_decay_outside_window(self):
        """Test SFR decays rapidly outside the window."""
        t = jnp.linspace(1e8, 1e10, 512)
        amplitude = 2.5
        t_start = 5e9
        t_end = 3e9
        sfr = top_hat(t, amplitude=amplitude, t_start=t_start, t_end=t_end, smooth_width=1e8)

        # Find max SFR values far from window (beyond 10 smooth_widths)
        far_before = sfr[t < (t_end - 1e9)]
        far_after = sfr[t > (t_start + 1e9)]

        # Values far outside should be much smaller than amplitude
        assert jnp.max(far_before) < 0.001 * amplitude, "SFR too large before window"
        assert jnp.max(far_after) < 0.001 * amplitude, "SFR too large after window"

    @pytest.mark.unit
    def test_top_hat_integral(self):
        """Test integrated mass is roughly amplitude * window_width."""
        t = jnp.linspace(1e8, 1.4e10, 2048)  # Fine grid
        amplitude = 1.0
        t_start = 5e9
        t_end = 3e9
        window_width = t_start - t_end
        sfr = top_hat(t, amplitude=amplitude, t_start=t_start, t_end=t_end, smooth_width=5e8)

        # Trapezoidal integration
        dt = jnp.gradient(t)
        mass = jnp.sum(sfr * dt)
        expected_mass = amplitude * window_width

        # Allow 5% error due to sigmoid smoothing
        assert jnp.abs(mass - expected_mass) / expected_mass < 0.05, "Integrated mass incorrect"

    @pytest.mark.unit
    def test_top_hat_gradient_finite(self):
        """Test gradient w.r.t. amplitude is finite and JIT-compatible."""

        def loss(amp):
            t = jnp.linspace(1e8, 1.4e10, 64)
            sfr = top_hat(t, amplitude=amp, t_start=5e9, t_end=3e9)
            return jnp.sum(sfr)

        grad_fn = jax.grad(loss)
        amp = 2.5
        grad_amp = grad_fn(amp)

        assert jnp.isfinite(grad_amp), "Gradient w.r.t. amplitude not finite"
        assert jnp.abs(grad_amp) > 1e-6, "Gradient w.r.t. amplitude suspiciously small"

    @pytest.mark.unit
    def test_top_hat_jit_compatible(self):
        """Test top_hat is JIT-compatible."""

        @jax.jit
        def compute_sfr(t, amp, t_start, t_end):
            return top_hat(t, amplitude=amp, t_start=t_start, t_end=t_end)

        t = jnp.linspace(1e8, 1.4e10, 64)
        sfr = compute_sfr(t, 1.5, 5e9, 3e9)
        assert sfr.shape == t.shape

    @pytest.mark.unit
    def test_top_hat_registry(self):
        """Test top_hat is registered and resolvable."""
        _fn, params, _param_map, _settings = resolve_sfh("top_hat")
        assert "sfh_top_hat_amplitude" in params
        assert "sfh_top_hat_t_start_gyr" in params
        assert "sfh_top_hat_t_end_gyr" in params


class TestGaussianBurst:
    """Tests for gaussian_burst parametric SFH model."""

    @pytest.mark.unit
    def test_gaussian_burst_shape(self):
        """Test output shape matches input."""
        t = jnp.logspace(7, 10.14, 64)
        sfr = gaussian_burst(t, amplitude=5.0, t_peak=1e9, sigma=1e8)
        assert sfr.shape == t.shape

    @pytest.mark.unit
    def test_gaussian_burst_peaks_at_t_peak(self):
        """Test SFR peaks at t_peak."""
        t = jnp.linspace(1e8, 2e9, 512)
        amplitude = 3.0
        t_peak = 1e9
        sigma = 1e8
        sfr = gaussian_burst(t, amplitude=amplitude, t_peak=t_peak, sigma=sigma)

        # Find peak
        peak_idx = jnp.argmax(sfr)
        sfr_peak = sfr[peak_idx]
        t_peak_found = t[peak_idx]

        # Peak should be close to amplitude and near t_peak
        assert jnp.abs(sfr_peak - amplitude) < 0.01 * amplitude
        assert jnp.abs(t_peak_found - t_peak) < 5 * sigma  # Within 5 sigma

    @pytest.mark.unit
    def test_gaussian_burst_integral(self):
        """Test integrated mass is approximately amplitude * sigma * sqrt(2*pi)."""
        t = jnp.linspace(0.0, 3e9, 4096)
        amplitude = 2.0
        t_peak = 1e9
        sigma = 1e8
        sfr = gaussian_burst(t, amplitude=amplitude, t_peak=t_peak, sigma=sigma)

        # Trapezoidal integration
        dt = jnp.gradient(t)
        mass = jnp.sum(sfr * dt)
        expected_mass = amplitude * sigma * jnp.sqrt(2 * jnp.pi)

        # Allow 2% error (due to finite grid)
        assert jnp.abs(mass - expected_mass) / expected_mass < 0.02, "Integrated mass incorrect"

    @pytest.mark.unit
    def test_gaussian_burst_nonzero_everywhere(self):
        """Test Gaussian burst is everywhere non-negative."""
        t = jnp.logspace(7, 10.14, 256)
        sfr = gaussian_burst(t, amplitude=5.0, t_peak=1e9, sigma=1e8)
        assert jnp.all(sfr >= 0.0), "SFR is negative somewhere"

    @pytest.mark.unit
    def test_gaussian_burst_gradient_finite(self):
        """Test gradient w.r.t. amplitude is finite and JIT-compatible."""

        def loss(amp):
            t = jnp.linspace(0.0, 3e9, 64)
            sfr = gaussian_burst(t, amplitude=amp, t_peak=1e9, sigma=1e8)
            return jnp.sum(sfr)

        grad_fn = jax.grad(loss)
        amp = 5.0
        grad_amp = grad_fn(amp)

        assert jnp.isfinite(grad_amp), "Gradient w.r.t. amplitude not finite"
        assert jnp.abs(grad_amp) > 1e-6, "Gradient w.r.t. amplitude suspiciously small"

    @pytest.mark.unit
    def test_gaussian_burst_jit_compatible(self):
        """Test gaussian_burst is JIT-compatible."""

        @jax.jit
        def compute_sfr(t, amp, t_peak, sigma):
            return gaussian_burst(t, amplitude=amp, t_peak=t_peak, sigma=sigma)

        t = jnp.linspace(0.0, 3e9, 64)
        sfr = compute_sfr(t, 5.0, 1e9, 1e8)
        assert sfr.shape == t.shape

    @pytest.mark.unit
    def test_gaussian_burst_registry(self):
        """Test gaussian_burst is registered and resolvable."""
        _fn, params, _param_map, _settings = resolve_sfh("gaussian_burst")
        assert "sfh_gaussian_burst_amplitude" in params
        assert "sfh_gaussian_burst_t_peak_gyr" in params
        assert "sfh_gaussian_burst_sigma_gyr" in params

    @pytest.mark.unit
    def test_gaussian_burst_composition(self):
        """Test gaussian_burst can be composed additively with other models."""
        # Should be able to compose gaussian_burst with any smooth model
        _fn, params, _param_map, _settings = resolve_sfh(["tsnorm", "gaussian_burst"])
        assert "sfh_tsnorm_log_peak_sfr" in params
        assert "sfh_gaussian_burst_amplitude" in params


class TestEvolvingMetallicity:
    """Tests for evolving metallicity support in SFH models."""

    def test_simulate_accepts_z_history(self):
        """Test that sed_from_sfh accepts Z(t) array (evolving metallicity)."""
        pytest.importorskip("tengri.analysis")

        from tengri.analysis.simulate import sed_from_sfh

        try:
            from tengri.io import load_ssp_data
        except (ImportError, ModuleNotFoundError):
            pytest.skip("SSP data module not available")

        try:
            ssp = load_ssp_data()
        except (FileNotFoundError, RuntimeError):
            pytest.skip("SSP data not available")

        t_gyr = jnp.linspace(0.1, 13.7, 32)
        sfr = jnp.ones_like(t_gyr)
        z_history = -0.3 * jnp.ones_like(t_gyr)  # Constant for this test

        result = sed_from_sfh(
            t_gyr,
            sfr,
            ssp,
            log_z=z_history,  # Pass array instead of scalar
        )

        assert "sed" in result
        assert result["sed"].shape[0] > 0
        assert jnp.all(jnp.isfinite(result["sed"]))  # noqa: RUF015

    def test_simulate_z_scalar_vs_array(self):
        """Test scalar and array metallicity give consistent results."""
        pytest.importorskip("tengri.analysis")

        from tengri.analysis.simulate import sed_from_sfh

        try:
            from tengri.io import load_ssp_data
        except (ImportError, ModuleNotFoundError):
            pytest.skip("SSP data module not available")

        try:
            ssp = load_ssp_data()
        except (FileNotFoundError, RuntimeError):
            pytest.skip("SSP data not available")

        t_gyr = jnp.linspace(0.1, 13.7, 32)
        sfr = jnp.ones_like(t_gyr)

        # Constant metallicity (scalar vs constant array)
        result_scalar = sed_from_sfh(t_gyr, sfr, ssp, log_z=-0.3)
        result_array = sed_from_sfh(t_gyr, sfr, ssp, log_z=-0.3 * jnp.ones_like(t_gyr))

        # Results should be nearly identical
        assert jnp.allclose(
            result_scalar["sed"], result_array["sed"], rtol=1e-3
        )  # noqa: RUF015
