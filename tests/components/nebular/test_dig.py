# SPDX-License-Identifier: BSD-3-Clause
"""Tests for diffuse ionized gas (DIG) nebular emission mixing."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.nebular.dig import mix_dig_emission


def fd_grad(f, x: float, eps: float = 1e-4) -> float:
    """Central finite difference: (f(x+eps) - f(x-eps)) / (2*eps)."""
    return float((f(x + eps) - f(x - eps)) / (2.0 * eps))


pytestmark = pytest.mark.bounds


class MockNebularBackend:
    """Mock backend returning logU-dependent SEDs for testing."""

    has_free_params = True

    def predict_nebular_sed(
        self,
        ssp_wave: jnp.ndarray,
        neb_logU: float = -3.0,
        **kwargs,
    ) -> jnp.ndarray:
        """Return a simple SED that scales with logU.

        L = 10^(logU + 3) so that logU=-3 gives L=1, logU=-4 gives L=0.1.
        """
        return jnp.ones_like(ssp_wave) * 10.0 ** (neb_logU + 3.0)


@pytest.fixture()
def backend():
    return MockNebularBackend()


@pytest.fixture()
def wave():
    return jnp.linspace(3000.0, 10000.0, 100)


@pytest.fixture()
def weights():
    return jnp.ones(50) / 50.0


@pytest.fixture()
def log_ages():
    return jnp.linspace(6.0, 10.0, 50)


@pytest.fixture()
def common_kw(wave, weights, log_ages):
    return dict(
        ssp_wave=wave,
        ssp_weights=weights,
        ssp_log_ages_yr=log_ages,
        log_z=-2.0,
    )


class TestMixDigEmission:
    """Tests for mix_dig_emission."""

    def test_dig_frac_zero_returns_pure_hii(self, backend, common_kw):
        """dig_frac=0 should return pure HII emission."""
        result = mix_dig_emission(
            nebular_backend=backend,
            neb_dig_frac=0.0,
            neb_logU=-3.0,
            **common_kw,
        )
        # MockBackend at logU=-3 returns 1.0
        expected = jnp.ones_like(common_kw["ssp_wave"]) * 1.0
        assert jnp.allclose(result, expected, atol=1e-12)

    def test_dig_frac_one_returns_pure_dig(self, backend, common_kw):
        """dig_frac=1 should return pure DIG emission."""
        result = mix_dig_emission(
            nebular_backend=backend,
            neb_dig_frac=1.0,
            neb_logU=-3.0,
            neb_dig_delta_logU=-1.0,
            **common_kw,
        )
        # MockBackend at logU=-4 returns 0.1
        expected = jnp.ones_like(common_kw["ssp_wave"]) * 0.1
        assert jnp.allclose(result, expected, atol=1e-12)

    def test_dig_frac_half_returns_average(self, backend, common_kw):
        """dig_frac=0.5 should return the weighted average."""
        result = mix_dig_emission(
            nebular_backend=backend,
            neb_dig_frac=0.5,
            neb_logU=-3.0,
            neb_dig_delta_logU=-1.0,
            **common_kw,
        )
        # 0.5 * 1.0 + 0.5 * 0.1 = 0.55
        expected = jnp.ones_like(common_kw["ssp_wave"]) * 0.55
        assert jnp.allclose(result, expected, atol=1e-12)

    def test_output_shape_matches_wave(self, backend, common_kw):
        """Output should have the same shape as ssp_wave."""
        result = mix_dig_emission(
            nebular_backend=backend,
            neb_dig_frac=0.3,
            **common_kw,
        )
        assert result.shape == common_kw["ssp_wave"].shape

    def test_custom_delta_logU(self, backend, common_kw):
        """Custom delta_logU should shift the DIG ionization parameter."""
        result = mix_dig_emission(
            nebular_backend=backend,
            neb_dig_frac=1.0,
            neb_logU=-3.0,
            neb_dig_delta_logU=-2.0,
            **common_kw,
        )
        # MockBackend at logU=-5 returns 10^(-5+3) = 0.01
        expected = jnp.ones_like(common_kw["ssp_wave"]) * 0.01
        assert jnp.allclose(result, expected, atol=1e-12)

    def test_jit_compatible(self, backend, common_kw):
        """mix_dig_emission should be JIT-compilable."""

        @jax.jit
        def f(dig_frac):
            return mix_dig_emission(
                nebular_backend=backend,
                neb_dig_frac=dig_frac,
                neb_logU=-3.0,
                neb_dig_delta_logU=-1.0,
                **common_kw,
            )

        result = f(0.5)
        expected = jnp.ones_like(common_kw["ssp_wave"]) * 0.55
        assert jnp.allclose(result, expected, atol=1e-12)

    def test_differentiable_wrt_dig_frac(self, backend, common_kw):
        """mix_dig_emission should be differentiable w.r.t. dig_frac."""

        def scalar_fn(dig_frac):
            sed = mix_dig_emission(
                nebular_backend=backend,
                neb_dig_frac=dig_frac,
                neb_logU=-3.0,
                neb_dig_delta_logU=-1.0,
                **common_kw,
            )
            return jnp.sum(sed)

        grad_fn = jax.grad(scalar_fn)
        grad_val = grad_fn(0.5)

        # d/d(f) [ sum( (1-f)*1 + f*0.1 ) ] = sum(0.1 - 1) = -0.9 * n_wave
        n_wave = common_kw["ssp_wave"].shape[0]
        expected_grad = -0.9 * n_wave
        assert jnp.allclose(grad_val, expected_grad, atol=1e-10)

    def test_differentiable_wrt_delta_logU(self, backend, common_kw):
        """mix_dig_emission should be differentiable w.r.t. delta_logU."""

        def scalar_fn(delta):
            sed = mix_dig_emission(
                nebular_backend=backend,
                neb_dig_frac=0.5,
                neb_logU=-3.0,
                neb_dig_delta_logU=delta,
                **common_kw,
            )
            return jnp.sum(sed)

        grad_jax = float(jax.grad(scalar_fn)(-1.0))
        grad_fd = fd_grad(scalar_fn, -1.0)
        np.testing.assert_allclose(
            grad_jax, grad_fd, rtol=1e-3, err_msg=f"autodiff={grad_jax:.4e}, FD={grad_fd:.4e}"
        )
        assert not jnp.allclose(grad_jax, 0.0)

    def test_kwargs_forwarded_to_backend(self, common_kw):
        """Extra kwargs should be forwarded to the backend."""

        class RecordingBackend:
            has_free_params = True

            def __init__(self):
                self.calls = []

            def predict_nebular_sed(self, ssp_wave, **kwargs):
                self.calls.append(kwargs)
                return jnp.ones_like(ssp_wave)

        rec = RecordingBackend()
        mix_dig_emission(
            nebular_backend=rec,
            neb_dig_frac=0.5,
            neb_logU=-3.0,
            neb_dig_delta_logU=-1.0,
            ionspec_index1=0.5,
            **common_kw,
        )
        # Two calls: HII and DIG
        assert len(rec.calls) == 2
        # Both should receive the extra kwarg
        assert rec.calls[0]["ionspec_index1"] == 0.5
        assert rec.calls[1]["ionspec_index1"] == 0.5
        # First call should have logU=-3, second logU=-4
        assert rec.calls[0]["neb_logU"] == -3.0
        assert rec.calls[1]["neb_logU"] == -4.0

    def test_monotonic_in_dig_frac(self, backend, common_kw):
        """With lower DIG logU, increasing dig_frac should decrease flux."""
        fracs = jnp.linspace(0.0, 1.0, 11)
        fluxes = jnp.array(
            [
                jnp.sum(
                    mix_dig_emission(
                        nebular_backend=backend,
                        neb_dig_frac=float(f),
                        neb_logU=-3.0,
                        neb_dig_delta_logU=-1.0,
                        **common_kw,
                    )
                )
                for f in fracs
            ]
        )
        # Each successive value should be <= previous (DIG is fainter)
        assert jnp.all(jnp.diff(fluxes) <= 0.0)
