# SPDX-License-Identifier: BSD-3-Clause
"""Tests for diffuse ionized gas (DIG) line-luminosity mixing.

Mirrors ``test_dig.py``'s cases for ``mix_dig_emission``, one level down:
:func:`~tengri.components.nebular.dig.mix_dig_line_luminosities` shares the
same core (:func:`~tengri.components.nebular.dig._mix_dig_backend_evaluations`)
but calls ``predict_nebular_line_luminosities`` and mixes only the second
element of its ``(line_waves, line_lums)`` return, keeping the HII call's
``line_waves`` (#2221).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.nebular.dig import mix_dig_line_luminosities


def fd_grad(f, x: float, eps: float = 1e-4) -> float:
    """Central finite difference: (f(x+eps) - f(x-eps)) / (2*eps)."""
    return float((f(x + eps) - f(x - eps)) / (2.0 * eps))


pytestmark = pytest.mark.bounds


class MockNebularBackend:
    """Mock backend returning logU-dependent line catalogs for testing.

    ``line_waves`` is offset by ``neb_logU`` so a test can tell which call's
    wavelengths survived a mix: the real backends catalog the same rest
    wavelengths at both ionization parameters, but the mock needs the two
    calls to be numerically distinguishable to pin the "keep HII's waves"
    contract.
    """

    has_free_params = True

    def predict_nebular_line_luminosities(
        self,
        ssp_wave: jnp.ndarray,
        neb_logU: float = -3.0,
        **kwargs,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Return a 3-line catalog whose luminosities scale with logU.

        L = 10^(logU + 3) so that logU=-3 gives L=1, logU=-4 gives L=0.1.
        """
        line_waves = jnp.array([4862.68, 5008.24, 6564.61]) + neb_logU
        line_lums = jnp.ones(3) * 10.0 ** (neb_logU + 3.0)
        return line_waves, line_lums


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


class TestMixDigLineLuminosities:
    """Tests for mix_dig_line_luminosities."""

    def test_dig_frac_zero_returns_pure_hii(self, backend, common_kw):
        """dig_frac=0 should return pure HII line luminosities and waves."""
        line_waves, line_lums = mix_dig_line_luminosities(
            nebular_backend=backend,
            neb_dig_frac=0.0,
            neb_logU=-3.0,
            **common_kw,
        )
        # MockBackend at logU=-3 returns lums=1.0, waves offset by -3.
        assert jnp.allclose(line_lums, jnp.ones(3), atol=1e-12)
        assert jnp.allclose(line_waves, jnp.array([4862.68, 5008.24, 6564.61]) - 3.0, atol=1e-12)

    def test_dig_frac_one_returns_pure_dig_lums_but_hii_waves(self, backend, common_kw):
        """dig_frac=1 mixes to pure DIG luminosities but keeps HII's waves."""
        line_waves, line_lums = mix_dig_line_luminosities(
            nebular_backend=backend,
            neb_dig_frac=1.0,
            neb_logU=-3.0,
            neb_dig_delta_logU=-1.0,
            **common_kw,
        )
        # MockBackend at logU=-4 returns lums=0.1.
        assert jnp.allclose(line_lums, jnp.ones(3) * 0.1, atol=1e-12)
        # Waves must be the HII call's (offset -3), not the DIG call's
        # (offset -4), even though the luminosities are pure DIG.
        assert jnp.allclose(line_waves, jnp.array([4862.68, 5008.24, 6564.61]) - 3.0, atol=1e-12)

    def test_dig_frac_half_returns_average(self, backend, common_kw):
        """dig_frac=0.5 should return the weighted average of luminosities."""
        _, line_lums = mix_dig_line_luminosities(
            nebular_backend=backend,
            neb_dig_frac=0.5,
            neb_logU=-3.0,
            neb_dig_delta_logU=-1.0,
            **common_kw,
        )
        # 0.5 * 1.0 + 0.5 * 0.1 = 0.55
        assert jnp.allclose(line_lums, jnp.ones(3) * 0.55, atol=1e-12)

    def test_output_shape_matches_backend_catalog(self, backend, common_kw):
        """Output should have the same shape as the backend's line catalog."""
        line_waves, line_lums = mix_dig_line_luminosities(
            nebular_backend=backend,
            neb_dig_frac=0.3,
            **common_kw,
        )
        assert line_waves.shape == (3,)
        assert line_lums.shape == (3,)

    def test_custom_delta_logU(self, backend, common_kw):
        """Custom delta_logU should shift the DIG ionization parameter."""
        _, line_lums = mix_dig_line_luminosities(
            nebular_backend=backend,
            neb_dig_frac=1.0,
            neb_logU=-3.0,
            neb_dig_delta_logU=-2.0,
            **common_kw,
        )
        # MockBackend at logU=-5 returns 10^(-5+3) = 0.01
        assert jnp.allclose(line_lums, jnp.ones(3) * 0.01, atol=1e-12)

    def test_jit_compatible(self, backend, common_kw):
        """mix_dig_line_luminosities should be JIT-compilable."""

        @jax.jit
        def f(dig_frac):
            _, line_lums = mix_dig_line_luminosities(
                nebular_backend=backend,
                neb_dig_frac=dig_frac,
                neb_logU=-3.0,
                neb_dig_delta_logU=-1.0,
                **common_kw,
            )
            return line_lums

        result = f(0.5)
        assert jnp.allclose(result, jnp.ones(3) * 0.55, atol=1e-12)

    def test_differentiable_wrt_dig_frac(self, backend, common_kw):
        """mix_dig_line_luminosities should be differentiable w.r.t. dig_frac."""

        def scalar_fn(dig_frac):
            _, line_lums = mix_dig_line_luminosities(
                nebular_backend=backend,
                neb_dig_frac=dig_frac,
                neb_logU=-3.0,
                neb_dig_delta_logU=-1.0,
                **common_kw,
            )
            return jnp.sum(line_lums)

        grad_fn = jax.grad(scalar_fn)
        grad_val = grad_fn(0.5)

        # d/d(f) [ sum( (1-f)*1 + f*0.1 ) ] = sum(0.1 - 1) = -0.9 * n_lines
        expected_grad = -0.9 * 3
        assert jnp.allclose(grad_val, expected_grad, atol=1e-10)

    def test_differentiable_wrt_delta_logU(self, backend, common_kw):
        """mix_dig_line_luminosities should be differentiable w.r.t. delta_logU."""

        def scalar_fn(delta):
            _, line_lums = mix_dig_line_luminosities(
                nebular_backend=backend,
                neb_dig_frac=0.5,
                neb_logU=-3.0,
                neb_dig_delta_logU=delta,
                **common_kw,
            )
            return jnp.sum(line_lums)

        grad_jax = float(jax.grad(scalar_fn)(-1.0))
        grad_fd = fd_grad(scalar_fn, -1.0)
        np.testing.assert_allclose(
            grad_jax, grad_fd, rtol=1e-3, err_msg=f"autodiff={grad_jax:.4e}, FD={grad_fd:.4e}"
        )
        assert not jnp.allclose(grad_jax, 0.0)
        assert np.all(np.isfinite(grad_jax)), (
            "`grad_jax` is non-finite: non-zero is not enough, `nan != 0.0` is True "
            "and a NaN satisfies a non-zero assertion (#2178)"
        )

    def test_kwargs_forwarded_to_backend(self, common_kw):
        """Extra kwargs should be forwarded to the backend, identically both calls."""

        class RecordingBackend:
            has_free_params = True

            def __init__(self):
                self.calls = []

            def predict_nebular_line_luminosities(self, ssp_wave, **kwargs):
                self.calls.append(kwargs)
                return jnp.array([4862.68, 5008.24, 6564.61]), jnp.ones(3)

        rec = RecordingBackend()
        mix_dig_line_luminosities(
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
        """With lower DIG logU, increasing dig_frac should decrease luminosity."""
        fracs = jnp.linspace(0.0, 1.0, 11)
        totals = jnp.array(
            [
                jnp.sum(
                    mix_dig_line_luminosities(
                        nebular_backend=backend,
                        neb_dig_frac=float(f),
                        neb_logU=-3.0,
                        neb_dig_delta_logU=-1.0,
                        **common_kw,
                    )[1]
                )
                for f in fracs
            ]
        )
        # Each successive value should be <= previous (DIG is fainter)
        assert jnp.all(jnp.diff(totals) <= 0.0)
