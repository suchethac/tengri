"""Unit tests for DIG (diffuse ionized gas) nebular emission mixing.

Tests three exact algebraic identities:

    frac=0  →  output == pure HII spectrum
    frac=1  →  output == pure DIG spectrum (logU_DIG = logU_HII + delta_logU)
    frac=0.5 → output == (HII + DIG) / 2  (exact arithmetic mean)

Uses a deterministic MockNebularBackend so the test runs without any data files.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.nebular.dig import mix_dig_emission

pytestmark = pytest.mark.bounds


def fd_grad(f, x: float, eps: float = 1e-4) -> float:
    """Central finite difference: (f(x+eps) - f(x-eps)) / (2*eps)."""
    return float((f(x + eps) - f(x - eps)) / (2.0 * eps))


# ── Minimal mock backend ──────────────────────────────────────────


class MockNebularBackend:
    """Fake backend whose SED is a known linear function of logU.

    predict_nebular_sed returns  wave * (logU + offset)  so that:
    - outputs are distinct for different logU values
    - the exact DIG-mixing formula is easy to verify numerically
    """

    def __init__(self, n_wave: int = 20, offset: float = 5.0) -> None:
        self._wave = jnp.linspace(3000.0, 7000.0, n_wave)
        self._offset = offset

    def predict_nebular_sed(
        self,
        ssp_wave,
        ssp_weights,
        ssp_log_ages_yr,
        log_z,
        neb_logU=-3.0,
        neb_logZ_gas=None,
        neb_fesc=0.0,
        neb_fesc_lya=0.0,
        line_sigma_aa=0.0,
        **kwargs,
    ) -> jnp.ndarray:
        """Return wave * (logU + offset) so the SED depends linearly on logU."""
        return self._wave * (neb_logU + self._offset)


# ── Shared fixtures ───────────────────────────────────────────────

N_WAVE = 20
N_AGE = 5


@pytest.fixture
def mock_backend():
    return MockNebularBackend(n_wave=N_WAVE)


@pytest.fixture
def common_kw():
    """Keyword arguments that every mix_dig_emission call needs (besides neb_logU)."""
    return dict(
        ssp_wave=jnp.linspace(3000.0, 7000.0, N_WAVE),
        ssp_weights=jnp.ones(N_AGE) * 1e8,
        ssp_log_ages_yr=jnp.linspace(6.5, 10.0, N_AGE),
        log_z=-1.848,
        neb_fesc=0.0,
        neb_fesc_lya=0.0,
        line_sigma_aa=0.0,
    )


# ── Core algebraic identity tests ─────────────────────────────────


def test_dig_frac_zero_equals_pure_hii(mock_backend, common_kw):
    """neb_dig_frac=0 must return exactly the pure HII spectrum."""
    logU = -3.0

    out = mix_dig_emission(mock_backend, neb_logU=logU, neb_dig_frac=0.0, **common_kw)
    hii = mock_backend.predict_nebular_sed(neb_logU=logU, **common_kw)

    assert jnp.allclose(out, hii), (
        f"frac=0 output differs from pure HII: max diff = {jnp.max(jnp.abs(out - hii)):.3g}"
    )


def test_dig_frac_one_equals_pure_dig(mock_backend, common_kw):
    """neb_dig_frac=1 must return exactly the pure DIG spectrum."""
    logU = -3.0
    delta_logU = -1.0
    logU_dig = logU + delta_logU

    out = mix_dig_emission(
        mock_backend,
        neb_logU=logU,
        neb_dig_frac=1.0,
        neb_dig_delta_logU=delta_logU,
        **common_kw,
    )
    dig = mock_backend.predict_nebular_sed(neb_logU=logU_dig, **common_kw)

    assert jnp.allclose(out, dig), (
        f"frac=1 output differs from pure DIG: max diff = {jnp.max(jnp.abs(out - dig)):.3g}"
    )


def test_dig_frac_half_equals_arithmetic_mean(mock_backend, common_kw):
    """neb_dig_frac=0.5 must equal (HII + DIG) / 2 exactly."""
    logU = -3.0
    delta_logU = -1.0
    logU_dig = logU + delta_logU

    out = mix_dig_emission(
        mock_backend,
        neb_logU=logU,
        neb_dig_frac=0.5,
        neb_dig_delta_logU=delta_logU,
        **common_kw,
    )
    hii = mock_backend.predict_nebular_sed(neb_logU=logU, **common_kw)
    dig = mock_backend.predict_nebular_sed(neb_logU=logU_dig, **common_kw)
    expected = 0.5 * hii + 0.5 * dig

    max_diff = jnp.max(jnp.abs(out - expected))
    assert jnp.allclose(out, expected, atol=1e-6), (
        f"frac=0.5 output is not the arithmetic mean: max diff = {max_diff:.3g}"
    )


# ── Intermediate fractions and custom delta_logU ──────────────────


@pytest.mark.parametrize("frac", [0.1, 0.3, 0.7, 0.9])
def test_dig_arbitrary_fraction(mock_backend, common_kw, frac):
    """Arbitrary fractions satisfy the linear mixing formula exactly."""
    logU = -2.5
    delta_logU = -1.5

    out = mix_dig_emission(
        mock_backend,
        neb_logU=logU,
        neb_dig_frac=frac,
        neb_dig_delta_logU=delta_logU,
        **common_kw,
    )
    hii = mock_backend.predict_nebular_sed(neb_logU=logU, **common_kw)
    dig = mock_backend.predict_nebular_sed(neb_logU=logU + delta_logU, **common_kw)
    expected = (1.0 - frac) * hii + frac * dig

    assert jnp.allclose(out, expected, atol=1e-5), (
        f"frac={frac}: mixing formula violated: max diff = {jnp.max(jnp.abs(out - expected)):.3g}"
    )


# ── JIT compatibility (traced neb_dig_frac) ───────────────────────


def test_dig_jit_traced_frac(mock_backend, common_kw):
    """mix_dig_emission runs correctly under jit with a traced neb_dig_frac."""

    @jax.jit
    def fn(frac):
        return mix_dig_emission(mock_backend, neb_logU=-3.0, neb_dig_frac=frac, **common_kw)

    frac = jnp.array(0.4)
    out = fn(frac)
    assert jnp.all(jnp.isfinite(out)), "JIT output contains non-finite values"
    assert out.shape == (N_WAVE,), f"Shape mismatch: {out.shape}"


def test_dig_jit_consistent_with_python(mock_backend, common_kw):
    """JIT result matches Python-mode result for the same inputs."""

    @jax.jit
    def fn(frac):
        return mix_dig_emission(mock_backend, neb_logU=-3.0, neb_dig_frac=frac, **common_kw)

    frac = jnp.array(0.6)
    jit_out = fn(frac)
    py_out = mix_dig_emission(mock_backend, neb_logU=-3.0, neb_dig_frac=float(frac), **common_kw)
    assert jnp.allclose(jit_out, py_out, atol=1e-6), (
        f"JIT/Python mismatch: max diff = {jnp.max(jnp.abs(jit_out - py_out)):.3g}"
    )


# ── Gradient check — gradient w.r.t. neb_dig_frac is finite ───────


def test_dig_grad_wrt_frac_finite(mock_backend, common_kw):
    """jax.grad w.r.t. neb_dig_frac is finite (linear mixing → constant gradient)."""

    def fn(frac):
        return jnp.sum(
            mix_dig_emission(mock_backend, neb_logU=-3.0, neb_dig_frac=frac, **common_kw)
        )

    grad_jax = float(jax.grad(fn)(jnp.array(0.5)))
    grad_fd = fd_grad(
        lambda f: float(
            jnp.sum(mix_dig_emission(mock_backend, neb_logU=-3.0, neb_dig_frac=f, **common_kw))
        ),
        0.5,
    )
    np.testing.assert_allclose(
        grad_jax, grad_fd, rtol=1e-3, err_msg=f"autodiff={grad_jax:.4e}, FD={grad_fd:.4e}"
    )


# ── Output shape and finiteness ───────────────────────────────────


def test_dig_output_shape(mock_backend, common_kw):
    """Output shape matches ssp_wave shape for all fractions."""
    for frac in (0.0, 0.5, 1.0):
        out = mix_dig_emission(mock_backend, neb_logU=-3.0, neb_dig_frac=frac, **common_kw)
        assert out.shape == (N_WAVE,), f"frac={frac}: shape {out.shape} != ({N_WAVE},)"


def test_dig_output_finite(mock_backend, common_kw):
    """Output is finite for all fractions when backend returns finite values."""
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        out = mix_dig_emission(mock_backend, neb_logU=-3.0, neb_dig_frac=frac, **common_kw)
        assert jnp.all(jnp.isfinite(out)), f"frac={frac}: output contains non-finite values"
