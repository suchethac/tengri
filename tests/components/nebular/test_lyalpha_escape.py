# SPDX-License-Identifier: BSD-3-Clause
"""Unit tests for the Lyman-alpha escape fraction in nebular emission.

``neb_fesc_lya`` is the Lyα escape fraction: the Lyα luminosity is multiplied
by it (1 = no extra scaling, 0 = Lyα removed). The helper under test is the
single shared implementation every photoionized backend calls
(``apply_lya_escape`` in ``components/nebular/_shared.py``).

Physical properties tested:
- neb_fesc_lya = 1 leaves Ly-alpha unchanged
- neb_fesc_lya = 0.2 gives 0.2 times the unattenuated Ly-alpha flux
- Other lines (H-alpha, H-beta, [OIII]) are NOT affected by neb_fesc_lya
- neb_fesc_lya = 0 removes Ly-alpha
- Gradient flows through neb_fesc_lya
"""

from __future__ import annotations

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.nebular._shared import apply_lya_escape

pytestmark = pytest.mark.bounds

_LYA_WAVE = 1215.67  # Angstrom, vacuum Ly-alpha
_HA_WAVE = 6564.61  # Angstrom, vacuum H-alpha


def fd_grad(f, x: float, eps: float = 1e-4) -> float:
    """Central finite difference: (f(x+eps) - f(x-eps)) / (2*eps)."""
    return float((f(x + eps) - f(x - eps)) / (2.0 * eps))


@pytest.fixture
def toy_lines():
    """Minimal line array: Ly-alpha + H-beta + [OIII]5007 + H-alpha."""
    waves = jnp.array([1215.67, 4862.68, 5008.24, 6564.61])
    lums = jnp.array([1.0, 1.0, 1.34, 2.86])  # arbitrary units
    return waves, lums


def _lya_index(waves) -> int:
    return int(jnp.argmin(jnp.abs(waves - _LYA_WAVE)))


class TestLyaEscapeScaling:
    """The shared helper multiplies Ly-alpha by neb_fesc_lya and nothing else."""

    def test_unity_leaves_lya_unchanged(self, toy_lines):
        waves, lums = toy_lines
        out = apply_lya_escape(lums, waves, 1.0)
        np.testing.assert_allclose(np.array(out), np.array(lums), rtol=1e-12)

    def test_fesc_lya_02_gives_02_times_unattenuated_lya(self, toy_lines):
        """At neb_fesc_lya = 0.2, Ly-alpha flux = 0.2 x the no-attenuation flux."""
        waves, lums = toy_lines
        lya_idx = _lya_index(waves)
        out = apply_lya_escape(lums, waves, 0.2)
        np.testing.assert_allclose(float(out[lya_idx]), 0.2 * float(lums[lya_idx]), rtol=1e-6)

    def test_zero_removes_lya(self, toy_lines):
        waves, lums = toy_lines
        out = apply_lya_escape(lums, waves, 0.0)
        assert float(out[_lya_index(waves)]) == pytest.approx(0.0, abs=1e-12)

    def test_other_lines_unchanged(self, toy_lines):
        """Changing neb_fesc_lya must NOT affect H-beta, [OIII] or H-alpha."""
        waves, lums = toy_lines
        out = apply_lya_escape(lums, waves, 0.3)
        for i, (w, l_in, l_out) in enumerate(zip(waves, lums, out)):
            if abs(float(w) - _LYA_WAVE) > 1.0:
                np.testing.assert_allclose(
                    float(l_out),
                    float(l_in),
                    rtol=1e-9,
                    err_msg=f"Line at {float(w):.1f} A (index {i}) changed under neb_fesc_lya",
                )

    def test_gradient_flows_through_neb_fesc_lya(self, toy_lines):
        """Gradient of total line luminosity w.r.t. neb_fesc_lya is finite and nonzero."""
        waves, lums = toy_lines

        def loss(fesc_lya):
            return jnp.sum(apply_lya_escape(lums, waves, fesc_lya))

        grad_jax = float(jax.grad(loss)(0.3))
        grad_fd = fd_grad(loss, 0.3)
        np.testing.assert_allclose(grad_jax, grad_fd, rtol=1e-3, atol=1e-12)
        assert np.isfinite(grad_jax), "gradient is non-finite (#2178)"
        assert grad_jax != 0.0

    def test_jit_compatible(self, toy_lines):
        waves, lums = toy_lines

        @jax.jit
        def run(fesc_lya):
            return apply_lya_escape(lums, waves, fesc_lya)

        result = run(0.5)
        chex.assert_equal_shape([result, lums])
        chex.assert_tree_all_finite(result)


class TestLyaEscapeMonotone:
    """Increasing neb_fesc_lya monotonically raises Ly-alpha; Balmer lines stay fixed."""

    def test_monotone_in_fesc_lya(self, toy_lines):
        waves, lums = toy_lines
        lya_idx = _lya_index(waves)
        prev = -float("inf")
        for fesc_lya in [0.0, 0.2, 0.5, 0.8, 1.0]:
            current = float(apply_lya_escape(lums, waves, fesc_lya)[lya_idx])
            assert current >= prev - 1e-10
            prev = current

    def test_halpha_constant_in_fesc_lya(self, toy_lines):
        waves, lums = toy_lines
        ha_idx = int(jnp.argmin(jnp.abs(waves - _HA_WAVE)))
        values = [float(apply_lya_escape(lums, waves, f)[ha_idx]) for f in [0.0, 0.3, 0.7, 1.0]]
        np.testing.assert_allclose(values, [values[0]] * 4, rtol=1e-9)
