# SPDX-License-Identifier: BSD-3-Clause
"""Unit tests for the Lyman-alpha escape fraction in nebular emission.

Synthesizer-inspired: synthesizer/tests/test_emissions.py verifies that
fesc_ly_alpha selectively suppresses Ly-α without touching other lines.

The escape fraction logic (cloudy_grid.py and cue.py):
    lya_scale = (1 - neb_fesc_lya) / max(1 - neb_fesc, 1e-10)
    lum_lya *= lya_scale

Physical properties tested:
- fesc_lya=0.5 reduces Ly-α relative to fesc_lya=0.0
- Other lines (Hα, Hβ, [OIII]) are NOT affected by fesc_lya
- fesc_lya=1.0 makes Ly-α zero
- Gradient flows through fesc_lya
- When fesc_lya == neb_fesc, scale = 1.0 (no extra suppression)
"""

from __future__ import annotations

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.bounds


def fd_grad(f, x: float, eps: float = 1e-4) -> float:
    """Central finite difference: (f(x+eps) - f(x-eps)) / (2*eps)."""
    return float((f(x + eps) - f(x - eps)) / (2.0 * eps))


# ── Pure-formula tests (no grid required) ─────────────────────────
# These test the escape-fraction scaling formula isolated from the grid backend.

_LYA_WAVE = 1215.67  # Å, vacuum Ly-α


def _apply_lya_escape(
    line_waves: jnp.ndarray,
    line_lums: jnp.ndarray,
    neb_fesc: float,
    neb_fesc_lya: float,
) -> jnp.ndarray:
    """Apply Ly-α escape fraction scaling (mirrors cloudy_grid.py logic)."""
    lya_idx = jnp.argmin(jnp.abs(line_waves - _LYA_WAVE))
    lya_scale = (1.0 - neb_fesc_lya) / jnp.maximum(1.0 - neb_fesc, 1e-10)
    return line_lums.at[lya_idx].multiply(lya_scale)


@pytest.fixture
def toy_lines():
    """Minimal line array: Ly-α + Hβ + [OIII]5007 + Hα."""
    waves = jnp.array([1215.67, 4862.68, 5008.24, 6564.61])
    # Nominal luminosities in Lsun
    lums = jnp.array([1.0, 1.0, 1.34, 2.86])
    return waves, lums


class TestLyaScaleFormula:
    """Pure formula tests, independent of CLOUDY grid."""

    def test_fesc_lya_0_leaves_lya_unchanged(self, toy_lines):
        """neb_fesc_lya=0 → no extra suppression on Ly-α."""
        waves, lums = toy_lines
        lums_out = _apply_lya_escape(waves, lums, neb_fesc=0.0, neb_fesc_lya=0.0)
        np.testing.assert_allclose(np.array(lums_out), np.array(lums), rtol=1e-9)

    def test_fesc_lya_half_reduces_lya(self, toy_lines):
        """neb_fesc_lya=0.5 → Ly-α is halved (factor 0.5/(1.0) = 0.5)."""
        waves, lums = toy_lines
        lums_out = _apply_lya_escape(waves, lums, neb_fesc=0.0, neb_fesc_lya=0.5)
        lya_idx = int(jnp.argmin(jnp.abs(waves - _LYA_WAVE)))
        np.testing.assert_allclose(float(lums_out[lya_idx]), float(lums[lya_idx]) * 0.5, rtol=1e-6)

    def test_fesc_lya_1_zeroes_lya(self, toy_lines):
        """neb_fesc_lya=1.0 → Ly-α luminosity = 0."""
        waves, lums = toy_lines
        lums_out = _apply_lya_escape(waves, lums, neb_fesc=0.0, neb_fesc_lya=1.0)
        lya_idx = int(jnp.argmin(jnp.abs(waves - _LYA_WAVE)))
        assert float(lums_out[lya_idx]) == pytest.approx(0.0, abs=1e-12)

    def test_other_lines_unchanged(self, toy_lines):
        """Changing neb_fesc_lya must NOT affect Hβ, [OIII], Hα."""
        waves, lums = toy_lines
        lums_out = _apply_lya_escape(waves, lums, neb_fesc=0.0, neb_fesc_lya=0.7)
        # Check every non-Ly-α line
        for i, (w, l_in, l_out) in enumerate(zip(waves, lums, lums_out)):
            if abs(float(w) - _LYA_WAVE) > 1.0:
                np.testing.assert_allclose(
                    float(l_out),
                    float(l_in),
                    rtol=1e-9,
                    err_msg=f"Line at {float(w):.1f}Å (index {i}) changed under fesc_lya",
                )

    def test_fesc_lya_equal_fesc_gives_scale_1(self, toy_lines):
        """When fesc_lya == neb_fesc, scale = 1 (Ly-α treated like other lines)."""
        waves, lums = toy_lines
        fesc = 0.3
        lums_out = _apply_lya_escape(waves, lums, neb_fesc=fesc, neb_fesc_lya=fesc)
        np.testing.assert_allclose(np.array(lums_out), np.array(lums), rtol=1e-9)

    def test_fesc_lya_greater_than_fesc(self, toy_lines):
        """neb_fesc_lya > neb_fesc → Ly-α more suppressed than other lines."""
        waves, lums = toy_lines
        lums_out = _apply_lya_escape(waves, lums, neb_fesc=0.1, neb_fesc_lya=0.8)
        lya_idx = int(jnp.argmin(jnp.abs(waves - _LYA_WAVE)))
        # Scale = (1-0.8)/(1-0.1) = 0.2/0.9 ≈ 0.222
        expected_scale = (1.0 - 0.8) / (1.0 - 0.1)
        np.testing.assert_allclose(
            float(lums_out[lya_idx]),
            float(lums[lya_idx]) * expected_scale,
            rtol=1e-6,
        )

    def test_gradient_flows_through_fesc_lya(self, toy_lines):
        """Gradient of total line luminosity w.r.t. neb_fesc_lya is finite and nonzero."""
        waves, lums = toy_lines

        def loss(fesc_lya):
            out = _apply_lya_escape(waves, lums, neb_fesc=0.0, neb_fesc_lya=fesc_lya)
            return jnp.sum(out)

        grad_jax = float(jax.grad(loss)(0.3))
        grad_fd = fd_grad(loss, 0.3)
        np.testing.assert_allclose(
            grad_jax,
            grad_fd,
            rtol=1e-3,
            atol=1e-12,
            err_msg=f"autodiff={grad_jax:.4e}, FD={grad_fd:.4e}",
        )
        assert grad_jax != 0.0, "Gradient w.r.t. neb_fesc_lya should be nonzero"

    def test_jit_compatible(self, toy_lines):
        """The escape fraction scaling should be JIT-compilable."""
        waves, lums = toy_lines

        @jax.jit
        def run(fesc_lya):
            return _apply_lya_escape(waves, lums, neb_fesc=0.0, neb_fesc_lya=fesc_lya)

        result = run(0.5)
        chex.assert_equal_shape([result, lums])
        chex.assert_tree_all_finite(result)


class TestLyaEscapePhysicalDirection:
    """Physical ordering: increasing fesc_lya monotonically reduces Ly-α."""

    def test_monotone_suppression(self, toy_lines):
        """Ly-α luminosity decreases monotonically as fesc_lya increases."""
        waves, lums = toy_lines
        lya_idx = int(jnp.argmin(jnp.abs(waves - _LYA_WAVE)))
        fesc_values = [0.0, 0.2, 0.5, 0.8, 0.99]
        prev_lya = float("inf")
        for fesc_lya in fesc_values:
            out = _apply_lya_escape(waves, lums, neb_fesc=0.0, neb_fesc_lya=fesc_lya)
            current_lya = float(out[lya_idx])
            assert current_lya <= prev_lya + 1e-10, (
                f"Ly-α not monotonically decreasing: fesc_lya={fesc_lya} gave {current_lya:.4f} "
                f"> previous {prev_lya:.4f}"
            )
            prev_lya = current_lya

    def test_balmer_lines_not_monotone_with_lya_fesc(self, toy_lines):
        """Balmer line (Hα) luminosity must be CONSTANT as fesc_lya varies."""
        waves, lums = toy_lines
        ha_idx = int(jnp.argmin(jnp.abs(waves - 6564.61)))
        ha_values = []
        for fesc_lya in [0.0, 0.3, 0.7, 1.0]:
            out = _apply_lya_escape(waves, lums, neb_fesc=0.0, neb_fesc_lya=fesc_lya)
            ha_values.append(float(out[ha_idx]))
        # All Hα values should be equal
        np.testing.assert_allclose(
            ha_values,
            [ha_values[0]] * 4,
            rtol=1e-9,
            err_msg="Hα should be unaffected by neb_fesc_lya",
        )
