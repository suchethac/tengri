# SPDX-License-Identifier: BSD-3-Clause
"""Unit tests for the Li et al. (2008, ApJ, 685, 1046) attenuation curve.

The Li08 curve uses 4 dimensionless parameters (c1-c4) in an analytical
form with Lorentzian-like terms for the UV/optical continuum, far-UV rise,
and 2175 A UV bump.  Reference values are computed from the formula in
Markov et al. (2025, arXiv:2504.12378, Eq. 1).

References
----------
- Li, A., Liang, S. L., Kann, D. A., et al. 2008, ApJ, 685, 1046
- Markov, V., et al. 2023, A&A, 679, A12
- Markov, V., et al. 2025, A&A (arXiv:2504.12378)
"""

import chex
import pytest

pytestmark = pytest.mark.regression_paper
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
from tengri.components.dust.attenuation import DUST_LAWS, li08
from tests._jit_parity import assert_jit_matches_eager


def fd_grad(f, x: float, eps: float = 1e-4) -> float:
    """Central finite difference: (f(x+eps) - f(x-eps)) / (2*eps)."""
    return float((f(x + eps) - f(x - eps)) / (2.0 * eps))


WAVS = jnp.array([912.0, 1216.0, 1500.0, 2175.0, 3000.0, 5500.0, 10000.0, 20000.0])
# Default params: c1=6, c2=4, c3=2, c4=0.04
DEFAULT_C1, DEFAULT_C2, DEFAULT_C3, DEFAULT_C4 = 6.0, 4.0, 2.0, 0.04


def _reference_li08(wave_aa, c1, c2, c3, c4):
    """Pure-numpy reference implementation of Eq. 1 in Markov+2025."""
    lam = wave_aa / 1e4  # micron
    t1 = c1 / ((lam / 0.08) ** c2 + (0.08 / lam) ** c2 + c3)
    d0 = 6.88**c2 + 0.145**c2 + c3
    norm_c1 = c1 / d0
    norm_c4 = c4 / 4.60
    t2 = 233.0 * (1.0 - norm_c1 - norm_c4) / ((lam / 0.046) ** 2 + (0.046 / lam) ** 2 + 90.0)
    t3 = c4 / ((lam / 0.2175) ** 2 + (0.2175 / lam) ** 2 - 1.95)
    a_lam = t1 + t2 + t3
    # Evaluate at V-band
    lv = 0.55
    t1v = c1 / ((lv / 0.08) ** c2 + (0.08 / lv) ** c2 + c3)
    t2v = 233.0 * (1.0 - norm_c1 - norm_c4) / ((lv / 0.046) ** 2 + (0.046 / lv) ** 2 + 90.0)
    t3v = c4 / ((lv / 0.2175) ** 2 + (0.2175 / lv) ** 2 - 1.95)
    a_v = t1v + t2v + t3v
    return np.clip(a_lam / a_v, 0.0, None)


class TestRegistry:
    """Li08 should be in the DUST_LAWS registry."""

    def test_registered(self):
        assert "li08" in DUST_LAWS


class TestReferenceValues:
    """Verify against independent numpy implementation of Markov+2025 Eq. 1."""

    def test_default_params(self):
        ref = _reference_li08(np.array(WAVS), DEFAULT_C1, DEFAULT_C2, DEFAULT_C3, DEFAULT_C4)
        tng = np.array(li08(WAVS))
        np.testing.assert_allclose(tng, ref, rtol=1e-10)

    @pytest.mark.parametrize(
        "c1,c2,c3,c4",
        [
            (5.0, 5.5, 1.5, 0.0),  # SMC-like (steep, no bump)
            (3.5, 2.5, 3.0, 0.0),  # Calzetti-like (flat, no bump)
            (6.0, 4.0, 2.0, 0.08),  # Strong bump
            (8.0, 3.0, 1.0, 0.02),  # Arbitrary
        ],
    )
    def test_various_params(self, c1, c2, c3, c4):
        ref = _reference_li08(np.array(WAVS), c1, c2, c3, c4)
        tng = np.array(li08(WAVS, dust_c1=c1, dust_c2=c2, dust_c3=c3, dust_c4=c4))
        np.testing.assert_allclose(tng, ref, rtol=1e-10)


class TestPhysicalProperties:
    """Physical sanity checks for the attenuation curve."""

    def test_normalized_at_v_band(self):
        """k(5500 A) should be ~1.0."""
        k = li08(jnp.array([5500.0]))
        np.testing.assert_allclose(float(k[0]), 1.0, atol=0.01)

    def test_positive_output(self):
        """k(lambda) >= 0 everywhere."""
        wavs = jnp.linspace(500.0, 50000.0, 2000)
        k = li08(wavs)
        assert jnp.all(k >= 0.0)

    def test_uv_greater_than_optical(self):
        """UV attenuation should exceed optical."""
        k = li08(WAVS)
        k_1500 = float(k[2])  # 1500 A
        k_5500 = float(k[5])  # 5500 A
        assert k_1500 > k_5500

    def test_bump_present_with_c4(self):
        """With c4 > 0, there should be a local maximum near 2175 A."""
        wavs = jnp.linspace(1800.0, 2600.0, 200)
        k = li08(wavs, dust_c4=0.06)
        k_np = np.array(k)
        # Find local max
        peak_idx = np.argmax(k_np)
        peak_wave = float(wavs[peak_idx])
        assert 2050.0 < peak_wave < 2300.0, f"Bump peak at {peak_wave} A, expected ~2175"

    def test_no_bump_with_c4_zero(self):
        """With c4=0, the 2175 A bump should vanish."""
        wavs = jnp.linspace(1800.0, 2600.0, 200)
        k_bump = np.array(li08(wavs, dust_c4=0.06))
        k_flat = np.array(li08(wavs, dust_c4=0.0))
        # The bump version should have higher values near 2175 A
        idx_2175 = np.argmin(np.abs(np.array(wavs) - 2175.0))
        assert k_bump[idx_2175] > k_flat[idx_2175]

    def test_higher_c2_reduces_uv_ratio(self):
        """Higher c2 reduces k(UV)/k(V) because t1 collapses at 1500 A.
        The t1 Drude denominator grows as (1500/800)^c2, so increasing c2 from
        2.5 to 5.5 drops t1(1500 A) from ~0.85 to ~0.18. The t2 far-UV term
        barely changes. Net: k(1500 A)/k(5500 A) decreases (~3.1 -> ~2.5).
        """
        wavs = jnp.array([1500.0, 5500.0])
        k_low_c2 = li08(wavs, dust_c2=2.5)
        k_high_c2 = li08(wavs, dust_c2=5.5)
        ratio_low = float(k_low_c2[0] / k_low_c2[1])
        ratio_high = float(k_high_c2[0] / k_high_c2[1])
        assert ratio_high < ratio_low


class TestJAXCompatibility:
    """JAX JIT and autodiff compatibility."""

    def test_jit_compatible(self):
        k = assert_jit_matches_eager(li08, WAVS)
        chex.assert_shape(k, (8,))

    def test_gradient_wrt_c1(self):
        def loss(c1):
            return li08(WAVS, dust_c1=c1).sum()

        grad_jax = float(jax.grad(loss)(6.0))
        grad_fd = fd_grad(loss, 6.0)
        np.testing.assert_allclose(
            grad_jax, grad_fd, rtol=1e-3, err_msg=f"autodiff={grad_jax:.4e}, FD={grad_fd:.4e}"
        )

    def test_gradient_wrt_c4(self):
        def loss(c4):
            return li08(WAVS, dust_c4=c4).sum()

        grad_jax = float(jax.grad(loss)(0.04))
        grad_fd = fd_grad(loss, 0.04)
        np.testing.assert_allclose(
            grad_jax, grad_fd, rtol=1e-3, err_msg=f"autodiff={grad_jax:.4e}, FD={grad_fd:.4e}"
        )

    def test_vmap_over_wavelengths(self):
        """Should work with vmap for batched evaluation."""
        batch_wavs = jnp.stack([WAVS, WAVS * 1.1])
        k_batch = jax.vmap(li08)(batch_wavs)
        chex.assert_shape(k_batch, (2, 8))

    def test_gradient_wrt_all_params(self):
        """Full gradient should be finite for all 4 parameters."""

        def loss(c1, c2, c3, c4):
            return li08(WAVS, dust_c1=c1, dust_c2=c2, dust_c3=c3, dust_c4=c4).sum()

        def loss_c1_only(c1):
            return li08(WAVS, dust_c1=c1, dust_c2=4.0, dust_c3=2.0, dust_c4=0.04).sum()

        def loss_c2_only(c2):
            return li08(WAVS, dust_c1=6.0, dust_c2=c2, dust_c3=2.0, dust_c4=0.04).sum()

        def loss_c3_only(c3):
            return li08(WAVS, dust_c1=6.0, dust_c2=4.0, dust_c3=c3, dust_c4=0.04).sum()

        def loss_c4_only(c4):
            return li08(WAVS, dust_c1=6.0, dust_c2=4.0, dust_c3=2.0, dust_c4=c4).sum()

        losses = [loss_c1_only, loss_c2_only, loss_c3_only, loss_c4_only]
        x_vals = [6.0, 4.0, 2.0, 0.04]
        param_names = ["c1", "c2", "c3", "c4"]
        for i, (loss_fn, x_val, name) in enumerate(zip(losses, x_vals, param_names)):
            grad_jax = float(jax.grad(loss_fn)(x_val))
            grad_fd = fd_grad(loss_fn, x_val)
            np.testing.assert_allclose(
                grad_jax,
                grad_fd,
                rtol=1e-3,
                err_msg=(
                    f"Gradient w.r.t. param {i} ({name}): "
                    f"autodiff={grad_jax:.4e}, FD={grad_fd:.4e}"
                ),
            )
