# SPDX-License-Identifier: BSD-3-Clause
"""Regression test for bug #2739: nthcomp interpolation coordinates in input dtype.

Bug: Kubota-Done warm Comptonization (nthcomp) casts interpolation coordinates
to float32, causing 1e-6 SED differences between JIT and eager evaluation.

Fix: Keep interpolation coordinates and weights in input dtype (float64 under x64,
float32 in pure-float32 mode). Template table values stay float32 but are promoted
before arithmetic.

Measured defect:
  - JIT vs eager at M_BH 7–7.5: 2.2e-6 difference in kubota_done disc L_nu
  - AD vs FD for agn_gamma_warm: 1.03e-4 to 2.2e-4 error (1e-4 bar)
"""

import jax
import jax.numpy as jnp
import pytest

from tengri.components.agn._nthcomp import load_nthcomp_table, nthcomp_lnu_interp


@pytest.fixture
def nthcomp_table():
    """Load nthcomp templates, skipping if unavailable."""
    table = load_nthcomp_table()
    if table is None:
        pytest.skip("nthcomp templates not available")
    return table


class TestNthcompJitEagerAgreement:
    """JIT vs eager agreement tests for nthcomp coordinates dtype fix."""

    @pytest.mark.regression_bug
    def test_jit_eager_agreement_lnu_shape(self, nthcomp_table):
        """JIT and eager evaluation agree to 1e-9 relative on L_nu at interior nodes.

        Tests nthcomp L_nu at various temperatures and photon indices. The
        interpolation must agree to relative 1e-9 between JIT and eager on every
        wavelength node where L_nu > 1e-6 of peak.
        """
        # Sample a range of temperatures and photon indices
        nu = jnp.logspace(10, 20, 50)  # Hz grid
        gammas = jnp.array([1.5, 1.8, 2.2])
        kTes = jnp.array([1e-2, 1e-1, 1.0])  # keV
        kTbbs = jnp.array([1e-4, 1e-3, 1e-2])  # keV

        jitted = jax.jit(
            lambda g, t, bb: nthcomp_lnu_interp(nu, g, t, bb, _template=nthcomp_table)
        )

        for gamma in gammas:
            for kTe in kTes:
                for kTbb in kTbbs:
                    eager_result = nthcomp_lnu_interp(
                        nu, gamma, kTe, kTbb, _template=nthcomp_table
                    )
                    jit_result = jitted(gamma, kTe, kTbb)

                    # Mask out low values (below 1e-6 of peak)
                    peak = jnp.max(eager_result)
                    mask = eager_result > 1e-6 * peak

                    if jnp.any(mask):
                        rel_diff = jnp.abs(jit_result[mask] - eager_result[mask]) / (
                            jnp.abs(eager_result[mask]) + 1e-30
                        )
                        max_rel_diff = jnp.max(rel_diff)
                        assert max_rel_diff < 1e-9, (
                            f"JIT/eager L_nu differ by {max_rel_diff:.2e} "
                            f"at gamma={gamma}, kTe={kTe}, kTbb={kTbb}"
                        )

    @pytest.mark.regression_bug
    def test_grad_ad_vs_fd(self, nthcomp_table):
        """AD vs FD gradients for gamma_warm and kt_warm agree to 1e-5.

        Tests automatic differentiation (AD) vs central finite difference (FD)
        for photon index and electron temperature. Verifies that FD is in its
        truncation-balanced regime by checking two step sizes agree.
        """
        nu = jnp.logspace(10, 20, 30)
        gamma = 2.0
        kTe = 0.1  # keV
        kTbb = 1e-3  # keV

        def f_gamma(g):
            return jnp.sum(nthcomp_lnu_interp(nu, g, kTe, kTbb, _template=nthcomp_table))

        def f_kte(t):
            return jnp.sum(nthcomp_lnu_interp(nu, gamma, t, kTbb, _template=nthcomp_table))

        # AD gradients
        grad_ad_gamma = jax.grad(f_gamma)(gamma)
        grad_ad_kte = jax.grad(f_kte)(kTe)

        # Central FD with two step sizes to verify truncation balance
        eps1 = 1e-5
        eps2 = 1e-6

        grad_fd_gamma_1 = (f_gamma(gamma + eps1) - f_gamma(gamma - eps1)) / (2 * eps1)
        grad_fd_gamma_2 = (f_gamma(gamma + eps2) - f_gamma(gamma - eps2)) / (2 * eps2)

        grad_fd_kte_1 = (f_kte(kTe + eps1) - f_kte(kTe - eps1)) / (2 * eps1)
        grad_fd_kte_2 = (f_kte(kTe + eps2) - f_kte(kTe - eps2)) / (2 * eps2)

        # Two FD steps should agree to show we're in truncation-balanced regime
        assert jnp.allclose(grad_fd_gamma_1, grad_fd_gamma_2, rtol=1e-4)
        assert jnp.allclose(grad_fd_kte_1, grad_fd_kte_2, rtol=1e-4)

        # AD should agree with FD to 1e-5 relative
        assert jnp.allclose(grad_ad_gamma, grad_fd_gamma_2, rtol=1e-5), (
            f"gamma gradient: AD={grad_ad_gamma:.6e}, FD={grad_fd_gamma_2:.6e}"
        )
        assert jnp.allclose(grad_ad_kte, grad_fd_kte_2, rtol=1e-5), (
            f"kTe gradient: AD={grad_ad_kte:.6e}, FD={grad_fd_kte_2:.6e}"
        )

    @pytest.mark.regression_bug
    def test_float32_mode_support(self, nthcomp_table):
        """Float32 mode: output is float32, finite, and within 1e-5 relative of float64.

        Tests that the fix preserves pure-float32 support (JAX_ENABLE_X64=0).
        In float32 mode, coordinates are kept in float32, and output must stay
        float32 and remain finite despite lower precision.
        """
        nu = jnp.logspace(10, 20, 40)
        gamma = 2.0
        kTe = 0.1
        kTbb = 1e-3

        # Float64 reference (current default)
        ref_result = nthcomp_lnu_interp(nu, gamma, kTe, kTbb, _template=nthcomp_table)
        assert ref_result.dtype in (jnp.float32, jnp.float64), "Reference should be float32/64"

        # Float32 mode (simulate by disabling x64)
        with jax.enable_x64(False):
            nu_f32 = nu.astype(jnp.float32)
            gamma_f32 = jnp.array(gamma, dtype=jnp.float32)
            kte_f32 = jnp.array(kTe, dtype=jnp.float32)
            ktbb_f32 = jnp.array(kTbb, dtype=jnp.float32)

            f32_result = nthcomp_lnu_interp(
                nu_f32, gamma_f32, kte_f32, ktbb_f32, _template=nthcomp_table
            )

            # Must be float32
            assert f32_result.dtype == jnp.float32, f"Expected float32, got {f32_result.dtype}"

            # Must be finite
            assert jnp.all(jnp.isfinite(f32_result)), "Float32 result contains inf or nan"

            # Within 1e-5 relative of float64 (accounting for precision loss)
            # Compare on nodes where both are > 1e-6 of respective peaks
            peak_ref = jnp.max(ref_result)
            peak_f32 = jnp.max(f32_result)
            mask = (ref_result > 1e-6 * peak_ref) & (f32_result > 1e-6 * peak_f32)

            if jnp.any(mask):
                # Relative diff, but account for float32 precision floor
                rel_diff = jnp.abs(f32_result[mask] - ref_result[mask]) / (
                    jnp.abs(ref_result[mask]) + 1e-30
                )
                max_rel_diff = jnp.max(rel_diff)
                assert max_rel_diff < 1e-5, (
                    f"Float32 mode result diverges by {max_rel_diff:.2e} from float64"
                )
