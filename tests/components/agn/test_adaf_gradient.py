# SPDX-License-Identifier: BSD-3-Clause
"""Gradient tests for ADAF disc model."""

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.gradient


def fd_grad(f, x: float, eps: float = 1e-4) -> float:
    """Central finite-difference gradient. O(eps^2) accurate."""
    return float((f(x + eps) - f(x - eps)) / (2.0 * eps))


# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture()
def optical_wavelength():
    """Optical/UV wavelength grid."""
    return jnp.logspace(2.5, 5.0, 200)  # 316 A to 100,000 A


# ── Gradient tests ────────────────────────────────────────────────


class TestAdafJitGrad:
    """JIT compilation and gradient finite-difference checks."""

    def test_jit_compatible(self):
        """adaf_disc is JIT-compilable."""
        from tengri.components.agn.disc import adaf_disc

        wavelength = jnp.logspace(0, 8, 500)

        @jax.jit
        def _run(wave):
            return adaf_disc(wave, agn_log_lbol=42.0, agn_lum_ratio=0.1)

        result = _run(wavelength)
        chex.assert_tree_all_finite(result)

    def test_gradient_wrt_lbol(self, optical_wavelength):
        """∂(∑SED)/∂log_lbol for adaf_disc.

        Checks finite-difference gradient accuracy for bolometric luminosity.
        Expected sign: positive (higher luminosity → brighter SED).
        """
        from tengri.components.agn.disc import adaf_disc

        def loss_fn(lbol):
            return jnp.sum(adaf_disc(optical_wavelength, agn_log_lbol=lbol, agn_lum_ratio=0.1))

        g = float(jax.grad(loss_fn)(42.0))
        np.testing.assert_allclose(
            g,
            fd_grad(loss_fn, 42.0, eps=0.01),
            rtol=1e-3,
            err_msg="adaf_disc: FD check ∂/∂log_lbol",
        )

    def test_gradient_wrt_r_tr(self, optical_wavelength):
        """∂(∑SED)/∂r_tr for adaf_disc.

        Truncation radius gradient. Expected sign: negative (larger r_tr
        → cooler inner disc → less hot-spot flux, especially UV).
        """
        from tengri.components.agn.disc import adaf_disc

        def loss_fn(r_tr):
            return jnp.sum(
                adaf_disc(optical_wavelength, agn_log_lbol=42.0, agn_lum_ratio=0.1, agn_r_tr=r_tr)
            )

        g = float(jax.grad(loss_fn)(100.0))
        np.testing.assert_allclose(
            g,
            fd_grad(loss_fn, 100.0, eps=0.1),
            rtol=1e-3,
            err_msg="adaf_disc: FD check ∂/∂r_tr",
        )

    def test_gradient_wrt_delta(self, optical_wavelength):
        """∂(∑SED)/∂delta for adaf_disc.

        ADAF δ parameter (comptonization efficiency) gradient.
        """
        from tengri.components.agn.disc import adaf_disc

        def loss_fn(delta):
            return jnp.sum(
                adaf_disc(
                    optical_wavelength, agn_log_lbol=42.0, agn_lum_ratio=0.1, agn_adaf_delta=delta
                )
            )

        g = float(jax.grad(loss_fn)(0.01))
        np.testing.assert_allclose(
            g,
            fd_grad(loss_fn, 0.01, eps=1e-4),
            rtol=1e-3,
            err_msg="adaf_disc: FD check ∂/∂adaf_delta",
        )

    def test_gradient_wrt_beta(self, optical_wavelength):
        """∂(∑SED)/∂beta for adaf_disc.

        ADAF β parameter (electron temperature ratio) gradient.
        Fixed by the Mahadevan 1997 rewrite (2026-04-21): The new implementation uses
        a more physical weighting of synchrotron/bremsstrahlung/IC components via
        magnetic field pressure (1-beta), which avoids the algebraic singularity
        that plagued the old linear weighting scheme.
        """
        from tengri.components.agn.disc import adaf_disc

        def loss_fn(beta):
            return jnp.sum(
                adaf_disc(
                    optical_wavelength, agn_log_lbol=42.0, agn_lum_ratio=0.1, agn_adaf_beta=beta
                )
            )

        g = float(jax.grad(loss_fn)(0.5))
        np.testing.assert_allclose(
            g,
            fd_grad(loss_fn, 0.5, eps=1e-3),
            rtol=1e-3,
            err_msg="adaf_disc: FD check ∂/∂adaf_beta",
        )
