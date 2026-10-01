# SPDX-License-Identifier: BSD-3-Clause
"""Gradient tests for the free instrument-LSF scale ``lsf_scale`` (#2526).

``lsf_scale`` enters the kernel only through ``resolution / lsf_scale``
(``apply_lsf`` computes ``sigma_inst_kms = C_KM_S / (FWHM_TO_SIGMA *
resolution)``, so scaling the resolution by ``1/lsf_scale`` scales
``sigma_inst`` by ``lsf_scale``). Both the forward-model output and the
closed-form kernel width must carry a finite, correctly-signed gradient
with respect to ``lsf_scale``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from tengri.observation.spectrum import apply_lsf

pytestmark = pytest.mark.gradient

_C_KMS = 299792.458
_FWHM_TO_SIGMA = 2.354820045030949


def _sigma_eff_kms(lsf_scale, resolution, sigma_lib_kms, sigma_v_kms):
    """Mirrors apply_lsf's internal kernel-width formula exactly."""
    sigma_inst_kms = _C_KMS / (_FWHM_TO_SIGMA * resolution / lsf_scale)
    deficit = sigma_inst_kms**2 - sigma_lib_kms**2
    return jnp.sqrt(jnp.maximum(deficit, 0.0) + sigma_v_kms**2)


def test_kernel_width_gradient_matches_analytic_derivative_where_resolved():
    """d(sigma_eff)/d(lsf_scale) at a point where sigma_inst > sigma_lib (the
    kernel is "resolved": apply_lsf's ``jnp.maximum`` clamp is inactive, so
    the closed form is differentiable and its gradient is exactly
    ``sigma_inst(lsf_scale) * sigma_inst_1 / sigma_eff`` (chain rule through
    the quadrature sum), positive since sigma_eff grows with lsf_scale.
    """
    resolution = 3000.0
    sigma_lib_kms = 20.0
    sigma_v_kms = 100.0
    lsf_scale = 1.0

    grad_fn = jax.grad(_sigma_eff_kms)
    grad = float(grad_fn(lsf_scale, resolution, sigma_lib_kms, sigma_v_kms))

    sigma_inst_1 = _C_KMS / (_FWHM_TO_SIGMA * resolution)
    sigma_inst = lsf_scale * sigma_inst_1
    assert sigma_inst**2 > sigma_lib_kms**2, "test point must be in the resolved regime"
    sigma_eff = float(_sigma_eff_kms(lsf_scale, resolution, sigma_lib_kms, sigma_v_kms))
    analytic = sigma_inst * sigma_inst_1 / sigma_eff

    assert jnp.isfinite(grad)
    assert grad > 0.0  # sigma_eff grows monotonically with lsf_scale here
    assert grad == pytest.approx(analytic, rel=1e-6)

    eps = 1e-5
    fd = (
        _sigma_eff_kms(lsf_scale + eps, resolution, sigma_lib_kms, sigma_v_kms)
        - _sigma_eff_kms(lsf_scale - eps, resolution, sigma_lib_kms, sigma_v_kms)
    ) / (2 * eps)
    assert grad == pytest.approx(float(fd), rel=1e-4)


def test_predicted_flux_gradient_wrt_lsf_scale_is_finite_and_matches_finite_diff():
    """jax.grad through apply_lsf (resolution / lsf_scale) end-to-end is
    finite and matches a central finite difference, at the same resolved
    operating point as the closed-form test above.
    """
    wave = jnp.linspace(3600.0, 7300.0, 500)
    flux = jnp.ones_like(wave) - 0.3 * jnp.exp(-0.5 * ((wave - 5000.0) / 3.0) ** 2)
    resolution = 3000.0
    sigma_lib_kms = 20.0
    sigma_v_kms = 100.0

    def objective(lsf_scale):
        out = apply_lsf(
            flux,
            wave,
            resolution=resolution / lsf_scale,
            sigma_lib_kms=sigma_lib_kms,
            sigma_v_kms=sigma_v_kms,
        )
        return jnp.sum(out)

    grad = float(jax.grad(objective)(1.0))
    assert jnp.isfinite(grad), "non-finite gradient: lsf_scale broke differentiability"
    assert grad != 0.0, "zero gradient: lsf_scale would be silently inert (#2100 shape)"

    eps = 1e-4
    fd = (objective(1.0 + eps) - objective(1.0 - eps)) / (2 * eps)
    assert grad == pytest.approx(float(fd), rel=5e-3)
