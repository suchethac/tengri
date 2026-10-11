# SPDX-License-Identifier: BSD-3-Clause
"""Float32 derivative of the ADAF spectrum with respect to ``T_e`` keeps the Compton ceiling.

The spectrum factor ``exp(-nu / nu_max_c)`` (Mahadevan 1997, Eq. 34 and the Comptonization
ceiling ``3 k T_e / h``) has a reverse-mode partial with respect to the linear ``nu_max_c`` of
order ``w nu / nu_max_c**2`` ~ 1e-39 per spectral point, a float32 subnormal that the backend
flushes to zero. The ceiling's contribution to ``d(sum L_nu) / d ln T_e`` is then lost, and
because the total derivative is a near-cancellation (net 1.4% of the sum of the terms'
magnitudes) the float32 gradient was off by ~1%. The ceiling reads ``AdafState.log_nu_max_c``
so the cotangent stays O(w).

The probe tilts the solved state by ``T_e -> T_e e^s`` with the frequency breaks following
their ``T_e`` scaling (``nu_p``, ``nu_min`` as ``T_e^2``, ``nu_max_c`` as ``T_e``), and
differentiates the normalized spectrum summed over a wavelength grid at ``s = 0``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.agn import adaf as adaf_mod

pytestmark = pytest.mark.gradient

_WAVE_AA = np.logspace(-3, 9, 1500)
_POINT = (12.0, 8.0, 0.3, 0.5, 0.1)  # (shape, log_mbh, alpha, beta, delta)
_RTOL = 1e-3


def _tilted_sum_gradient(dtype):
    """``d/ds`` at ``s = 0`` of ``sum_i L_nu(nu_i)`` with ``T_e`` tilted by ``e^s``."""
    shape, log_mbh, alpha, beta, delta = _POINT
    wave = jnp.asarray(_WAVE_AA, dtype=dtype)
    s0 = adaf_mod.adaf_scalar_state(2.0, 1e-3, log_mbh, alpha, beta, delta, shape, dtype=dtype)
    s0 = jax.tree_util.tree_map(lambda x: jnp.asarray(x, dtype), s0)
    nu = adaf_mod._wavelength_to_nu(wave)

    def loss(s):
        e = jnp.exp(s)
        tilted = s0._replace(
            t_e=s0.t_e * e,
            nu_p=s0.nu_p * e**2,
            nu_min=s0.nu_min * e**2,
            nu_max_c=s0.nu_max_c * e,
            log_nu_max_c=s0.log_nu_max_c + s,
        )
        return jnp.sum(jnp.exp(adaf_mod._adaf_log_total(nu, tilted) - s0.log_integral))

    return float(jax.grad(loss)(jnp.asarray(0.0, dtype)))


def test_float32_te_gradient_of_summed_spectrum_matches_float64():
    """The float32 ``T_e`` derivative of the summed normalized spectrum matches float64."""
    g64 = _tilted_sum_gradient(jnp.float64)
    with jax.enable_x64(False):
        g32 = _tilted_sum_gradient(jnp.float32)
    assert np.isfinite(g64) and g64 != 0.0
    assert abs(g32 - g64) <= _RTOL * abs(g64), f"f32 {g32} vs f64 {g64}"
