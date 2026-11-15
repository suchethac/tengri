# SPDX-License-Identifier: BSD-3-Clause
r"""The LSF width has a finite gradient where the kernel has zero width (#3546).

``apply_lsf`` builds its kernel width as

.. math::

    \sigma_{\rm eff} = \sqrt{\max(\sigma_{\rm inst}^2 - \sigma_{\rm lib}^2, 0)
                             + \sigma_v^2} .

Where the library is coarser than the instrument and ``sigma_v = 0`` the
argument is exactly zero. The derivative of the square root there is infinite
and the derivative of the clamp is zero, so reverse mode returned
``inf * 0 = NaN``. With a per-wavelength library curve the library width at a
pixel depends on redshift, so the NaN reached d(spectrum)/dz: a free-redshift
fit to a NIRSpec prism spectrum stopped at 0 MAP iterations.

The fixture is a prism resolution curve and a library curve that is coarser than
the prism below 3000 A rest, so some pixels are clamped and some are not. The
reference for the gradient is a central finite difference of the same function.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.observation.spectrum import (
    apply_lsf,
    nirspec_prism_resolution,
    resolve_sigma_lib_kms,
)

pytestmark = pytest.mark.regression_bug

_Z = 8.3
_WAVE_OBS = jnp.linspace(6000.0, 53000.0, 327)
_RESOLUTION = nirspec_prism_resolution(_WAVE_OBS / 1e4)
_WAVE_REST = jnp.linspace(500.0, 9000.0, 400)
_SIGMA_LIB_REST = 30.0 + 60.0 * _WAVE_REST / 9000.0 + 3000.0 * (_WAVE_REST < 3000.0)
_SPECTRUM = 1.0 + 0.3 * jnp.sin(_WAVE_OBS / 900.0)


def _loss(redshift, sigma_v_kms):
    sigma_lib = resolve_sigma_lib_kms(_WAVE_OBS, redshift, 70.0, (_WAVE_REST, _SIGMA_LIB_REST))
    smoothed = apply_lsf(
        _SPECTRUM,
        _WAVE_OBS,
        resolution=_RESOLUTION,
        sigma_lib_kms=sigma_lib,
        sigma_v_kms=sigma_v_kms,
    )
    return jnp.sum(smoothed**2)


def _clamped_pixels(redshift):
    sigma_lib = resolve_sigma_lib_kms(_WAVE_OBS, redshift, 70.0, (_WAVE_REST, _SIGMA_LIB_REST))
    sigma_inst = 299792.458 / (2.354820045030949 * _RESOLUTION)
    return int(jnp.sum(sigma_inst < sigma_lib))


def test_fixture_has_clamped_and_unclamped_pixels():
    n_clamped = _clamped_pixels(_Z)
    assert 0 < n_clamped < _WAVE_OBS.size


def test_redshift_gradient_is_finite_and_matches_finite_difference():
    grad = float(jax.grad(_loss, argnums=0)(_Z, 0.0))
    step = 1e-4
    reference = float((_loss(_Z + step, 0.0) - _loss(_Z - step, 0.0)) / (2.0 * step))
    assert np.isfinite(grad)
    # Measured: grad and the finite difference agree to 9e-9 relative at step 1e-4.
    np.testing.assert_allclose(grad, reference, rtol=1e-6)


def test_sigma_v_gradient_is_finite_at_zero():
    grad = float(jax.grad(_loss, argnums=1)(_Z, 0.0))
    assert np.isfinite(grad)


def test_zero_width_pixels_leave_the_value_unchanged():
    """Library coarser than the instrument everywhere and sigma_v = 0: identity."""
    out = apply_lsf(
        _SPECTRUM,
        _WAVE_OBS,
        resolution=_RESOLUTION,
        sigma_lib_kms=jnp.full(_WAVE_OBS.shape, 1.0e5),
        sigma_v_kms=0.0,
    )
    np.testing.assert_allclose(np.asarray(out), np.asarray(_SPECTRUM), rtol=1e-6)
    grad = jax.grad(
        lambda s: jnp.sum(
            apply_lsf(
                _SPECTRUM,
                _WAVE_OBS,
                resolution=_RESOLUTION,
                sigma_lib_kms=s * jnp.ones(_WAVE_OBS.shape),
                sigma_v_kms=0.0,
            )
            ** 2
        )
    )(1.0e5)
    assert float(grad) == 0.0
