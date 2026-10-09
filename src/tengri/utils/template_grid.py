# SPDX-License-Identifier: BSD-3-Clause
"""Frequency integral of a tabulated template on its own wavelength grid.

The normalization that every tabulated SED component divides by: AGN torus
libraries (``components/agn/_template_grid.py``) and dust emission templates
(``components/dust/emission_templates.py``) share it, so it lives here, in a
neutral module that neither component imports from the other.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from tengri.utils.grid_interp import loglog_integral
from tengri.utils.physics_constants import C_AA as _C_AA

__all__ = ["native_bolometric_nu", "native_bolometric_nu_np", "native_nu_integral"]


def native_bolometric_nu(
    lnu_native: jnp.ndarray, wave_native: jnp.ndarray, *, floor: float = 1e-100
) -> jnp.ndarray:
    r"""Frequency integral of a template on its OWN wavelength grid.

    The normalization every tabulated component divides by. Taken on the
    template's native grid, before resampling, it is a property of the template
    and the model coordinates alone: the caller's wavelength sampling and range
    never enter. (A trapezoid over the caller's grid would clip the template
    wherever the grid stops short of its support and would shift with the node
    density.)

    The integral is that of the interpolant
    :func:`~tengri.utils.grid_interp.resample_template` builds (a power law
    between adjacent nodes in :math:`\lambda`), so the resampled template
    integrates to the same value on any grid fine enough to resolve it. A
    trapezoid in :math:`\nu` over the native nodes would instead lay a chord
    across a convex segment and differ by up to ~1e-2 on the coarse (R ~ 7)
    libraries.

    Parameters
    ----------
    lnu_native : array_like, shape (n_native,)
        Template :math:`L_\nu` on ``wave_native`` [arbitrary units per Hz;
        shape only].
    wave_native : array_like, shape (n_native,)
        Template wavelength grid [Angstrom], ascending.
    floor : float, optional
        Lower bound on the returned magnitude, so the division cannot hit zero.

    Returns
    -------
    ndarray, shape ()
        :math:`\max(|\int L_\nu\,d\nu|, \text{floor})` in the template's units.

    Notes
    -----
    .. math::

        \int L_\nu\,\mathrm{d}\nu
        = \int L_\nu(\lambda)\,\frac{c}{\lambda^2}\,\mathrm{d}\lambda,

    where the integrand is a power law in :math:`\lambda` on every segment
    (:math:`L_\nu\propto\lambda^{s}` gives :math:`\lambda^{s-2}`), integrated
    in closed form by :func:`~tengri.utils.grid_interp.loglog_integral`.

    **JIT-compatible**: yes. **Gradient-safe**: yes. The resampling kernels used
    after this are homogeneous of degree one in the template values (log-flux
    interpolation is not linear, but scaling the template scales the result by the
    same factor), so dividing the resampled template by this integral equals
    resampling the normalized template.
    """
    wave = jnp.asarray(wave_native)
    integrand = jnp.asarray(lnu_native) * (_C_AA / wave**2)
    return jnp.maximum(jnp.abs(loglog_integral(wave, integrand)), floor)


def native_bolometric_nu_np(
    lnu_native: np.ndarray, wave_native: np.ndarray, *, floor: float = 1e-100
) -> float:
    r"""NumPy twin of :func:`native_bolometric_nu`, for build-time precompute tables.

    Same integral (a power law in wavelength between nodes, closed form), in
    float64 whatever the JAX precision mode, so a precomputed table and the
    runtime path normalize a template identically.

    Parameters
    ----------
    lnu_native : array_like, shape (n_native,)
        Template :math:`L_\nu` on ``wave_native`` [arbitrary units per Hz].
    wave_native : array_like, shape (n_native,)
        Template wavelength grid [Angstrom], ascending.
    floor : float, optional
        Lower bound on the returned magnitude.

    Returns
    -------
    float
        :math:`\max(|\int L_\nu\,d\nu|, \text{floor})`.
    """
    x = np.asarray(wave_native, dtype=np.float64)
    y = np.asarray(lnu_native, dtype=np.float64) * (_C_AA / x**2)
    x0, x1, y0, y1 = x[:-1], x[1:], y[:-1], y[1:]
    positive = (y0 > 0.0) & (y1 > 0.0)
    y0_safe = np.where(positive, y0, 1.0)
    y1_safe = np.where(positive, y1, 1.0)
    a = np.log(x1 / x0)
    u = a + np.log(y1_safe) - np.log(y0_safe)
    small = np.abs(u) < 1e-7
    ratio = np.where(small, 1.0 + 0.5 * u, np.expm1(u) / np.where(small, 1.0, u))
    power_law = x0 * y0_safe * a * ratio
    chord = 0.5 * (y0 + y1) * (x1 - x0)
    return max(abs(float(np.sum(np.where(positive, power_law, chord)))), floor)


def native_nu_integral(lnu_native, wave_native):
    r"""Frequency integral :math:`\int L_\nu\,d\nu` of a template on its own grid.

    Every tabulated dust emission closure divides the resampled template by this
    value, so the emitted power is the template's absorbed power ``L_absorbed``
    whatever wavelength grid the caller supplies. Integrating the resampled
    spectrum on the caller's grid instead would tie the normalization to that
    grid's density and extent.

    Parameters
    ----------
    lnu_native : array_like, shape (n_native,)
        Template :math:`L_\nu` on ``wave_native`` [any scale].
    wave_native : array_like, shape (n_native,)
        Template wavelength grid [Angstrom], ascending.

    Returns
    -------
    ndarray, shape ()
        :math:`\int L_\nu\,d\nu` in the units of ``lnu_native``.

    Notes
    -----
    The template is divided by its (stop-gradient) peak before the integral,
    so float32 cannot overflow; the factor is restored afterwards, which is
    algebraically exact. **JIT-compatible**: yes. **Gradient-safe**: yes.
    """
    lnu = jnp.asarray(lnu_native)
    peak = jax.lax.stop_gradient(jnp.max(jnp.abs(lnu)))
    peak = jnp.where(peak > 0.0, peak, 1.0)
    return native_bolometric_nu(lnu / peak, wave_native, floor=1e-30) * peak
