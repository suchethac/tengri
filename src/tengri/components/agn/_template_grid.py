# SPDX-License-Identifier: BSD-3-Clause
"""Shared carrier and evaluator for tabulated AGN torus template libraries.

Several torus families (CAT3D-Wind, Nenkova/CLUMPY, SKIRTOR-AGNfitter) are the
same computation over different data: PCHIP-interpolate an N-D template grid at
the model's coordinates, resample onto the requested wavelengths, then
renormalize by the frequency integral and scale to
``L_bol × f_torus``. This module holds that computation once.

The reason the grid is a :class:`TorusTemplateGrid` **argument** rather than a
closed-over array is threading. A closure's captured arrays are concrete at
trace time, so JAX freezes them into the graph as ``Constant`` ops, the whole
library, inlined, every time. A pytree passed as an argument becomes a
``Parameter`` instead. See ``tengri.components.agn.blocks._protocol.collect_block_templates``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from tengri.components.agn._phys import (
    bolometric_integral_nu as _bolometric_integral_nu,
    wavelength_to_nu as _wavelength_to_nu,
)
from tengri.utils.grid_interp import interp_nd_pchip, resample_template
from tengri.utils.physics_constants import L_SUN as _LSUN_ERG

__all__ = [
    "TorusTemplateGrid",
    "analytic_bolometric_nu",
    "native_bolometric_nu",
    "scale_to_lbol_native",
    "torus_lnu_from_grid",
]


class TorusTemplateGrid(NamedTuple):
    """A tabulated torus library, as a JAX pytree.

    Attributes
    ----------
    template : ndarray, shape (n_ax1, ..., n_axk, n_wave)
        Tabulated SEDs [arbitrary units: shape only; renormalized on use].
    axes : tuple of ndarray
        One 1-D coordinate array per leading template axis, ascending.
    wave_grid : ndarray, shape (n_wave,)
        Template rest-frame wavelength grid [Angstrom].

    Notes
    -----
    Leaves are normally ``np.ndarray`` when loaded from disk and JAX tracers
    when threaded through ``jax.jit``; both work, because every use site
    calls ``jnp.asarray`` (identity on a tracer).
    """

    template: jnp.ndarray
    axes: tuple[jnp.ndarray, ...]
    wave_grid: jnp.ndarray


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

    Parameters
    ----------
    lnu_native : array_like, shape (n_native,)
        Template on ``wave_native`` [arbitrary units per Hz; shape only].
    wave_native : array_like, shape (n_native,)
        Template wavelength grid [Angstrom].
    floor : float, optional
        Lower bound on the returned magnitude, so the division cannot hit zero.

    Returns
    -------
    ndarray, shape ()
        :math:`\max(|\int L_\nu\,d\nu|, \text{floor})` in the template's units.

    Notes
    -----
    **JIT-compatible**: yes. **Gradient-safe**: yes. The resampling kernels used
    after this are linear in the template values, so dividing the resampled
    template by this integral equals resampling the normalized template.
    """
    return _bolometric_integral_nu(
        jnp.asarray(lnu_native), _wavelength_to_nu(jnp.asarray(wave_native)), floor=floor
    )


def scale_to_lbol_native(
    template_native: jnp.ndarray,
    wave_native: jnp.ndarray,
    sed: jnp.ndarray,
    l_scale: float,
) -> jnp.ndarray:
    r"""Scale a resampled shape-only template to ``l_scale`` of bolometric power.

    .. math::

        L_\nu(\lambda) = L_{\rm scale}\,
            \frac{T(\lambda)}{\int T(\nu)\,\mathrm{d}\nu},

    with the integral taken over the template's NATIVE grid (``wave_native``),
    never over the caller's wavelength array, so the result does not depend on
    how the caller samples or truncates wavelength.

    Parameters
    ----------
    template_native : array_like, shape (n_native,)
        Interpolated template on ``wave_native`` [shape only; any scale].
    wave_native : array_like, shape (n_native,)
        Template wavelength grid [Angstrom].
    sed : array_like, shape (n_wave,)
        The same template resampled onto the caller's grid [same units as
        ``template_native``].
    l_scale : float
        Bolometric power the full template integrates to [erg/s].

    Returns
    -------
    ndarray, shape (n_wave,)
        :math:`L_\nu` [erg/s/Hz].

    Notes
    -----
    **JIT-compatible**: yes. **Gradient-safe**: yes.

    Float32 (#1206): the template's bolometric integral can reach ~1e45 erg/s,
    which overflows float32 and would flush the disc to zero, and the
    ``1e-100`` floor is itself below the float32 minimum. In float32 the
    integrand is therefore divided by its (stop-gradient) peak before it is
    integrated and the factors regrouped as
    ``(l_scale / hat_int) * (sed / peak)``, algebraically identical.
    """
    template_native = jnp.asarray(template_native)
    if sed.dtype == jnp.float32:
        peak = jax.lax.stop_gradient(jnp.max(jnp.abs(template_native)))
        peak = jnp.where(peak > 0.0, peak, 1.0)
        hat_int = native_bolometric_nu(template_native / peak, wave_native, floor=1e-30)
        return (l_scale / hat_int) * (sed / peak)
    return l_scale * sed / native_bolometric_nu(template_native, wave_native)


def analytic_bolometric_nu(
    shape_fn: Callable[[jnp.ndarray], jnp.ndarray],
    wave_lo: float,
    wave_hi: float,
    n_nodes: int = 4097,
) -> jnp.ndarray:
    r"""Frequency integral of an analytic shape on a FIXED internal grid.

    For shapes with no tabulated native grid and no closed form (a Planck
    function times an opacity, a blend of sigmoids). The integration grid is
    ``n_nodes`` log-spaced wavelengths over ``[wave_lo, wave_hi]``, chosen by
    the caller to span the shape's support; the caller's own wavelength
    array never enters, so the normalization cannot depend on it.

    Parameters
    ----------
    shape_fn : callable
        Maps wavelength ``array_like, shape (n,)`` [Angstrom] to
        :math:`L_\nu` ``ndarray, shape (n,)`` [arbitrary units per Hz].
    wave_lo, wave_hi : float
        Wavelength span of the internal grid [Angstrom]; the shape must be
        negligible outside it.
    n_nodes : int, optional
        Grid nodes. The trapezoid error on a smooth shape is
        :math:`O((\Delta\ln\lambda)^2)`, ~1e-6 at the default.

    Returns
    -------
    ndarray, shape ()
        :math:`\int L_\nu\,d\nu` in the shape's units.

    Notes
    -----
    **JIT-compatible**: yes (the grid is a trace-time constant).
    **Gradient-safe**: yes.
    """
    wave = jnp.asarray(np.geomspace(wave_lo, wave_hi, n_nodes))
    return _bolometric_integral_nu(shape_fn(wave), _wavelength_to_nu(wave))


def torus_lnu_from_grid(
    grid: TorusTemplateGrid,
    wavelength: jnp.ndarray,
    coords: tuple,
    *,
    agn_log_lbol: float,
    agn_torus_frac: float,
) -> jnp.ndarray:
    r"""Interpolate a torus template library and scale it to ``L_bol``.

    Parameters
    ----------
    grid : TorusTemplateGrid
        Template arrays, passed in so they can thread through ``jax.jit``.
    wavelength : array_like, shape (n_wave,)
        Rest-frame output wavelength grid [Angstrom].
    coords : tuple of float
        Interpolation coordinate per leading axis of ``grid.template``,
        in the same order as ``grid.axes``.
    agn_log_lbol : float
        :math:`\log_{10}(L_{\rm bol}/L_\odot)`.
    agn_torus_frac : float
        Fraction of :math:`L_{\rm bol}` reprocessed by the torus [0, 1].

    Returns
    -------
    ndarray, shape (n_wave,)
        Spectral luminosity density [erg/s/Hz].

    Notes
    -----
    .. math::

        L_\nu(\lambda) = L_{\rm bol}\, f_{\rm torus}\,
                         \frac{T(\lambda;\, \mathbf{c})}
                              {\int T(\nu;\, \mathbf{c})\,\mathrm{d}\nu}

    where :math:`T` is the interpolated template, :math:`\mathbf{c}` the
    coordinates, :math:`L_{\rm bol} = 10^{\rm agn\_log\_lbol} L_\odot`, and
    the integral runs over the template's own native frequency grid (before
    resampling), so the result does not depend on how ``wavelength`` is
    sampled or where it starts and stops. The template carries shape only; its
    absolute scale is divided out.

    **JIT-compatible**: yes. **Gradient-safe**: yes, node-exact PCHIP is
    C¹-continuous across every axis.
    """
    template = interp_nd_pchip(
        jnp.asarray(grid.template),
        tuple(jnp.asarray(axis) for axis in grid.axes),
        coords,
    )
    wave_native = jnp.asarray(grid.wave_grid)
    integral_safe = native_bolometric_nu(template, wave_native)
    sed = resample_template(wavelength, wave_native, template, left=0.0, right=0.0)
    l_scale = 10.0**agn_log_lbol * _LSUN_ERG * agn_torus_frac
    return l_scale * sed / integral_safe
