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

from typing import NamedTuple

import jax.numpy as jnp

from tengri.components.agn._phys import (
    bolometric_integral_nu as _bolometric_integral_nu,
    wavelength_to_nu as _wavelength_to_nu,
)
from tengri.utils.grid_interp import interp_nd_pchip, resample_template
from tengri.utils.physics_constants import L_SUN as _LSUN_ERG

__all__ = ["TorusTemplateGrid", "torus_lnu_from_grid"]


class TorusTemplateGrid(NamedTuple):
    """A tabulated torus library, as a JAX pytree.

    Attributes
    ----------
    template: ndarray, shape (n_ax1, ..., n_axk, n_wave)
        Tabulated SEDs [arbitrary units: shape only; renormalized on use].
    axes: tuple of ndarray
        One 1-D coordinate array per leading template axis, ascending.
    wave_grid: ndarray, shape (n_wave,)
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
    grid: TorusTemplateGrid
        Template arrays, passed in so they can thread through ``jax.jit``.
    wavelength: array_like, shape (n_wave,)
        Rest-frame output wavelength grid [Angstrom].
    coords: tuple of float
        Interpolation coordinate per leading axis of ``grid.template``,
        in the same order as ``grid.axes``.
    agn_log_lbol: float
        :math:`\log_{10}(L_{\rm bol}/L_\odot)`.
    agn_torus_frac: float
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
    the integral runs over the frequency grid matching ``wavelength``.
    The template carries shape only; its absolute scale is divided out.

    **JIT-compatible**: yes. **Gradient-safe**: yes, node-exact PCHIP is
    C¹-continuous across every axis.
    """
    template = interp_nd_pchip(
        jnp.asarray(grid.template),
        tuple(jnp.asarray(axis) for axis in grid.axes),
        coords,
    )
    sed = resample_template(wavelength, jnp.asarray(grid.wave_grid), template, left=0.0, right=0.0)
    nu = _wavelength_to_nu(wavelength)
    integral_safe = _bolometric_integral_nu(sed, nu, floor=1e-100)
    l_scale = 10.0**agn_log_lbol * _LSUN_ERG * agn_torus_frac
    return l_scale * sed / integral_safe
