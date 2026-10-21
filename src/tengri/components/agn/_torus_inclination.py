# SPDX-License-Identifier: BSD-3-Clause
r"""Angle-integrated power and energy ledger of a tabulated torus library.

A torus library whose records sit on one radiative-transfer scale across inclination keeps
the flux ratio between sightlines: a face-on CLUMPY torus is brighter in the infrared than
an edge-on one at the same central luminosity. Such a library is normalized ONCE per
parameter point, by the sphere mean of the power its records carry,

.. math::

    \bar P = \int_0^1 P(\mu)\,d\mu, \qquad
    P(\mu) = \int T_\nu(\mu)\,d\nu, \qquad \mu = \cos i,

so the angle-integrated torus power is :math:`f L_{\rm bol}` and every sightline keeps the
library's own flux relative to the others. By symmetry about the equator
:math:`\tfrac12\int_0^\pi P(\theta)\sin\theta\,d\theta = \int_0^1 P(\mu)\,d\mu`.

The quadrature is Gauss-Legendre of order ``INCLINATION_GL_ORDER`` on each interval between
the library's inclination nodes, in the variable the library is interpolated in (each panel
is one cubic of the interpolant), weighted by :math:`|d\mu/dx|`. The panels run over the
whole of :math:`\mu \in [0, 1]`; beyond an end node the row is whatever the lookup returns
there (the end record, or the continued end cubic), which is what the component emits along
such a sightline.

A torus-only component has no disc of its own: the disc share :math:`(1 - f) L_{\rm bol}` is
published as ``log_L_agn_disc_unemitted`` so the ledger closes on the accretion power,
``10**agn_log_lbol L_sun`` (NAMING_CONTRACT section 4c, "Template tori").
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from tengri.components.agn._publication import INCLINATION_GL_ORDER, with_lsun_siblings
from tengri.components.agn._template_grid import (
    TorusTemplateGrid,
    float32_safe_template,
    native_bolometric_nu,
)
from tengri.utils.grid_interp import interp_nd_pchip, resample_template
from tengri.utils.physics_constants import L_SUN as _LSUN_ERG
from tengri.utils.scale import log10_magnitude

__all__ = [
    "TORUS_LEDGER_KEYS",
    "TorusInclination",
    "absent_torus_ledger",
    "inclination_quadrature",
    "sphere_mean_power_at_nodes",
    "torus_ledger",
    "torus_ledger_from_grid",
    "torus_lnu_sphere_normalized",
    "torus_log_power",
    "torus_rows_by_inclination",
]

#: ``log10(L_sun / (erg/s))`` [dex].
_LOG10_L_SUN: float = float(np.log10(_LSUN_ERG))

#: Interval of the inclination variable that spans ``cos i`` in ``[0, 1]``, ascending.
_X_RANGE: dict[str, tuple[float, float]] = {
    "cos_inc": (0.0, 1.0),
    "incl_deg": (0.0, 90.0),
    "elevation_deg": (0.0, 90.0),
}


class TorusInclination(NamedTuple):
    """Where a library keeps its inclination, and in which variable.

    Attributes
    ----------
    axis : int
        Index of the inclination axis in ``grid.axes``.
    variable : {'cos_inc', 'incl_deg', 'elevation_deg'}
        The axis coordinate: :math:`\\cos i` [dimensionless], the inclination :math:`i` from
        the polar axis [deg], or the elevation :math:`\\psi = 90^\\circ - i` above the
        equatorial plane [deg].
    extrapolate : bool
        Whether the lookup continues the end cubic past the end nodes (else it clamps).
    """

    axis: int
    variable: str
    extrapolate: bool = False


def _abs_dmu_dx(x, variable: str):
    """``|d cos i / dx|`` of the inclination variable ``x``."""
    if variable == "cos_inc":
        return jnp.ones_like(x)
    rad = jnp.pi / 180.0
    if variable == "incl_deg":
        return jnp.sin(x * rad) * rad
    if variable == "elevation_deg":
        return jnp.cos(x * rad) * rad
    raise ValueError(f"unknown inclination variable {variable!r}; expected one of {_X_RANGE}")


def inclination_quadrature(x_nodes, variable: str, order: int = INCLINATION_GL_ORDER):
    r"""Points and weights for :math:`\int_0^1 f\,d\mu` in a library's inclination variable.

    Parameters
    ----------
    x_nodes : array_like, shape (n_inc,)
        The library's inclination nodes, ascending, in ``variable``.
    variable : {'cos_inc', 'incl_deg', 'elevation_deg'}
        See :class:`TorusInclination`.
    order : int, optional
        Gauss-Legendre points per interval.

    Returns
    -------
    x : ndarray, shape ((n_inc + 1) * order,)
        Quadrature points in ``variable``.
    weight : ndarray, shape ((n_inc + 1) * order,)
        Weights [dimensionless], summing to 1.

    Notes
    -----
    **JIT/vmap-safe**: yes; the panel count is static, so the nodes may be a tracer. The
    panel edges are the nodes clipped into the range of ``variable`` plus its two ends; a
    node at an end gives an empty panel of zero weight. The weights are normalized to sum
    to 1, so the mean of a constant is that constant to rounding.
    """
    lo, hi = _X_RANGE[variable]
    nodes = jnp.asarray(x_nodes)
    ends = jnp.asarray([lo, hi], dtype=nodes.dtype)
    edges = jnp.concatenate([ends[:1], jnp.clip(nodes, lo, hi), ends[1:]])
    gx, gw = np.polynomial.legendre.leggauss(order)
    a, b = edges[:-1, None], edges[1:, None]
    x = (0.5 * (b - a) * gx + 0.5 * (b + a)).ravel()
    weight = (0.5 * (b - a) * gw).ravel() * _abs_dmu_dx(x, variable)
    return x, weight / jnp.sum(weight)


def torus_rows_by_inclination(
    grid: TorusTemplateGrid, coords: tuple, inc: TorusInclination, dtype
):
    """Template rows at every inclination node, the other axes interpolated at ``coords``.

    Parameters
    ----------
    grid : TorusTemplateGrid
        The library.
    coords : tuple of float
        One coordinate per axis of ``grid.axes``; the inclination entry is not read.
    inc : TorusInclination
        The library's inclination axis.
    dtype : dtype
        Evaluation dtype (the caller's wavelength dtype), for the float32-safe table.

    Returns
    -------
    rows : ndarray, shape (n_inc, n_native)
        Interpolated template rows [library units: shape across inclination is kept].
    x_nodes : ndarray, shape (n_inc,)
        The inclination nodes, in ``inc.variable``.
    """
    template = jnp.asarray(float32_safe_template(grid.template, dtype))
    n_axes = len(grid.axes)
    template = jnp.moveaxis(template, inc.axis, n_axes - 1)
    other = tuple(i for i in range(n_axes) if i != inc.axis)
    x_nodes = jnp.asarray(grid.axes[inc.axis])
    if not other:
        return template, x_nodes
    rows = interp_nd_pchip(
        template,
        tuple(jnp.asarray(grid.axes[i]) for i in other),
        tuple(coords[i] for i in other),
    )
    return rows, x_nodes


def _row_at(rows, x_nodes, x, extrapolate: bool):
    """PCHIP in the inclination variable of an ``(n_inc, n_native)`` table."""
    return interp_nd_pchip(rows, (x_nodes,), (x,), extrapolate=(extrapolate,))


def _sphere_mean_power(rows, x_nodes, wave_native, inc: TorusInclination):
    """Sphere-mean frequency integral of the rows, on the native grid [library units]."""
    x_q, w_q = inclination_quadrature(x_nodes, inc.variable)

    def _power(x):
        return native_bolometric_nu(_row_at(rows, x_nodes, x, inc.extrapolate), wave_native)

    return jnp.sum(w_q * jax.vmap(_power)(x_q))


def sphere_mean_power_at_nodes(grid: TorusTemplateGrid, inc: TorusInclination) -> np.ndarray:
    """Sphere-mean power of the library at every node of its other axes, for build-time tables.

    Parameters
    ----------
    grid : TorusTemplateGrid
        The library (host arrays).
    inc : TorusInclination
        The library's inclination axis.

    Returns
    -------
    ndarray
        :math:`\bar P` [library units] at every node of the other axes, the denominator
        :func:`torus_lnu_sphere_normalized` divides by there, in float64; shape
        ``grid.template.shape[:-1]`` with the inclination axis of length 1, so it
        broadcasts against the template's leading axes.

    Notes
    -----
    **JIT-compatible**: no; a host loop over the nodes, evaluated with the runtime lookup
    so a precomputed table and the runtime path normalize identically.
    """
    other = [i for i in range(len(grid.axes)) if i != inc.axis]
    shape = tuple(len(grid.axes[i]) for i in other)
    out = np.empty(shape, dtype=np.float64)
    wave_native = jnp.asarray(grid.wave_grid, dtype=jnp.float64)
    for idx in np.ndindex(*shape):
        coords = [0.0] * len(grid.axes)
        for i, j in zip(other, idx, strict=True):
            coords[i] = float(np.asarray(grid.axes[i])[j])
        rows, x_nodes = torus_rows_by_inclination(grid, tuple(coords), inc, jnp.float64)
        out[idx] = float(_sphere_mean_power(rows, x_nodes, wave_native, inc))
    return np.expand_dims(out, inc.axis)


def torus_lnu_sphere_normalized(
    grid: TorusTemplateGrid,
    wavelength: jnp.ndarray,
    coords: tuple,
    inc: TorusInclination,
    *,
    agn_log_lbol: float,
    agn_torus_frac: float,
) -> jnp.ndarray:
    r"""Torus :math:`L_\nu` along one sightline, the library scaled once by its sphere mean.

    Parameters
    ----------
    grid : TorusTemplateGrid
        Library whose records share one scale across inclination (:math:`T_\nu`, any units).
    wavelength : array_like, shape (n_wave,)
        Rest-frame output wavelength grid [Angstrom].
    coords : tuple of float
        Interpolation coordinate per axis of ``grid.axes``; the inclination entry is the
        sightline.
    inc : TorusInclination
        The library's inclination axis.
    agn_log_lbol : float
        :math:`\log_{10}(L_{\rm bol}/L_\odot)`, the accretion power [dex].
    agn_torus_frac : float
        Fraction of :math:`L_{\rm bol}` reprocessed by the torus [dimensionless].

    Returns
    -------
    ndarray, shape (n_wave,)
        Spectral luminosity density [erg/s/Hz].

    Notes
    -----
    .. math::

        L_\nu(\lambda; \mu) = f\,L_{\rm bol}\,\frac{T_\nu(\lambda; \mu)}{\bar P},
        \qquad \bar P = \int_0^1 \!\! \int T_\nu(\mu)\,d\nu\,d\mu,

    with :math:`L_{\rm bol} = 10^{\rm agn\_log\_lbol} L_\odot`, :math:`f` = ``agn_torus_frac``,
    :math:`T_\nu` the library interpolated node-exactly (PCHIP, Fritsch & Carlson [1]_) at the
    parameter point, and the frequency integral that of the interpolant on the library's own
    wavelength nodes (``native_bolometric_nu``). The angle-integrated power is
    :math:`f L_{\rm bol}`; the power along :math:`\mu` is :math:`f L_{\rm bol} P(\mu)/\bar P`.

    **JIT/grad/vmap-safe**: yes.

    References
    ----------
    .. [1] F. N. Fritsch & R. E. Carlson, "Monotone Piecewise Cubic Interpolation,"
       SIAM J. Numer. Anal. 17, 238 (1980). https://doi.org/10.1137/0717021
    """
    rows, x_nodes = torus_rows_by_inclination(grid, coords, inc, wavelength.dtype)
    wave_native = jnp.asarray(grid.wave_grid)
    row = _row_at(rows, x_nodes, coords[inc.axis], inc.extrapolate)
    sed = resample_template(wavelength, wave_native, row, left=0.0, right=0.0)
    mean_power = _sphere_mean_power(rows, x_nodes, wave_native, inc)
    l_scale = 10.0**agn_log_lbol * _LSUN_ERG * agn_torus_frac
    return sed * (l_scale / mean_power)


def _log10_anisotropy(rows, x_nodes, wave_native, x_los, inc: TorusInclination):
    """``log10(P(mu) / P_mean)`` of a row table along ``x_los`` [dex]; a ratio, float32 safe."""
    los = native_bolometric_nu(_row_at(rows, x_nodes, x_los, inc.extrapolate), wave_native)
    return jnp.log10(los / _sphere_mean_power(rows, x_nodes, wave_native, inc))


def torus_log_power(agn_log_lbol, agn_torus_frac):
    r"""``log10`` of the angle-integrated torus power :math:`f L_{\rm bol}` [dex re erg/s].

    Parameters
    ----------
    agn_log_lbol : float
        :math:`\log_{10}(L_{\rm bol}/L_\odot)`, the accretion power [dex].
    agn_torus_frac : float
        Fraction of :math:`L_{\rm bol}` reprocessed by the torus [dimensionless].

    Returns
    -------
    ndarray, shape ()
        :math:`\log_{10}(f L_{\rm bol} / {\rm erg\,s^{-1}})` [dex]; ``-inf`` at ``f = 0``.

    Notes
    -----
    **JIT/grad/vmap-safe**: yes; formed in log space (NAMING_CONTRACT section 4c).
    """
    return jnp.asarray(agn_log_lbol) + _LOG10_L_SUN + log10_magnitude(jnp.asarray(agn_torus_frac))


#: The three powers a torus-only component publishes [dex re erg/s] (NAMING_CONTRACT 4c).
TORUS_LEDGER_KEYS: tuple[str, ...] = (
    "log_L_agn_torus",
    "log_L_agn_torus_los",
    "log_L_agn_disc_unemitted",
)


def absent_torus_ledger() -> dict:
    """The ledger of an absent torus (no templates loaded): every key ``-inf``, siblings 0.

    Returns
    -------
    dict
        ``{key: -inf}`` for :data:`TORUS_LEDGER_KEYS`, with their ``L_agn_*_lsun`` siblings.
    """
    return with_lsun_siblings({k: jnp.array(-jnp.inf) for k in TORUS_LEDGER_KEYS})


def torus_ledger(agn_log_lbol, agn_torus_frac, log_los) -> dict:
    r"""Published powers of a torus-only component: sphere mean, sightline, unemitted disc.

    Parameters
    ----------
    agn_log_lbol : float
        :math:`\log_{10}(L_{\rm bol}/L_\odot)`, the accretion power [dex].
    agn_torus_frac : float
        Fraction of :math:`L_{\rm bol}` reprocessed by the torus [dimensionless].
    log_los : float
        ``log10`` of the power along the sightline, the frequency integral of the emitted
        torus SED [dex re erg/s].

    Returns
    -------
    dict
        ``log_L_agn_torus`` (:math:`f L_{\rm bol}`, angle-integrated),
        ``log_L_agn_torus_los`` (``log_los``) and ``log_L_agn_disc_unemitted``
        (:math:`(1-f) L_{\rm bol}`, the disc share a torus-only component does not emit), all
        [dex re erg/s], plus their ``L_agn_*_lsun`` [Lsun] siblings.

    Notes
    -----
    .. math::

        L_{\rm torus} + L_{\rm disc,\,unemitted} = f L_{\rm bol} + (1-f) L_{\rm bol}
        = 10^{\rm agn\_log\_lbol} L_\odot.

    **JIT/grad/vmap-safe**: yes. Formed in log space, so finite in pure float32
    (NAMING_CONTRACT section 4c); ``f = 0`` gives ``-inf`` for the torus keys and ``f = 1``
    for the unemitted disc (an absent piece).
    """
    log_lbol = jnp.asarray(agn_log_lbol) + _LOG10_L_SUN
    return with_lsun_siblings(
        {
            "log_L_agn_torus": torus_log_power(agn_log_lbol, agn_torus_frac),
            "log_L_agn_torus_los": jnp.asarray(log_los),
            "log_L_agn_disc_unemitted": log_lbol
            + log10_magnitude(1.0 - jnp.asarray(agn_torus_frac)),
        }
    )


def torus_ledger_from_grid(
    grid: TorusTemplateGrid,
    coords: tuple,
    inc: TorusInclination,
    *,
    agn_log_lbol,
    agn_torus_frac,
    dtype,
) -> dict:
    r"""The :func:`torus_ledger` of :func:`torus_lnu_sphere_normalized` at one point.

    Parameters
    ----------
    grid : TorusTemplateGrid
        Library whose records share one scale across inclination.
    coords : tuple of float
        Interpolation coordinate per axis; the inclination entry is the sightline.
    inc : TorusInclination
        The library's inclination axis.
    agn_log_lbol : float
        :math:`\log_{10}(L_{\rm bol}/L_\odot)` [dex].
    agn_torus_frac : float
        Fraction of :math:`L_{\rm bol}` reprocessed by the torus [dimensionless].
    dtype : dtype
        Evaluation dtype (the caller's wavelength dtype).

    Returns
    -------
    dict
        As :func:`torus_ledger`; the sightline power is
        :math:`f L_{\rm bol} P(\mu)/\bar P`, the integral of the SED that
        :func:`torus_lnu_sphere_normalized` emits along ``coords``.

    Notes
    -----
    **JIT/grad/vmap-safe**: yes, and float32 safe: only the ratio
    :math:`P(\mu)/\bar P` is formed from the table.
    """
    rows, x_nodes = torus_rows_by_inclination(grid, coords, inc, dtype)
    log_aniso = _log10_anisotropy(
        rows, x_nodes, jnp.asarray(grid.wave_grid), coords[inc.axis], inc
    )
    log_los = torus_log_power(agn_log_lbol, agn_torus_frac) + log_aniso
    return torus_ledger(agn_log_lbol, agn_torus_frac, log_los)
