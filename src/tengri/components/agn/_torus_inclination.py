# SPDX-License-Identifier: BSD-3-Clause
r"""Angle-integrated power and energy ledger of a tabulated torus library.

A torus library whose records sit on one radiative-transfer scale per model keeps the flux
ratio between sightlines: a face-on CLUMPY torus is brighter in the infrared than an edge-on
one at the same central luminosity. Such a library is emitted as two factors. The SHAPE is
the library normalized per sightline (each record to unit power, interpolated as before,
``_template_grid.torus_lnu_from_grid``). The POWER along a sightline is a scalar table, the
records' own powers over their model's mean, interpolated with the same node-exact PCHIP:

.. math::

    L_\nu(\mu) = f L_{\rm bol}\,\hat T_\nu(\mu)\,\frac{a(\mu)}{\bar a}, \qquad
    \bar a = \int_0^1 a(\mu)\,d\mu, \qquad \mu = \cos i,

so the angle-integrated torus power is :math:`f L_{\rm bol}`, every sightline keeps the
library's flux relative to the others, and the shape between nodes is exactly that of the
per-sightline library. By symmetry about the equator
:math:`\tfrac12\int_0^\pi P(\theta)\sin\theta\,d\theta = \int_0^1 P(\mu)\,d\mu`.

The sphere mean is Gauss-Legendre of order ``TORUS_GL_ORDER`` on each interval between the
library's inclination nodes, in the variable the library is interpolated in, weighted by
:math:`|d\mu/dx|`. The panels run over the whole of :math:`\mu \in [0, 1]`; beyond an end
node the profile is whatever the lookup returns there (the end value, or the continued end
cubic), as the shape is.

A torus-only component has no disc of its own: the disc share :math:`(1 - f) L_{\rm bol}` is
published as ``log_L_agn_disc_unemitted`` so the ledger closes on the accretion power,
``10**agn_log_lbol L_sun`` (NAMING_CONTRACT section 4c, "Template tori"). The standalone
template tori publish it. The composable runner publishes the same angle-integrated torus power
from :func:`log10_anisotropy` of the block's library (``blocks/torus_anisotropy``), in its own
ledger (NAMING_CONTRACT section 4c, "Composable runner", #3076).
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from tengri.components.agn._publication import with_lsun_siblings
from tengri.components.agn._template_grid import (
    TorusTemplateGrid,
    float32_safe_template,
    native_bolometric_nu,
    native_bolometric_nu_np,
    torus_lnu_from_grid,
)
from tengri.utils.grid_interp import interp_nd_pchip
from tengri.utils.physics_constants import L_SUN as _LSUN_ERG
from tengri.utils.scale import log10_magnitude, pow10

__all__ = [
    "TORUS_GL_ORDER",
    "TORUS_LEDGER_KEYS",
    "AnisotropyGrid",
    "TorusInclination",
    "absent_torus_ledger",
    "anisotropy_at_nodes",
    "anisotropy_grid",
    "anisotropy_values",
    "inclination_quadrature_in",
    "library_anisotropy",
    "log10_anisotropy",
    "node_power_scale",
    "torus_ledger",
    "torus_ledger_from_grid",
    "torus_lnu_anisotropic",
    "torus_log_power",
]

#: ``log10(L_sun / (erg/s))`` [dex].
_LOG10_L_SUN: float = float(np.log10(_LSUN_ERG))

#: Interval of the inclination variable that spans ``cos i`` in ``[0, 1]``, ascending.
_X_RANGE: dict[str, tuple[float, float]] = {
    "cos_inc": (0.0, 1.0),
    "incl_deg": (0.0, 90.0),
    "elevation_deg": (0.0, 90.0),
}


#: Gauss-Legendre points per inclination panel. Each panel of the anisotropy profile is one
#: cubic times the smooth Jacobian |d mu/dx|. Measured against order 32 of the same
#: interpolant, for every restored library (Fritz at five points including the corners
#: (150, 10, 0, 6, 60) and (10, 0.1, -1, 0, 20), NK0_mean_1p/2p/3p across the opening-angle
#: and tau_V axes, SKIRTOR_mean_1p): the sphere mean is within 3.3e-16 at order 6 and
#: 6.5e-13 at order 4.
TORUS_GL_ORDER: int = 6


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


def inclination_quadrature_in(x_nodes, variable: str, order: int = TORUS_GL_ORDER):
    r"""Points and weights for :math:`\int_0^1 f\,d\mu` in a library's inclination variable.

    The traced, variable-aware sibling of ``_publication.inclination_quadrature`` (host
    constants, nodes in ``cos i`` only).

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

    Raises
    ------
    ValueError
        If concrete ``x_nodes`` are not strictly ascending (a descending axis would give
        negative panel widths).

    Notes
    -----
    **JIT/vmap-safe**: yes; the panel count is static, so the nodes may be a tracer. The
    panel edges are the nodes clipped into the range of ``variable`` plus its two ends; a
    node at an end gives an empty panel of zero weight. The weights are normalized to sum
    to 1, so the mean of a constant is that constant to rounding.
    """
    lo, hi = _X_RANGE[variable]
    if not isinstance(x_nodes, jax.core.Tracer) and np.any(np.diff(np.asarray(x_nodes)) <= 0):
        raise ValueError(f"inclination nodes must be strictly ascending, got {x_nodes!r}")
    nodes = jnp.asarray(x_nodes)
    ends = jnp.asarray([lo, hi], dtype=nodes.dtype)
    edges = jnp.concatenate([ends[:1], jnp.clip(nodes, lo, hi), ends[1:]])
    gx, gw = np.polynomial.legendre.leggauss(order)
    a, b = edges[:-1, None], edges[1:, None]
    x = (0.5 * (b - a) * gx + 0.5 * (b + a)).ravel()
    weight = (0.5 * (b - a) * gw).ravel() * _abs_dmu_dx(x, variable)
    return x, weight / jnp.sum(weight)


class AnisotropyGrid(NamedTuple):
    """A library's power along each sightline, relative to its model's mean over the sphere.

    Attributes
    ----------
    values : ndarray, shape (n_ax1, ..., n_axk)
        One scalar per library record, the leading shape of the template: the record's
        power over its model's mean over the inclination nodes [dimensionless].
    axes : tuple of ndarray
        The library's axes, as in ``TorusTemplateGrid.axes``.
    """

    values: jnp.ndarray
    axes: tuple[jnp.ndarray, ...]


def _mu_of(x, variable: str):
    """``cos i`` at the inclination coordinate ``x``."""
    if variable == "cos_inc":
        return x
    rad = jnp.pi / 180.0
    return jnp.cos(x * rad) if variable == "incl_deg" else jnp.sin(x * rad)


def anisotropy_grid(powers, axes, inc: TorusInclination) -> AnisotropyGrid:
    r"""Per-record powers divided by their model's mean over the sphere.

    Parameters
    ----------
    powers : array_like, shape (n_ax1, ..., n_axk)
        The power each record carries along its own sightline, on one common scale per
        model [any units].
    axes : tuple of array_like
        The library's axes.
    inc : TorusInclination
        The library's inclination axis.

    Returns
    -------
    AnisotropyGrid
        ``powers / pbar`` with ``pbar`` the model's trapezoid mean over :math:`\mu = \cos i`
        at its inclination nodes, so every model carries unit mean and its profile alone; the
        exact sphere mean at a parameter point is taken later on the interpolated profile.

    Notes
    -----
    **JIT-compatible**: yes.
    """
    powers = jnp.asarray(powers)
    mu = _mu_of(jnp.asarray(axes[inc.axis]), inc.variable)
    mean = jnp.trapezoid(powers, mu, axis=inc.axis) / (mu[-1] - mu[0])
    return AnisotropyGrid(
        values=powers / jnp.expand_dims(mean, inc.axis),
        axes=tuple(jnp.asarray(a) for a in axes),
    )


def anisotropy_values(template, wave, axes, inc: TorusInclination) -> np.ndarray:
    r"""Host (float64) :class:`AnisotropyGrid` values of a library, for caching at load.

    Parameters
    ----------
    template : array_like, shape (n_ax1, ..., n_axk, n_wave)
        Stored records [any common scale per model].
    wave : array_like, shape (n_wave,)
        The records' wavelength nodes [Angstrom].
    axes : tuple of array_like
        The library's axes.
    inc : TorusInclination
        The library's inclination axis.

    Returns
    -------
    ndarray, shape (n_ax1, ..., n_axk)
        Each record's frequency integral (``native_bolometric_nu_np``) over its model's
        trapezoid mean in :math:`\mu`, O(1) and float32-representable.
    """
    t = np.asarray(template, dtype=np.float64)
    flat = t.reshape(-1, t.shape[-1])
    powers = np.array([native_bolometric_nu_np(row, wave, floor=0.0) for row in flat])
    return np.asarray(anisotropy_grid(powers.reshape(t.shape[:-1]), axes, inc).values)


def library_anisotropy(grid: TorusTemplateGrid, inc: TorusInclination, dtype) -> AnisotropyGrid:
    r"""The :class:`AnisotropyGrid` of a library whose records share one scale per model.

    Parameters
    ----------
    grid : TorusTemplateGrid
        The library (its records at their stored scale).
    inc : TorusInclination
        The library's inclination axis.
    dtype : dtype
        Evaluation dtype; a float32 host table is divided by its peak first.

    Returns
    -------
    AnisotropyGrid
        Each record's frequency integral on its own wavelength nodes
        (``native_bolometric_nu``), over its model's mean (:func:`anisotropy_grid`).

    Notes
    -----
    **JIT/vmap-safe**: yes; the integrals are evaluated in the graph for a threaded grid.
    """
    if grid.anisotropy is not None:
        return AnisotropyGrid(
            values=jnp.asarray(grid.anisotropy), axes=tuple(jnp.asarray(a) for a in grid.axes)
        )
    return anisotropy_grid(_record_powers(grid, dtype), grid.axes, inc)


def _record_powers(grid: TorusTemplateGrid, dtype):
    """Each record's frequency integral on its own wavelength nodes, leading shape."""
    template = jnp.asarray(float32_safe_template(grid.template, dtype))
    wave = jnp.asarray(grid.wave_grid)
    flat = template.reshape(-1, template.shape[-1])
    powers = jax.vmap(lambda row: native_bolometric_nu(row, wave, floor=0.0))(flat)
    return powers.reshape(template.shape[:-1])


def _profile(aniso: AnisotropyGrid, coords: tuple, inc: TorusInclination):
    """The anisotropy at every inclination node, the other axes interpolated at ``coords``."""
    n_axes = len(aniso.axes)
    values = jnp.moveaxis(jnp.asarray(aniso.values), inc.axis, n_axes - 1)
    other = tuple(i for i in range(n_axes) if i != inc.axis)
    if not other:
        return values
    return interp_nd_pchip(
        values, tuple(aniso.axes[i] for i in other), tuple(coords[i] for i in other)
    )


def _profile_at(profile, x_nodes, x, extrapolate: bool):
    """Node-exact PCHIP of an inclination profile at ``x``."""
    return interp_nd_pchip(profile, (x_nodes,), (x,), extrapolate=(extrapolate,))


def _profile_mean(profile, x_nodes, inc: TorusInclination):
    """Sphere mean of the interpolated profile (Gauss-Legendre over the inclination panels)."""
    x_q, w_q = inclination_quadrature_in(x_nodes, inc.variable)
    return jnp.sum(
        w_q * jax.vmap(lambda x: _profile_at(profile, x_nodes, x, inc.extrapolate))(x_q)
    )


def log10_anisotropy(aniso: AnisotropyGrid, coords: tuple, inc: TorusInclination):
    r"""``log10`` of the sightline power over the sphere-mean power, :math:`a(\mu)/\bar a`.

    Parameters
    ----------
    aniso : AnisotropyGrid
        The library's anisotropy table.
    coords : tuple of float
        One coordinate per axis; the inclination entry is the sightline.
    inc : TorusInclination
        The library's inclination axis.

    Returns
    -------
    ndarray, shape ()
        :math:`\log_{10}(a(\mu)/\bar a)` [dex], with :math:`a` the table interpolated with
        node-exact PCHIP (Fritsch & Carlson [1]_, the lookup of #2606) over the other axes
        and then along the inclination, and :math:`\bar a = \int_0^1 a\,d\mu` (Gauss-Legendre
        of order ``TORUS_GL_ORDER`` on each inclination panel, in the library's variable).

    Notes
    -----
    **JIT/grad/vmap-safe**: yes; a ratio of the same profile, so it is float32 safe.

    References
    ----------
    .. [1] F. N. Fritsch & R. E. Carlson, "Monotone Piecewise Cubic Interpolation,"
       SIAM J. Numer. Anal. 17, 238 (1980). https://doi.org/10.1137/0717021
    """
    profile = _profile(aniso, coords, inc)
    x_nodes = aniso.axes[inc.axis]
    los = _profile_at(profile, x_nodes, coords[inc.axis], inc.extrapolate)
    return jnp.log10(los / _profile_mean(profile, x_nodes, inc))


def anisotropy_at_nodes(aniso: AnisotropyGrid, inc: TorusInclination) -> np.ndarray:
    r""":math:`a/\bar a` at every library node, for build-time tables (float64, host).

    Parameters
    ----------
    aniso : AnisotropyGrid
        The library's anisotropy table.
    inc : TorusInclination
        The library's inclination axis.

    Returns
    -------
    ndarray, shape (n_ax1, ..., n_axk)
        The factor :func:`torus_lnu_anisotropic` applies at each node (at a node the
        profile is the stored column, so this is the runtime value there).
    """
    n_axes = len(aniso.axes)
    values = jnp.moveaxis(jnp.asarray(aniso.values, dtype=jnp.float64), inc.axis, n_axes - 1)
    x_nodes = jnp.asarray(aniso.axes[inc.axis], dtype=jnp.float64)
    flat = values.reshape(-1, values.shape[-1])
    means = jax.jit(jax.vmap(lambda p: _profile_mean(p, x_nodes, inc)))(flat)
    ratio = np.asarray(flat / means[:, None]).reshape(values.shape)
    return np.moveaxis(ratio, n_axes - 1, inc.axis)


def node_power_scale(grid: TorusTemplateGrid, inc: TorusInclination) -> np.ndarray:
    r"""Per-record factor that turns a stored record into the emitted torus per unit power.

    Parameters
    ----------
    grid : TorusTemplateGrid
        The library (host arrays, as loaded).
    inc : TorusInclination
        The library's inclination axis.

    Returns
    -------
    ndarray, shape (n_ax1, ..., n_axk)
        :math:`(a/\bar a) / P` at each node, float64: a build-time table multiplied by it
        and by :math:`f L_{\rm bol}` equals :func:`torus_lnu_anisotropic` at that node.
    """
    powers = np.asarray(_record_powers(grid, jnp.float64))
    ratio = anisotropy_at_nodes(library_anisotropy(grid, inc, jnp.float64), inc)
    return ratio / powers


def torus_lnu_anisotropic(
    grid: TorusTemplateGrid,
    wavelength: jnp.ndarray,
    coords: tuple,
    inc: TorusInclination,
    *,
    agn_log_lbol: float,
    agn_torus_frac: float,
) -> jnp.ndarray:
    r"""Torus :math:`L_\nu` along one sightline: the per-sightline shape times the anisotropy.

    Parameters
    ----------
    grid : TorusTemplateGrid
        Library whose records share one scale per model across inclination.
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

        L_\nu(\lambda; \mu) = f\,L_{\rm bol}\,\frac{T_\nu(\lambda; \mu)}{\int T_\nu\,d\nu}
        \;\frac{a(\mu)}{\bar a},

    The first factor is ``torus_lnu_from_grid``, the records normalized to unit power and
    interpolated as for a library normalized per sightline (so the shape between nodes is
    that one exactly). The second is :func:`log10_anisotropy`, the scalar table of the
    records' own powers interpolated with the same node-exact PCHIP, so the angle-integrated
    power is :math:`f L_{\rm bol}` and the sightline power :math:`f L_{\rm bol}\,a/\bar a`.

    **JIT/grad/vmap-safe**: yes.
    """
    shape = torus_lnu_from_grid(
        grid, wavelength, coords, agn_log_lbol=agn_log_lbol, agn_torus_frac=agn_torus_frac
    )
    aniso = library_anisotropy(grid, inc, wavelength.dtype)
    return shape * pow10(log10_anisotropy(aniso, coords, inc))


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
    aniso: AnisotropyGrid,
    coords: tuple,
    inc: TorusInclination,
    *,
    agn_log_lbol,
    agn_torus_frac,
) -> dict:
    r"""The :func:`torus_ledger` of :func:`torus_lnu_anisotropic` at one point.

    Parameters
    ----------
    aniso : AnisotropyGrid
        The library's anisotropy table (:func:`library_anisotropy`; Fritz builds its own from
        ``norm``).
    coords : tuple of float
        Interpolation coordinate per axis; the inclination entry is the sightline.
    inc : TorusInclination
        The library's inclination axis.
    agn_log_lbol : float
        :math:`\log_{10}(L_{\rm bol}/L_\odot)` [dex].
    agn_torus_frac : float
        Fraction of :math:`L_{\rm bol}` reprocessed by the torus [dimensionless].

    Returns
    -------
    dict
        As :func:`torus_ledger`; the sightline power is :math:`f L_{\rm bol}\,a/\bar a`,
        the integral of the SED the kernel emits along ``coords``.

    Notes
    -----
    **JIT/grad/vmap-safe**: yes, and float32 safe: only the ratio :math:`a/\bar a` is formed
    from the table.
    """
    log_los = torus_log_power(agn_log_lbol, agn_torus_frac) + log10_anisotropy(aniso, coords, inc)
    return torus_ledger(agn_log_lbol, agn_torus_frac, log_los)
