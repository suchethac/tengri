# SPDX-License-Identifier: BSD-3-Clause
r"""Power-weighted photometry tables for torus libraries normalized after interpolation.

Every template torus runtime kernel here (``_template_grid.torus_lnu_from_grid``, used by
the SKIRTOR-AGNfitter and CAT3D-WIND reductions and, for its shape factor, by the
anisotropic ``_torus_inclination.torus_lnu_anisotropic`` of SKIRTOR ``mean_1p`` and the
Nenkova reductions; and the Silva et al. 2004 kernel) interpolates the *raw* library
templates and then normalizes the interpolated template by its own bolometric integral,

.. math::

    L_\nu(\lambda;\,\mathbf{c}) = L_{\rm bol}\,f_{\rm torus}\,
        \frac{\mathcal{I}[T](\lambda;\,\mathbf{c})}
             {\int \mathcal{I}[T](\nu;\,\mathbf{c})\,\mathrm{d}\nu},

where :math:`\mathcal{I}` is the node-exact PCHIP interpolant over the library axes
:math:`\mathbf{c}` and :math:`T` the stored templates. The normalized interpolant is
therefore weighted by each node's raw power. A table that interpolates node-normalized
band photometry weights every node equally instead. Where the raw power changes by a
large factor between nodes (the SKIRTOR-AGNfitter type-1/type-2 step at
:math:`i = 90^\circ - \theta_{\rm oa}`, the CAT3D-WIND low-inclination cells) the two
departed off-node by up to 30%.

The single-axis anisotropic tables (``skirtor_agnfitter_1p``, ``nenkova_agnfitter``) are already
power-weighted, exactly, without this module: they store each record times
:math:`(a/\bar a)/P`, and with one library axis the anisotropy is the record power over
one constant, :math:`a \propto P`, so every node holds :math:`\Phi^{\rm raw}/\bar a`, the
raw band photometry over a constant. PCHIP is scale-equivariant, so reading that table is
the ratio of interpolants times :math:`a/\bar a`. The Nenkova ``mean_2p``/``mean_3p``
reductions need this module because :math:`\bar a` differs between their opening-angle
and optical-depth models, which breaks the identity.

The table built here carries the same two factors the runtime forms: the band
photometry of the raw templates and, in one extra trailing channel, their native
bolometric integral. The lookup interpolates both and divides,

.. math::

    \Phi_b(\mathbf{c}) = L_\odot\,
        \frac{\mathcal{I}[\Phi_b^{\rm raw}](\mathbf{c})}{\mathcal{I}[P^{\rm raw}](\mathbf{c})},
    \qquad
    \Phi_b^{\rm raw} = \frac{\int T\,R_b\,\lambda\,\mathrm{d}\lambda}
                            {\int R_b\,\lambda\,\mathrm{d}\lambda},
    \qquad
    P^{\rm raw} = \int T\,\mathrm{d}\nu,

where :math:`R_b` is the filter transmission, :math:`\Phi_b` the band :math:`L_\nu`
per :math:`L_\odot` of torus power [erg/s/Hz], and :math:`L_\odot` [erg/s]. The band
integral is the runtime's own: the template resampled log-log onto the filter
quadrature grid and folded with ``lnu_filter_integral``
(:func:`runtime_band_photometry`, below), not a linear quadrature
between the template's own wavelengths, which sat up to 50% off the runtime at the
nodes (#3160). At a node the lookup is therefore the runtime's photometry to rounding;
between nodes it commutes with the runtime wherever the interpolant is linear in the
tabulated values, leaving only the data dependence of the PCHIP tangents and the
nonlinearity of the log-log resampling.

**The runtime's band quadrature.** Every torus table here, and ``skirtor_precompute``,
integrates the filters as the runtime does: the template is resampled log-log
(:func:`~tengri.utils.grid_interp.resample_template`) onto the quadrature grid of
:func:`quadrature_grid` (the template nodes that bracket each filter's rest-frame
support plus every rest-frame filter node), and the runtime's filter integral, which
is linear in the SED, is applied as a matrix (:func:`band_fold`, its Jacobian):

.. math::

    \Phi_b = \sum_q \mathcal{R}[T](\lambda_q)\,F_{qb},
    \qquad
    F_{qb} = \frac{\partial\,\mathcal{L}_b[L_\nu]}{\partial L_\nu(\lambda_q)},

where :math:`\mathcal{R}` is the log-log resampling, :math:`\lambda_q` [Angstrom] the
quadrature grid and :math:`\mathcal{L}_b` the band integral of
``lnu_filter_integral``. The result equals the runtime's band photometry on a model
grid of the native and filter nodes to rounding.
"""

from __future__ import annotations

import dataclasses
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from tengri.components.agn._template_grid import native_bolometric_nu_np
from tengri.forward.precompute.templates import precompute_template_photometry
from tengri.observation.photometry import lnu_filter_integral
from tengri.utils.grid_interp import PreintegratedGrid, interp_nd_pchip, resample_template
from tengri.utils.physics_constants import L_SUN as _LSUN_ERG

#: Records resampled and folded per ``lax.map`` step.
_RECORD_BATCH: int = 64


def quadrature_grid(
    wave: np.ndarray, filter_waves: list, redshift: float
) -> tuple[np.ndarray, np.ndarray]:
    """Template wavelengths the filters read, and the rest-frame quadrature grid.

    Parameters
    ----------
    wave : array_like, shape (n_wave,)
        Template rest-frame wavelengths [Angstrom], ascending.
    filter_waves : list of array_like
        Wavelength grid per filter [Angstrom], observed frame, ascending.
    redshift : float
        Source redshift [dimensionless].

    Returns
    -------
    read : ndarray of int, shape (n_read,)
        Indices of the template nodes that bracket any filter's rest-frame support.
    wave_quad : ndarray, shape (n_quad,)
        The union of those nodes with every filter node in the rest frame [Angstrom].
        Every quadrature point lies between two adjacent kept nodes, so the log-log
        resampling over the kept nodes equals the runtime's over the whole template.
    """
    wave = np.asarray(wave, dtype=np.float64)
    n_wave = wave.shape[0]
    keep = np.zeros(n_wave, dtype=bool)
    rest = [np.asarray(fw, dtype=np.float64) / (1.0 + redshift) for fw in filter_waves]
    for fw in rest:
        lo = max(int(np.searchsorted(wave, fw[0], side="right")) - 1, 0)
        hi = min(int(np.searchsorted(wave, fw[-1], side="left")), n_wave - 1)
        keep[lo : max(hi, lo + 1) + 1] = True
    read = np.flatnonzero(keep)
    return read, np.unique(np.concatenate([wave[read], *rest]))


def band_fold(
    wave_quad: np.ndarray, filter_waves: list, filter_trans: list, redshift: float
) -> jnp.ndarray:
    """The runtime's filter quadrature on ``wave_quad`` as a matrix.

    Parameters
    ----------
    wave_quad : array_like, shape (n_quad,)
        Rest-frame quadrature grid [Angstrom].
    filter_waves : list of array_like
        Wavelength grid per filter [Angstrom], observed frame.
    filter_trans : list of array_like
        Transmission per filter [dimensionless].
    redshift : float
        Source redshift [dimensionless].

    Returns
    -------
    ndarray, shape (n_quad, n_filters)
        :math:`F_{qb}` [dimensionless]: an SED on ``wave_quad`` [erg/s/Hz] times this
        matrix is its band L_nu [erg/s/Hz]. ``lnu_filter_integral`` is linear in the
        SED, so its Jacobian is the quadrature.
    """
    wq = jnp.asarray(wave_quad)

    def bands(lnu):
        return jnp.stack(
            [
                lnu_filter_integral(lnu, wq, jnp.asarray(fw), jnp.asarray(ft), redshift)
                for fw, ft in zip(filter_waves, filter_trans, strict=True)
            ]
        )

    return jax.jacrev(bands)(jnp.zeros_like(wq)).T


def runtime_band_photometry(
    templates: np.ndarray,
    wave_rest: np.ndarray,
    filter_waves: list,
    filter_trans: list,
    redshift: float,
) -> np.ndarray:
    r"""Band photometry of every template, emitted and integrated as the runtime does.

    Parameters
    ----------
    templates : array_like, shape (*grid_dims, n_wave)
        Templates, :math:`L_\nu`-like [any units per Hz], non-negative.
    wave_rest : array_like, shape (n_wave,)
        Rest-frame template wavelengths [Angstrom], ascending.
    filter_waves : list of array_like
        Wavelength grid per filter [Angstrom], observed frame, ascending.
    filter_trans : list of array_like
        Transmission per filter [dimensionless].
    redshift : float
        Source redshift [dimensionless].

    Returns
    -------
    ndarray, shape (*grid_dims, n_filters)
        :math:`\Phi_b` of every template, in the templates' units [per Hz], float64.

    Notes
    -----
    **JIT-compatible**: no, this is a build-time function.

    Each record is divided by its own peak (in float64, on the host) before it reaches
    JAX and multiplied back afterwards. The resampling is homogeneous of degree one in
    the template values, so this changes nothing in float64, and in float32 it keeps
    the libraries stored at :math:`\sim 10^{-70}` from flushing to zero.
    """
    templates = np.asarray(templates, dtype=np.float64)
    wave_rest = np.asarray(wave_rest, dtype=np.float64)
    grid_dims = templates.shape[:-1]
    flat = templates.reshape(-1, templates.shape[-1])
    peak = np.max(np.abs(flat), axis=-1)
    peak = np.where(peak > 0.0, peak, 1.0)

    read, wave_quad = quadrature_grid(wave_rest, filter_waves, redshift)
    fold = band_fold(wave_quad, filter_waves, filter_trans, redshift)
    wave_read = jnp.asarray(wave_rest[read])
    wq = jnp.asarray(wave_quad)

    def one(record):
        return resample_template(wq, wave_read, record, left=0.0, right=0.0) @ fold

    shapes = jnp.asarray(flat[:, read] / peak[:, None])
    phot = jax.jit(lambda r: jax.lax.map(one, r, batch_size=_RECORD_BATCH))(shapes)
    phot = np.asarray(phot, dtype=np.float64) * peak[:, None]
    return phot.reshape(*grid_dims, phot.shape[-1])


def runtime_preintegrated_grid(
    templates: np.ndarray,
    wave_rest: np.ndarray,
    filter_waves: list,
    filter_trans: list,
    redshift: float,
    axes: tuple[np.ndarray, ...],
) -> PreintegratedGrid:
    r"""A :class:`PreintegratedGrid` whose band photometry is the runtime's.

    Parameters
    ----------
    templates : array_like, shape (*grid_dims, n_wave)
        Templates, :math:`L_\nu` [erg/s/Hz per unit of the lookup's scale].
    wave_rest : array_like, shape (n_wave,)
        Rest-frame template wavelengths [Angstrom], ascending.
    filter_waves : list of array_like
        Wavelength grid per filter [Angstrom], observed frame, ascending.
    filter_trans : list of array_like
        Transmission per filter [dimensionless].
    redshift : float
        Source redshift [dimensionless].
    axes : tuple of array_like
        Node coordinates, one array per grid dimension.

    Returns
    -------
    PreintegratedGrid
        ``phot``, shape ``(*grid_dims, n_filters)``, from :func:`runtime_band_photometry`
        [units of ``templates``]; the remaining fields (axes, edges, effective
        wavelengths) from ``precompute_template_photometry``, for the Fixed-axis
        bookkeeping of ``collapse_fixed_axes``.
    """
    filter_waves = [np.asarray(fw) for fw in filter_waves]
    filter_trans = [np.asarray(ft) for ft in filter_trans]
    bookkeeping = precompute_template_photometry(
        templates=templates,
        wave_rest=wave_rest,
        filter_waves=filter_waves,
        filter_trans=filter_trans,
        redshift=redshift,
        dl_cm=1.0,
        axes=axes,
        energy_normalize=False,
        units="lnu",
    )
    phot = runtime_band_photometry(templates, wave_rest, filter_waves, filter_trans, redshift)
    return dataclasses.replace(
        bookkeeping, phot=jnp.asarray(phot, dtype=jnp.asarray(bookkeeping.phot).dtype)
    )


class PowerWeightedTable(NamedTuple):
    """A power-weighted torus photometry table and its node-normalized band grid."""

    table: jnp.ndarray
    r"""shape (*grid_dims, n_filters + 1): the band channels
    :math:`L_\odot\,\Phi_b^{\rm raw}/P_{\rm ref}` [erg/s/Hz per unit template power], then
    the power channel :math:`P^{\rm raw}/P_{\rm ref}` [dimensionless]."""
    preint: PreintegratedGrid
    r"""The node-normalized band photometry, :math:`L_\nu` per :math:`L_\odot` [erg/s/Hz],
    with ``n_filters`` equal to its channel count. It equals ``table[..., :-1] /
    table[..., -1:]`` at every node; interpolated on its own it weights the nodes
    equally, which is not the runtime's reading."""


def power_weighted_template_photometry(
    templates: np.ndarray,
    wave_rest: np.ndarray,
    filter_waves: list,
    filter_trans: list,
    redshift: float,
    axes: tuple[np.ndarray, ...],
) -> PowerWeightedTable:
    r"""Band photometry of the raw templates plus their bolometric power as a last channel.

    Parameters
    ----------
    templates : array_like, shape (*grid_dims, n_wave)
        Raw library templates, :math:`L_\nu`-like, shape only [arbitrary units per Hz].
    wave_rest : array_like, shape (n_wave,)
        Rest-frame template wavelength grid [Angstrom].
    filter_waves : list of array_like
        Wavelength grid per filter [Angstrom], observed frame.
    filter_trans : list of array_like
        Transmission per filter [dimensionless].
    redshift : float
        Source redshift [dimensionless].
    axes : tuple of array_like
        Node coordinates, one array per grid dimension.

    Returns
    -------
    PowerWeightedTable
        ``table``, shape ``(*grid_dims, n_filters + 1)``, read with
        ``lnu_per_lsun_from_power_table`` after interpolating; :math:`P_{\rm ref}` is the
        median node power, so every channel stays of order the node-normalized values.
        ``preint``, the node-normalized band grid (``n_filters`` channels), for the
        Fixed-axis bookkeeping of ``collapse_fixed_axes``.

    Raises
    ------
    ValueError
        If any record's bolometric power is not finite and positive: the power channel
        is the divisor of every lookup, and a zero there is 0/0 off-node.

    Notes
    -----
    **JIT-compatible**: no, this is a build-time function.

    The band channels are the runtime's band photometry of each record
    (:func:`runtime_preintegrated_grid`): the record resampled
    log-log onto the filter quadrature grid and integrated with ``lnu_filter_integral``.
    """
    templates = np.asarray(templates, dtype=np.float64)
    wave_rest = np.asarray(wave_rest, dtype=np.float64)
    grid_dims = templates.shape[:-1]
    flat = templates.reshape(-1, templates.shape[-1])
    power = np.array([native_bolometric_nu_np(t, wave_rest, floor=0.0) for t in flat]).reshape(
        grid_dims
    )
    bad = ~(np.isfinite(power) & (power > 0.0))
    if bad.any():
        raise ValueError(
            f"{int(bad.sum())} torus template record(s) have non-positive or non-finite "
            f"bolometric power (first at grid index {np.argwhere(bad)[0].tolist()}); the "
            "power-weighted table divides by that power and cannot represent them."
        )
    preint = runtime_preintegrated_grid(
        _LSUN_ERG * templates / power[..., None],
        wave_rest,
        filter_waves,
        filter_trans,
        redshift,
        axes,
    )
    weight = jnp.asarray(power / float(np.median(power)), dtype=jnp.asarray(preint.phot).dtype)
    table = jnp.concatenate([preint.phot * weight[..., None], weight[..., None]], axis=-1)
    return PowerWeightedTable(table=table, preint=preint)


def lnu_per_lsun_from_power_table(values: jnp.ndarray) -> jnp.ndarray:
    r"""Divide an interpolated power-weighted table row by its power channel.

    Parameters
    ----------
    values : array_like, shape (n_filters + 1,)
        A row of ``PowerWeightedTable.table``, interpolated.

    Returns
    -------
    ndarray, shape (n_filters,)
        Band :math:`L_\nu` per :math:`L_\odot` of torus power [erg/s/Hz].
    """
    return values[..., :-1] / values[..., -1:]


def full_coords(free_values: tuple, collapsed: dict[int, float], n_axes: int) -> tuple:
    """Library coordinates of every axis, the collapsed ones at their fixed values.

    Parameters
    ----------
    free_values : tuple
        Coordinates of the axes left free, in axis order.
    collapsed : dict of int to float
        Axis index to the value it was collapsed at.
    n_axes : int
        Number of library axes.

    Returns
    -------
    tuple
        One coordinate per library axis, in axis order.
    """
    free = iter(free_values)
    return tuple(collapsed[k] if k in collapsed else next(free) for k in range(n_axes))


def read_at_fixed_axes(
    table,
    axes,
    free_values: tuple,
    collapsed: dict[int, float],
    *,
    extrapolate: tuple[bool, ...] | None = None,
):
    """Interpolate the full table with the Fixed axes held at their values.

    The tensor-product PCHIP is not independent of axis order (the Fritsch-Carlson
    tangents depend on the data), so contracting a Fixed axis first would read the table
    in another order than the runtime kernel does. The table is therefore kept whole and
    read at every axis, in the runtime's order; the read costs four nodes per axis either
    way.

    Parameters
    ----------
    table : array_like, shape (*grid_dims, n_channels)
        ``PowerWeightedTable.table``.
    axes : tuple of array_like
        Node coordinates of every grid axis.
    free_values : tuple
        Coordinates of the free axes, in axis order, in the table's variables.
    collapsed : dict of int to float
        Axis index to its Fixed coordinate, in the table's variable.
    extrapolate : tuple of bool, optional
        Per-axis extrapolation flags of the runtime kernel, passed to
        :func:`~tengri.utils.grid_interp.interp_nd_pchip`. Default: clip on every axis.

    Returns
    -------
    ndarray, shape (n_channels,)
        The interpolated table row.
    """
    return interp_nd_pchip(
        table, axes, full_coords(free_values, collapsed, len(axes)), extrapolate=extrapolate
    )
