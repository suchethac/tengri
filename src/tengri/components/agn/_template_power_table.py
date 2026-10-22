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
per :math:`L_\odot` of torus power [erg/s/Hz], and :math:`L_\odot` [erg/s]. At a node
this is the node-normalized photometry exactly; between nodes it commutes with the
runtime wherever the interpolant is linear in the tabulated values, leaving only the
data dependence of the PCHIP tangents.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from tengri.components.agn._template_grid import native_bolometric_nu_np
from tengri.forward.precompute.templates import precompute_template_photometry
from tengri.utils.grid_interp import PreintegratedGrid, interp_nd_pchip
from tengri.utils.physics_constants import L_SUN as _LSUN_ERG


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
    preint = precompute_template_photometry(
        templates=_LSUN_ERG * templates / power[..., None],
        wave_rest=wave_rest,
        filter_waves=[np.asarray(fw) for fw in filter_waves],
        filter_trans=[np.asarray(ft) for ft in filter_trans],
        redshift=redshift,
        dl_cm=1.0,
        axes=axes,
        energy_normalize=False,
        units="lnu",
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


def read_at_fixed_axes(table, axes, free_values: tuple, collapsed: dict[int, float]):
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

    Returns
    -------
    ndarray, shape (n_channels,)
        The interpolated table row.
    """
    return interp_nd_pchip(table, axes, full_coords(free_values, collapsed, len(axes)))
