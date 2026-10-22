# SPDX-License-Identifier: BSD-3-Clause
r"""Power-weighted photometry tables for the SKIRTOR-AGNfitter ``mean_2p``/``mean_3p`` tori.

The runtime kernel (``_template_grid.torus_lnu_from_grid``)
interpolates the *raw* library templates and then normalizes the interpolated template
by its own bolometric integral,

.. math::

    L_\nu(\lambda;\,\mathbf{c}) = L_{\rm bol}\,f_{\rm torus}\,
        \frac{\mathcal{I}[T](\lambda;\,\mathbf{c})}
             {\int \mathcal{I}[T](\nu;\,\mathbf{c})\,\mathrm{d}\nu},

where :math:`\mathcal{I}` is the node-exact PCHIP interpolant over the library axes
:math:`\mathbf{c}` and :math:`T` the stored templates. The raw templates of these two
reductions differ in bolometric power between nodes by up to a factor of several across
the type-1/type-2 step at :math:`i = 90^\circ - \theta_{\rm oa}`, so the normalized
interpolant is weighted by each node's raw power. A table that interpolates
node-normalized band photometry weights every node equally instead, and off-node it
departed from the runtime by up to 16.8% (``mean_2p``) and 29.6% (``mean_3p``).

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
data dependence of the PCHIP tangents (measured below 2.4e-2 on the five-filter
contract set).
"""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp
import numpy as np

from tengri.components.agn._template_grid import native_bolometric_nu_np
from tengri.forward.precompute.templates import precompute_template_photometry
from tengri.utils.grid_interp import PreintegratedGrid
from tengri.utils.physics_constants import L_SUN as _LSUN_ERG


def power_weighted_template_photometry(
    templates: np.ndarray,
    wave_rest: np.ndarray,
    filter_waves: list,
    filter_trans: list,
    redshift: float,
    axes: tuple[np.ndarray, ...],
) -> PreintegratedGrid:
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
    PreintegratedGrid
        ``phot`` has shape ``(*grid_dims, n_filters + 1)``: the first ``n_filters``
        channels are :math:`L_\odot\,\Phi_b^{\rm raw}/P_{\rm ref}` [erg/s/Hz per unit
        template power], the last :math:`P^{\rm raw}/P_{\rm ref}` [dimensionless], with
        :math:`P_{\rm ref}` the median node power so both stay of order the
        node-normalized values. Read it with ``lnu_per_lsun_from_power_table``.
    """
    templates = np.asarray(templates, dtype=np.float64)
    wave_rest = np.asarray(wave_rest, dtype=np.float64)
    grid_dims = templates.shape[:-1]
    flat = templates.reshape(-1, templates.shape[-1])
    power = np.array([native_bolometric_nu_np(t, wave_rest) for t in flat]).reshape(grid_dims)
    p_ref = float(np.median(power))
    preint = precompute_template_photometry(
        templates=_LSUN_ERG * templates / p_ref,
        wave_rest=wave_rest,
        filter_waves=[np.asarray(fw) for fw in filter_waves],
        filter_trans=[np.asarray(ft) for ft in filter_trans],
        redshift=redshift,
        dl_cm=1.0,
        axes=axes,
        energy_normalize=False,
        units="lnu",
    )
    channel = jnp.asarray(power / p_ref, dtype=jnp.asarray(preint.phot).dtype)[..., None]
    return dataclasses.replace(preint, phot=jnp.concatenate([preint.phot, channel], axis=-1))


def lnu_per_lsun_from_power_table(values: jnp.ndarray) -> jnp.ndarray:
    r"""Divide an interpolated power-weighted table row by its power channel.

    Parameters
    ----------
    values : array_like, shape (n_filters + 1,)
        A row of the ``phot`` of ``power_weighted_template_photometry``, interpolated.

    Returns
    -------
    ndarray, shape (n_filters,)
        Band :math:`L_\nu` per :math:`L_\odot` of torus power [erg/s/Hz].
    """
    return values[..., :-1] / values[..., -1:]
