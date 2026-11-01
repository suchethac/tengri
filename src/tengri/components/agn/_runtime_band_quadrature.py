# SPDX-License-Identifier: BSD-3-Clause
r"""Band photometry of tabulated torus templates on the runtime's own quadrature.

A torus runtime kernel emits a template by resampling it log-log onto the model's
wavelength grid (:func:`~tengri.utils.grid_interp.resample_template`, a power law
between adjacent template nodes) and the photometry step integrates that SED through
each filter with :func:`~tengri.observation.photometry.lnu_filter_integral`. A table
built by integrating the filters linearly between the coarse template wavelengths
(:math:`R \sim 7` for several libraries) is a different quadrature and sits off the
runtime at the very nodes it tabulates, by up to 19% on a band that falls between two
template wavelengths.

The functions here build the band photometry the runtime produces instead. The
quadrature grid is the union of the template nodes that bracket each filter's
rest-frame support with every filter node in the rest frame; the template is
resampled onto it with the runtime's resampling, and the runtime's filter integral,
which is linear in the SED, is applied as a matrix (its Jacobian):

.. math::

    \Phi_b = \sum_q \mathcal{R}[T](\lambda_q)\,F_{qb},
    \qquad
    F_{qb} = \frac{\partial\,\mathcal{L}_b[L_\nu]}{\partial L_\nu(\lambda_q)},

where :math:`T` is a template [arbitrary units per Hz], :math:`\mathcal{R}` the
log-log resampling, :math:`\lambda_q` [Angstrom] the quadrature grid,
:math:`\mathcal{L}_b` the runtime's band integral and :math:`\Phi_b` the band
:math:`L_\nu` in the template's units. Because the grid holds every filter node and
every template node inside the band, the result equals the runtime's band
photometry evaluated on a model grid of those nodes, to rounding.

``skirtor_precompute`` (#3158) builds its quadrature with the same two steps; the
functions are kept here so both constructions can share them.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from tengri.forward.precompute.templates import precompute_template_photometry
from tengri.observation.photometry import lnu_filter_integral
from tengri.utils.grid_interp import PreintegratedGrid, resample_template

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
