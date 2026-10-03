# SPDX-License-Identifier: BSD-3-Clause
r"""Exact IGM fold of the sub-band photometry tensors, fixed and free redshift.

The node fold multiplies each sub-band integral by the transmission at the
sub-band's node, which forms :math:`\langle S\rangle\langle T\rangle` where the
integral needs :math:`\langle S T\rangle`. Where Ly-alpha falls inside a
sub-band that product is badly wrong (tens of percent in the dropout band). The
exact fold instead integrates the transmission inside the bandpass integral:

.. math::

   R_{k}(z) = \frac{\int_{k} S(\lambda)\,T_\mathrm{IGM}(\lambda (1+z), z)\,
                    T_b(\lambda (1+z))\,w\,d\lambda}
                   {\int_{k} S(\lambda)\,T_b(\lambda (1+z))\,w\,d\lambda},

where :math:`S` is an SSP template [erg/s/Hz], :math:`T_\mathrm{IGM}` the IGM
transmission [dimensionless], :math:`T_b` the filter response, :math:`w` the
bandpass weight of the filter convention, and :math:`k` a sub-band of
:func:`tengri.utils.grid_interp.subband_quadrature`. The folded tensor is the
sub-band tensor times :math:`R`.

The ratio is only meaningful against a table built with the **same partition**:
the same ``K``, the same forced Lyman-limit edge (``lyc_gate``) and the same
bandpass weight (``convention``). All three are explicit arguments here, never
defaults, because a mismatch keeps the shapes equal and fails silently.

Build-time only: numpy, eager float64.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

__all__ = ["SubbandFold", "subband_fold", "subband_fold_table"]


class SubbandFold(NamedTuple):
    """The exact fold of one sub-band tensor.

    Attributes
    ----------
    ratio : ndarray, shape (..., n_filters, n_chunks)
        With-IGM over bare sub-band integral [dimensionless].
    nodes_rest : ndarray, same shape
        Flux-weighted centroid of each chunk WITH the IGM inside [Angstrom], rest
        frame: where the light that survives the IGM sits, which is where a dust
        screen multiplying it must be evaluated. NaN for bands the IGM cannot
        reach (their node is the bare one) and the bare node wherever the IGM
        removes a chunk entirely.
    """

    ratio: np.ndarray
    nodes_rest: np.ndarray


def _transmission(igm_model, wave_rest, z):
    """IGM transmission on the rest grid, evaluated at observed wavelengths."""
    from tengri.components.igm import igm_absorption

    # The one frame conversion: T_IGM takes observed-frame wavelength, the SSP
    # grid is rest-frame, and preintegrate_grid expects rest-frame templates.
    return np.asarray(
        igm_absorption(
            wave_rest * (1.0 + z), z, igm_patchy=False, igm_model=igm_model, use_dla=False
        ),
        dtype=np.float64,
    )


def _absorbed_filters(transmission, wave_rest, z, filter_waves):
    """Indices of filters whose support reaches wavelengths the IGM absorbs.

    ``T < 1`` only blueward of Ly-alpha, so a filter lying wholly redward of the
    last absorbed template node sees identical integrands with and without the
    IGM and its ratio is exactly one. The bound is the first *unabsorbed* node
    after the last absorbed one: the quadrature interpolates templates linearly
    between nodes, so the segment up to it still carries the absorption.
    """
    absorbed = np.nonzero(transmission < 1.0)[0]
    if absorbed.size == 0:
        return []
    bound_obs = wave_rest[min(absorbed[-1] + 1, wave_rest.size - 1)] * (1.0 + z)
    return [i for i, fw in enumerate(filter_waves) if float(np.min(fw)) < bound_obs]


def subband_fold(
    ssp_data, filters, z, *, igm_model, n_subbands, lyc_gate, convention
) -> SubbandFold:
    """Exact-to-bare ratio of the sub-band integrals, and the with-IGM nodes, at one z.

    Parameters
    ----------
    ssp_data : SSPData
        Template grid, ``ssp_flux`` shape ``(n_met, n_age, n_wave)`` [erg/s/Hz/Msun].
    filters : sequence of (wave, trans) pairs
        Filter curves, observed frame [Angstrom], [dimensionless].
    z : float
        Redshift [dimensionless].
    igm_model : str
        Transmission law passed to :func:`igm_absorption`.
    n_subbands : int
        Base sub-band count ``K`` of the table the ratio multiplies (without
        the extra chunk ``lyc_gate`` adds).
    lyc_gate : bool
        Whether that table carries the forced Lyman-limit edge (``K + 1`` chunks).
    convention : FilterConvention
        Bandpass weight of that table.

    Returns
    -------
    SubbandFold
        Arrays of shape ``(n_met, n_age, n_filters, n_chunks)``, ``n_chunks =
        K + 1`` under ``lyc_gate`` else ``K``. The ratio is zero where the bare
        integral is zero (no flux either way) and exactly one for filters the
        IGM cannot reach.
    """
    from tengri.utils.grid_interp import preintegrate_grid

    wave_rest = np.asarray(ssp_data.ssp_wave, dtype=np.float64)
    templates = np.asarray(ssp_data.ssp_flux, dtype=np.float64)
    filter_waves = [np.asarray(fw, dtype=np.float64) for fw, _ in filters]
    filter_trans = [np.asarray(ft, dtype=np.float64) for _, ft in filters]
    n_chunks = n_subbands + 1 if lyc_gate else n_subbands

    shape = (*templates.shape[:-1], len(filters), n_chunks)
    ratio = np.ones(shape, dtype=np.float64)
    nodes = np.full(shape, np.nan, dtype=np.float64)
    transmission = _transmission(igm_model, wave_rest, z)
    reached = _absorbed_filters(transmission, wave_rest, z, filter_waves)
    if not reached:
        return SubbandFold(ratio, nodes)

    def _quadrature(templates_in):
        # dl_cm is a constant factor of both integrals and cancels in the ratio.
        grid = preintegrate_grid(
            templates=templates_in,
            wave_rest=wave_rest,
            filter_waves=[filter_waves[i] for i in reached],
            filter_trans=[filter_trans[i] for i in reached],
            redshift=z,
            dl_cm=1.0,
            axes=(np.asarray(ssp_data.ssp_lgmet), np.asarray(ssp_data.ssp_lg_age_gyr)),
            taylor=False,
            n_subbands=n_subbands,
            convention=convention,
            lyc_gate=lyc_gate,
        )
        return (
            np.asarray(grid.subband_phot, dtype=np.float64),
            np.asarray(grid.subband_waves_rest, dtype=np.float64),
        )

    bare, bare_nodes = _quadrature(templates)
    folded, folded_nodes = _quadrature(templates * transmission)
    ratio[..., reached, :] = np.where(bare != 0.0, folded / np.where(bare != 0.0, bare, 1.0), 0.0)
    nodes[..., reached, :] = np.where(folded != 0.0, folded_nodes, bare_nodes)
    return SubbandFold(ratio, nodes)


def subband_fold_table(
    ssp_data, filters, z_grid, *, igm_model, n_subbands, lyc_gate, convention
) -> SubbandFold:
    """:func:`subband_fold` on every node of a z grid, content-cached.

    The table is a build-time constant of (SSP grid, filters, z grid, partition,
    IGM law), so it is persisted beside the photometry z-table under
    ``~/.cache/tengri_precomp`` and memoized in process; see
    ``tengri.components.igm._subband_cache`` for the key and the knobs.

    Parameters
    ----------
    ssp_data, filters, igm_model, n_subbands, lyc_gate, convention
        As :func:`subband_fold`.
    z_grid : array_like, shape (n_z,)
        Redshift nodes of the z-table the ratio multiplies.

    Returns
    -------
    SubbandFold
        Arrays of shape ``(n_z, n_met, n_age, n_filters, n_chunks)``.
    """
    from tengri.components.igm import _subband_cache

    z_grid = np.asarray(z_grid, dtype=np.float64)
    key = _subband_cache.exact_fold_key(
        ssp_data,
        filters,
        z_grid,
        igm_model=igm_model,
        n_subbands=n_subbands,
        lyc_gate=lyc_gate,
        convention=convention,
    )
    cached = _subband_cache.memo_get(key)
    if cached is None:
        cached = _subband_cache.load(key, prefix=_subband_cache.EXACT_FOLD_PREFIX)
        if cached is not None:
            _subband_cache.memo_put(key, cached)
    if cached is not None:
        return SubbandFold(*np.asarray(cached))

    folds = [
        subband_fold(
            ssp_data,
            filters,
            float(z),
            igm_model=igm_model,
            n_subbands=n_subbands,
            lyc_gate=lyc_gate,
            convention=convention,
        )
        for z in z_grid
    ]
    # One array on disk: (2, n_z, ...) = (ratio, nodes).
    table = np.stack([np.stack([f.ratio for f in folds]), np.stack([f.nodes_rest for f in folds])])
    _subband_cache.memo_put(key, table)
    _subband_cache.store(key, table, prefix=_subband_cache.EXACT_FOLD_PREFIX)
    return SubbandFold(*table)
