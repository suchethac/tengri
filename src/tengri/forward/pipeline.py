# SPDX-License-Identifier: BSD-3-Clause
"""Stellar SED interpolation kernels used by :class:`~tengri.SEDModel`.

Computational annex of ``forward/sed_model.py`` (its only importer):
SSP grid interpolation, metallicity weighting, and CSP assembly helpers
that translate an SFH into a rest-frame stellar SED.

Not the component chain — the driver that threads ``ForwardState``
through the ordered physics components (dust, nebular, AGN, IGM, …)
lives in :mod:`tengri.forward.orchestrator`.

All functions take a ``model`` argument (the
:class:`~tengri.forward.sed_model.SEDModel` instance) instead of
``self``, allowing the heavy computation to live outside the class
while preserving access to model state.
"""

from __future__ import annotations

from tengri.components.stellar.sps.dsps_wrapper import (
    interpolate_metallicity,
    interpolate_metallicity_evolving,
    interpolate_metallicity_smooth,
    interpolate_metallicity_smooth_evolving,
)


def interp_metallicity(model, log_z, ssp_flux=None, ssp_lgmet=None):
    """Dispatch metallicity interpolation on SSP grid (single Z value).

    Parameters
    ----------
    model : SEDModel
        The model instance.
    log_z : float
        log10(Z) absolute metallicity [dimensionless].
    ssp_flux : ndarray, optional
        Traced override for ``model.ssp_data.ssp_flux``. When provided
        with ``ssp_lgmet``, the SSP arrays enter the JIT graph as
        runtime tensors instead of closure-captured constants — the
        memory-efficient path; see
        ``docs/dev/quickstart_oom_diagnosis.md``.
    ssp_lgmet : ndarray, optional
        Traced override for ``model.ssp_data.ssp_lgmet``.

    Returns
    -------
    ndarray, shape (n_age, n_wave)
        SSP flux interpolated to target metallicity [erg/s/Hz].

    Notes
    -----
    **JIT-compatible**: yes — uses smooth or nearest-neighbor interpolation.
    """
    flux = ssp_flux if ssp_flux is not None else model.ssp_data.ssp_flux
    lgmet = ssp_lgmet if ssp_lgmet is not None else model.ssp_data.ssp_lgmet
    if model._met_interp == "smooth":
        return interpolate_metallicity_smooth(
            flux,
            lgmet,
            log_z,
            model._lgmet_scatter,
        )
    return interpolate_metallicity(flux, lgmet, log_z)


def interp_metallicity_evolving(model, log_z_per_age, ssp_flux=None, ssp_lgmet=None):
    """Dispatch per-age metallicity interpolation on SSP grid.

    Parameters
    ----------
    model : SEDModel
        The model instance.
    log_z_per_age : ndarray, shape (n_age,)
        log10(Z) absolute metallicity at each SSP age bin [dimensionless].

    Returns
    -------
    ndarray, shape (n_age, n_wave)
        SSP flux with age-dependent metallicity [erg/s/Hz].

    Notes
    -----
    **JIT-compatible**: yes — uses smooth or nearest-neighbor interpolation.
    """
    flux = ssp_flux if ssp_flux is not None else model.ssp_data.ssp_flux
    lgmet = ssp_lgmet if ssp_lgmet is not None else model.ssp_data.ssp_lgmet
    if model._met_interp == "smooth":
        return interpolate_metallicity_smooth_evolving(
            flux,
            lgmet,
            log_z_per_age,
            model._lgmet_scatter,
        )
    return interpolate_metallicity_evolving(flux, lgmet, log_z_per_age)
