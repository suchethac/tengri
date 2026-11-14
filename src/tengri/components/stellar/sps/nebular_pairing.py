# SPDX-License-Identifier: BSD-3-Clause
"""Pair a nebular backend with the SSP grid that does not already hold its emission (#3465).

The default grid (:data:`~tengri._data_setup.DEFAULT_SSP`) is nebular-baked (wNE): its
flux already carries the emission at a fixed logU. A nebular backend that adds emission
of its own (Cue, CLOUDY, CB19, or ``neb={'type': 'none'}``) therefore needs the bare
stellar twin of that grid, or the emission is stacked. This module decides which grid a
build uses. :func:`~tengri.forward.sed_model.SEDModel.build` calls it once, before the
spec is parsed, so every nebular choice sees the grid it was paired with.

The rule, in order:

* neb omitted, or ``neb={'type': 'ssp'}``: baked-in. The grid is kept, wNE or not.
* A default wNE grid (``default_selected``) with a twin in
  :data:`~tengri._data_setup._BARE_TWIN_OF_WNE`: any other backend, including
  ``neb={'type': 'none'}``, is built on the bare twin. The swap is logged at INFO.
* An explicitly named wNE grid: the grid is kept and each backend keeps its own
  refusal (Cue and CLOUDY refuse, CB19 refuses in its precompute). ``neb={'type': 'none'}``
  raises, because the grid carries emission that "none" cannot remove.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from tengri._data_setup import _BARE_TWIN_OF_WNE, _HOSTED_SSPS
from tengri.components.stellar.sps.dsps_wrapper import is_default_selected

logger = logging.getLogger(__name__)


def _neb_type(neb: Any) -> str | None:
    """The ``type`` of a ``neb=`` group, or ``None`` when it is omitted."""
    if neb is None:
        return None
    if isinstance(neb, dict):
        return neb.get("type")
    return getattr(neb, "type", None)


def _wne_short_name(ssp_data) -> str | None:
    """The wNE short name ``ssp_data`` was loaded from, when it is one with a bare twin."""
    for short in _BARE_TWIN_OF_WNE:
        if Path(_HOSTED_SSPS[short].filename).stem == ssp_data.source:
            return short
    return None


def _check_twin_matches(wne, bare) -> None:
    """Refuse a twin whose wavelength, metallicity or age axes differ from the grid it replaces."""
    same = (
        np.shape(wne.ssp_flux) == np.shape(bare.ssp_flux)
        and np.allclose(np.asarray(wne.ssp_wave), np.asarray(bare.ssp_wave), rtol=1e-12)
        and np.allclose(np.asarray(wne.ssp_lgmet), np.asarray(bare.ssp_lgmet), rtol=1e-12)
        and np.allclose(
            np.asarray(wne.ssp_lg_age_gyr), np.asarray(bare.ssp_lg_age_gyr), rtol=1e-12
        )
    )
    if not same:
        raise ValueError(
            f"the bare twin {bare.source!r} does not share the axes of {wne.source!r}; "
            "it cannot replace that grid under a nebular backend."
        )


def resolve_nebular_grid(ssp_data, neb):
    """Return the SSP grid a model built with this ``neb=`` group must use.

    Parameters
    ----------
    ssp_data : SSPData or None
        The grid the caller passed to :meth:`SEDModel.build`.
    neb : dict, object or None
        The ``neb=`` group as passed (``None`` when omitted).

    Returns
    -------
    SSPData or None
        ``ssp_data`` itself, or the bare twin loaded in its place.

    Raises
    ------
    ValueError
        For ``neb={'type': 'none'}`` on an explicitly named wNE grid, or when a twin
        does not match the grid it would replace.
    """
    if ssp_data is None or getattr(ssp_data, "nebular", "unknown") != "included":
        return ssp_data
    ntype = _neb_type(neb)
    if neb is None or ntype == "ssp":
        return ssp_data
    short = _wne_short_name(ssp_data)
    if not is_default_selected(ssp_data):
        if ntype == "none":
            twin_hint = (
                f"; use its bare twin, load_ssp({_BARE_TWIN_OF_WNE[short]!r}), "
                "for no nebular emission"
                if short in _BARE_TWIN_OF_WNE
                else ""
            )
            raise ValueError(
                f"neb={{'type': 'none'}} asks for no nebular emission, but the SSP grid "
                f"{ssp_data.source!r} includes it (nebular_included). The emission is "
                f"baked into the flux and cannot be switched off{twin_hint}."
            )
        return ssp_data
    if short is None:
        return ssp_data
    from tengri.components.stellar.sps.dsps_wrapper import load_ssp

    twin_name = _BARE_TWIN_OF_WNE[short]
    twin = load_ssp(twin_name, download=True)
    _check_twin_matches(ssp_data, twin)
    logger.info(
        "nebular %r needs the bare stellar grid: the default wNE grid %r is replaced by its "
        "bare twin %r (#3465)",
        ntype,
        ssp_data.source,
        twin_name,
    )
    return twin
