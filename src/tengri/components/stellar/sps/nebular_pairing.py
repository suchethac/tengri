# SPDX-License-Identifier: BSD-3-Clause
"""Pair a nebular backend with the SSP grid that does not already hold its emission (#3465).

A nebular-baked (wNE) grid already carries nebular emission at a fixed logU. A nebular
backend that adds emission of its own (Cue, CLOUDY, CB19, or ``neb={'type': 'none'}``)
therefore needs the bare stellar twin of that grid, or the emission is stacked or cannot be
removed. :func:`~tengri.forward.sed_model.SEDModel.build` calls :func:`resolve_nebular_grid`
once, before the spec is parsed.

The rule:

* neb omitted, or ``neb={'type': 'ssp'}``: baked-in. The grid is kept, wNE or not.
* A wNE grid with a twin in the ``_BARE_TWIN_OF_WNE`` table in ``tengri._data_setup``: every other
  mode, including ``neb={'type': 'none'}``, is built on the bare twin (logged at INFO).
* A wNE grid without a twin: Cue and CLOUDY refuse (their own checks), CB19 refuses
  (:class:`~tengri.components.nebular.cloudy_cb19.CB19WNESSPError`), and
  ``neb={'type': 'none'}`` warns that the grid's emission stays in the flux.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from tengri._data_setup import _BARE_TWIN_OF_WNE, _HOSTED_SSPS

logger = logging.getLogger(__name__)


def _neb_type(neb: Any) -> str | None:
    """The ``type`` of a ``neb=`` group, or ``None`` when it is omitted."""
    if neb is None:
        return None
    if isinstance(neb, dict):
        return neb.get("type")
    return getattr(neb, "type", None)


def _wne_short_name(ssp_data) -> str | None:
    """The hosted wNE short name ``ssp_data`` was loaded from, or ``None``."""
    for short, entry in _HOSTED_SSPS.items():
        if Path(entry.filename).stem == ssp_data.source:
            return short
    return None


#: Relative tolerance of the axis check at float64.
_AXIS_RTOL_F64 = 1e-12


def _axis_rtol(*arrays) -> float:
    """The axis tolerance: ``_AXIS_RTOL_F64``, or 16 ulp of the coarsest precision given.

    A float32 build loads the bare twin as float32 while the wNE grid it replaces stays
    float64. The cast moves the wavelengths by about 6e-8 relative, which the float64
    tolerance refuses (#3465, test_chevallard13_plumbing::test_float32_matches_float64).
    """
    eps = max(float(np.finfo(np.asarray(a).dtype).eps) for a in arrays)
    return max(_AXIS_RTOL_F64, 16.0 * eps)


def _check_twin_matches(wne, bare) -> None:
    """Refuse a twin whose wavelength, metallicity or age axes differ from the grid it replaces."""
    rtol = _axis_rtol(
        wne.ssp_wave, bare.ssp_wave, wne.ssp_lgmet, bare.ssp_lgmet, wne.ssp_lg_age_gyr
    )
    same = (
        np.shape(wne.ssp_flux) == np.shape(bare.ssp_flux)
        and np.allclose(np.asarray(wne.ssp_wave), np.asarray(bare.ssp_wave), rtol=rtol)
        and np.allclose(np.asarray(wne.ssp_lgmet), np.asarray(bare.ssp_lgmet), rtol=rtol)
        and np.allclose(np.asarray(wne.ssp_lg_age_gyr), np.asarray(bare.ssp_lg_age_gyr), rtol=rtol)
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
        If a twin does not match the grid it would replace.
    """
    if ssp_data is None or getattr(ssp_data, "nebular", "unknown") != "included":
        return ssp_data
    ntype = _neb_type(neb)
    if neb is None or ntype == "ssp":
        return ssp_data
    short = _wne_short_name(ssp_data)
    if short is None or short not in _BARE_TWIN_OF_WNE:
        if ntype == "none":
            warnings.warn(
                f"neb type 'none' does not remove the nebular emission already in the SSP grid "
                f"{ssp_data.source!r} (nebular_included): this grid includes nebular emission, "
                "and the model keeps it. Use a bare grid for no nebular emission.",
                UserWarning,
                stacklevel=3,
            )
        return ssp_data
    from tengri.components.stellar.sps.dsps_wrapper import load_ssp

    twin_name = _BARE_TWIN_OF_WNE[short]
    twin = load_ssp(twin_name, download=True)
    _check_twin_matches(ssp_data, twin)
    logger.info(
        "nebular %r needs the bare stellar grid: the wNE grid %r is replaced by its bare twin "
        "%r (#3465)",
        ntype,
        ssp_data.source,
        twin_name,
    )
    return twin
