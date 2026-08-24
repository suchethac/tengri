# SPDX-License-Identifier: BSD-3-Clause
"""Survey-published zero-point corrections for photometric fluxes.

Each entry: (survey, release, filter) -> magnitude offset (mag) and
fractional systematic (dimensionless). Applied as:
    flux_corrected = flux * 10**(-0.4 * mag_offset)
    err_corrected = sqrt(err**2 + (frac_sys * flux_corrected)**2)

NOTE: All numerical values in this registry are placeholders pending human
verification against the published survey calibration notes. See the source
string in each entry for verification status.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass(frozen=True)
class ZeropointEntry:
    """Zero-point correction for a single filter in a survey/release.

    Parameters
    ----------
    survey: str
        Survey name (e.g., "JADES", "CEERS", "COSMOS-Web").
    release: str
        Data release identifier (e.g., "DR5", "v1").
    filter_name: str
        Filter identifier (e.g., "F150W", "F277W").
    mag_offset: float, optional
        AB magnitude offset to ADD to the data. Default 0.0.
    fractional_sys_err: float, optional
        Fractional multiplicative systematic error floor.
        Added in quadrature to reported flux uncertainty. Default 0.0.
    source: str, optional
        Citation string or URL describing the origin of these values.
        Default "".

    Notes
    -----
    The `frozen=True` parameter makes this dataclass immutable, preventing
    accidental mutations of registry entries.
    """

    survey: str
    release: str
    filter_name: str
    mag_offset: float = 0.0
    fractional_sys_err: float = 0.0
    source: str = ""


# ─────────────────────────────────────────────────────────────────────────
# Zero-point registry (placeholder values, all pending verification)
# ─────────────────────────────────────────────────────────────────────────

ZEROPOINT_REGISTRY: list[ZeropointEntry] = [
    # JADES DR5 NIRCam broadband filters
    # All values placeholder: verify against JADES DR5 photometry notes
    ZeropointEntry(
        survey="JADES",
        release="DR5",
        filter_name="F090W",
        mag_offset=0.0,
        fractional_sys_err=0.02,
        source="placeholder (needs JADES DR5 verification)",
    ),
    ZeropointEntry(
        survey="JADES",
        release="DR5",
        filter_name="F115W",
        mag_offset=0.0,
        fractional_sys_err=0.02,
        source="placeholder (needs JADES DR5 verification)",
    ),
    ZeropointEntry(
        survey="JADES",
        release="DR5",
        filter_name="F150W",
        mag_offset=0.0,
        fractional_sys_err=0.02,
        source="placeholder (needs JADES DR5 verification)",
    ),
    ZeropointEntry(
        survey="JADES",
        release="DR5",
        filter_name="F200W",
        mag_offset=0.0,
        fractional_sys_err=0.02,
        source="placeholder (needs JADES DR5 verification)",
    ),
    ZeropointEntry(
        survey="JADES",
        release="DR5",
        filter_name="F277W",
        mag_offset=0.0,
        fractional_sys_err=0.02,
        source="placeholder (needs JADES DR5 verification)",
    ),
    ZeropointEntry(
        survey="JADES",
        release="DR5",
        filter_name="F356W",
        mag_offset=0.0,
        fractional_sys_err=0.02,
        source="placeholder (needs JADES DR5 verification)",
    ),
    ZeropointEntry(
        survey="JADES",
        release="DR5",
        filter_name="F410M",
        mag_offset=0.0,
        fractional_sys_err=0.02,
        source="placeholder (needs JADES DR5 verification)",
    ),
    ZeropointEntry(
        survey="JADES",
        release="DR5",
        filter_name="F444W",
        mag_offset=0.0,
        fractional_sys_err=0.02,
        source="placeholder (needs JADES DR5 verification)",
    ),
    # CEERS v1 NIRCam medium-band filters
    # Higher systematic floor due to medium-band calibration uncertainties
    ZeropointEntry(
        survey="CEERS",
        release="v1",
        filter_name="F277W",
        mag_offset=0.0,
        fractional_sys_err=0.05,
        source="placeholder (CEERS medium-band systematics, not yet verified)",
    ),
    ZeropointEntry(
        survey="CEERS",
        release="v1",
        filter_name="F410M",
        mag_offset=0.0,
        fractional_sys_err=0.05,
        source="placeholder (CEERS medium-band systematics, not yet verified)",
    ),
    ZeropointEntry(
        survey="CEERS",
        release="v1",
        filter_name="F430M",
        mag_offset=0.0,
        fractional_sys_err=0.05,
        source="placeholder (CEERS medium-band systematics, not yet verified)",
    ),
    ZeropointEntry(
        survey="CEERS",
        release="v1",
        filter_name="F460M",
        mag_offset=0.0,
        fractional_sys_err=0.05,
        source="placeholder (CEERS medium-band systematics, not yet verified)",
    ),
    ZeropointEntry(
        survey="CEERS",
        release="v1",
        filter_name="F470M",
        mag_offset=0.0,
        fractional_sys_err=0.05,
        source="placeholder (CEERS medium-band systematics, not yet verified)",
    ),
    # CEERS v1 NIRCam broadband filters
    ZeropointEntry(
        survey="CEERS",
        release="v1",
        filter_name="F115W",
        mag_offset=0.0,
        fractional_sys_err=0.02,
        source="placeholder (CEERS broadband systematics, not yet verified)",
    ),
    ZeropointEntry(
        survey="CEERS",
        release="v1",
        filter_name="F150W",
        mag_offset=0.0,
        fractional_sys_err=0.02,
        source="placeholder (CEERS broadband systematics, not yet verified)",
    ),
    ZeropointEntry(
        survey="CEERS",
        release="v1",
        filter_name="F200W",
        mag_offset=0.0,
        fractional_sys_err=0.02,
        source="placeholder (CEERS broadband systematics, not yet verified)",
    ),
    ZeropointEntry(
        survey="CEERS",
        release="v1",
        filter_name="F356W",
        mag_offset=0.0,
        fractional_sys_err=0.02,
        source="placeholder (CEERS broadband systematics, not yet verified)",
    ),
    ZeropointEntry(
        survey="CEERS",
        release="v1",
        filter_name="F444W",
        mag_offset=0.0,
        fractional_sys_err=0.02,
        source="placeholder (CEERS broadband systematics, not yet verified)",
    ),
    # COSMOS-Web DR0 placeholder entries
    ZeropointEntry(
        survey="COSMOS-Web",
        release="DR0",
        filter_name="F115W",
        mag_offset=0.0,
        fractional_sys_err=0.03,
        source="placeholder (COSMOS-Web DR0 systematics, not yet verified)",
    ),
    ZeropointEntry(
        survey="COSMOS-Web",
        release="DR0",
        filter_name="F150W",
        mag_offset=0.0,
        fractional_sys_err=0.03,
        source="placeholder (COSMOS-Web DR0 systematics, not yet verified)",
    ),
    ZeropointEntry(
        survey="COSMOS-Web",
        release="DR0",
        filter_name="F200W",
        mag_offset=0.0,
        fractional_sys_err=0.03,
        source="placeholder (COSMOS-Web DR0 systematics, not yet verified)",
    ),
    ZeropointEntry(
        survey="COSMOS-Web",
        release="DR0",
        filter_name="F277W",
        mag_offset=0.0,
        fractional_sys_err=0.03,
        source="placeholder (COSMOS-Web DR0 systematics, not yet verified)",
    ),
    ZeropointEntry(
        survey="COSMOS-Web",
        release="DR0",
        filter_name="F356W",
        mag_offset=0.0,
        fractional_sys_err=0.03,
        source="placeholder (COSMOS-Web DR0 systematics, not yet verified)",
    ),
    ZeropointEntry(
        survey="COSMOS-Web",
        release="DR0",
        filter_name="F444W",
        mag_offset=0.0,
        fractional_sys_err=0.03,
        source="placeholder (COSMOS-Web DR0 systematics, not yet verified)",
    ),
]


def lookup_zeropoints(survey: str, release: str, filters: list[str]) -> list[ZeropointEntry]:
    """Return zero-point entries for the given filters.

    Parameters
    ----------
    survey: str
        Survey name (e.g., "JADES").
    release: str
        Data release identifier (e.g., "DR5").
    filters: list of str
        Filter names to retrieve (e.g., ["F150W", "F277W"]).

    Returns
    -------
    list of ZeropointEntry
        Entries matching the survey, release, and filters, in the order
        requested.

    Raises
    ------
    KeyError
        If any requested filter is not found in the registry for the given
        survey and release.

    Notes
    -----
    **JIT-compatible**: no, pure Python, uses dict lookups.

    Examples
    --------
    >>> entries = lookup_zeropoints("JADES", "DR5", ["F150W", "F277W"])
    >>> print(entries[0].mag_offset)
    0.0
    """
    # Build a dict for fast lookup
    lookup_dict = {}
    for entry in ZEROPOINT_REGISTRY:
        if entry.survey == survey and entry.release == release:
            lookup_dict[entry.filter_name] = entry

    # Retrieve in order, raising KeyError if any filter is missing
    results = []
    for filt in filters:
        if filt not in lookup_dict:
            raise KeyError(
                f"Filter '{filt}' not found in registry for survey='{survey}', release='{release}'"
            )
        results.append(lookup_dict[filt])

    return results


def apply_zeropoints(
    flux: NDArray,
    flux_err: NDArray,
    entries: list[ZeropointEntry],
) -> tuple[NDArray, NDArray]:
    """Apply magnitude offsets and add systematic error floors.

    For each entry:
        1. Apply magnitude offset: flux_corrected = flux * 10^(-0.4 * mag_offset)
        2. Scale error by same factor: err_scaled = err * 10^(-0.4 * mag_offset)
        3. Add systematic floor in quadrature:
           err_corrected = sqrt(err_scaled^2 + (fractional_sys_err * flux_corrected)^2)

    Parameters
    ----------
    flux: ndarray, shape (n_filters,) or (n_sources, n_filters)
        Flux array in original units (e.g., microjansky).
    flux_err: ndarray, same shape as flux
        Flux uncertainty array.
    entries: list of ZeropointEntry
        Zero-point entries to apply. Must have length matching flux.shape[-1].

    Returns
    -------
    flux_corrected: ndarray, same shape as flux
        Flux after magnitude offset corrections.
    err_corrected: ndarray, same shape as flux_err
        Error after magnitude offset and systematic floor additions.

    Raises
    ------
    ValueError
        If the number of entries does not match the number of filters.

    Notes
    -----
    **JIT-compatible**: no, operates on a list of dataclass entries.

    Examples
    --------
    >>> entries = lookup_zeropoints("JADES", "DR5", ["F150W", "F277W"])
    >>> flux = np.array([100.0, 50.0])
    >>> err = np.array([5.0, 3.0])
    >>> flux_c, err_c = apply_zeropoints(flux, err, entries)
    """
    flux_arr = np.asarray(flux)
    err_arr = np.asarray(flux_err)

    if flux_arr.shape[-1] != len(entries):
        raise ValueError(
            f"Number of entries ({len(entries)}) does not match "
            f"number of filters ({flux_arr.shape[-1]})"
        )

    # Copy arrays to avoid in-place mutation
    flux_corrected = flux_arr.copy()
    err_corrected = err_arr.copy()

    # Apply each entry
    for i, entry in enumerate(entries):
        # Magnitude offset: flux_new = flux_old * 10^(-0.4 * mag_offset)
        mag_factor = 10.0 ** (-0.4 * entry.mag_offset)
        flux_corrected[..., i] = flux_arr[..., i] * mag_factor
        err_corrected[..., i] = err_arr[..., i] * mag_factor

        # Add systematic floor in quadrature
        if entry.fractional_sys_err > 0:
            sys_floor = entry.fractional_sys_err * flux_corrected[..., i]
            err_corrected[..., i] = np.sqrt(err_corrected[..., i] ** 2 + sys_floor**2)

    return flux_corrected, err_corrected
