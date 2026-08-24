# SPDX-License-Identifier: BSD-3-Clause
"""Upper limit detection and conversion for photometric measurements."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def detect_upper_limits(
    flux: NDArray,
    flux_err: NDArray,
    sn_threshold: float = 1.0,
) -> NDArray:
    """Return boolean mask of non-detections (low signal-to-noise).

    A measurement is flagged as a potential upper limit if
    |flux| / flux_err < sn_threshold.

    Parameters
    ----------
    flux: ndarray
        Flux array in any shape.
    flux_err: ndarray
        Flux uncertainty array, same shape as `flux`.
    sn_threshold: float, optional
        Signal-to-noise ratio threshold for classification.
        Default 1.0 (conservative: typical observational S/N
        limits are ~2-3 sigma).

    Returns
    -------
    is_upper_limit: ndarray, dtype bool, same shape as flux
        True where |flux| / flux_err < sn_threshold.

    Notes
    -----
    **JIT-compatible**: no; pure Python, uses boolean indexing.

    The function does not modify the input arrays. The caller is
    responsible for interpreting the mask (e.g., applying an
    upper-limit likelihood, censoring the band, etc.).

    Examples
    --------
    >>> flux = np.array([10.0, 0.5, 100.0])
    >>> flux_err = np.array([2.0, 1.0, 5.0])
    >>> mask = detect_upper_limits(flux, flux_err, sn_threshold=2.0)
    >>> print(mask)
    [False  True False]
    """
    flux_arr = np.asarray(flux)
    err_arr = np.asarray(flux_err)

    # Avoid division by zero with a safe check
    sn = np.divide(
        np.abs(flux_arr),
        err_arr,
        out=np.zeros_like(flux_arr),
        where=(err_arr > 0),
    )

    return sn < sn_threshold


def sigma_upper_limit_from_flux(
    flux_err: NDArray,
    n_sigma: float = 3.0,
) -> NDArray:
    """Convert 1-sigma flux uncertainty into an N-sigma upper-limit flux.

    Given a flux error (1-sigma), compute the corresponding upper limit
    as n_sigma * flux_err.

    Parameters
    ----------
    flux_err: ndarray
        1-sigma flux uncertainty in any shape.
    n_sigma: float, optional
        Number of sigma for the upper limit. Default 3.0
        (typical astrophysical convention for a detection threshold).

    Returns
    -------
    upper_limit_flux: ndarray, same shape as flux_err
        Upper limit flux = n_sigma * flux_err.

    Notes
    -----
    **JIT-compatible**: no, pure numpy multiplication.

    This function is a convenience wrapper that assumes the input
    error is symmetric and Gaussian. For non-Gaussian or
    asymmetric error distributions, calibration against the
    specific survey's conventions is recommended.

    Examples
    --------
    >>> flux_err = np.array([1.0, 2.0, 0.5])
    >>> ul = sigma_upper_limit_from_flux(flux_err, n_sigma=3.0)
    >>> print(ul)
    [3.0 6.0 1.5]
    """
    err_arr = np.asarray(flux_err)
    return n_sigma * err_arr
