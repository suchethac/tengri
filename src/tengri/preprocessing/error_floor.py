# SPDX-License-Identifier: BSD-3-Clause
"""Systematic error floor application for photometric measurements."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def add_systematic_floor(
    flux: NDArray,
    flux_err: NDArray,
    fractional: float = 0.02,
) -> NDArray:
    """Add a multiplicative systematic error floor in quadrature.

    Computes err_total = sqrt(err^2 + (fractional * |flux|)^2), returning
    the combined error array. Does not modify inputs.

    Parameters
    ----------
    flux: ndarray
        Flux array in any shape. Can be negative (absolute value used).
    flux_err: ndarray
        Flux uncertainty array, same shape as `flux`.
    fractional: float, optional
        Fraction of flux magnitude to add as systematic floor.
        Default 0.02 (2%, a common conservative choice for e.g.
        photometric zero-point uncertainty).

    Returns
    -------
    err_total: ndarray, same shape as flux_err
        Combined statistical + systematic uncertainty.

    Notes
    -----
    **JIT-compatible**: no, uses np.sqrt which may not be JIT-friendly
    in all contexts. Pure-JAX version would use jax.numpy.sqrt.

    Examples
    --------
    >>> flux = np.array([100.0, 50.0, -30.0])
    >>> flux_err = np.array([5.0, 3.0, 2.0])
    >>> err_total = add_systematic_floor(flux, flux_err, fractional=0.02)
    >>> print(err_total)
    [5.19... 3.07... 2.6...]
    """
    flux_arr = np.asarray(flux)
    err_arr = np.asarray(flux_err)

    # Systematic floor: fractional * |flux|
    sys_floor = fractional * np.abs(flux_arr)

    # Combine in quadrature
    err_total = np.sqrt(err_arr**2 + sys_floor**2)

    return err_total
