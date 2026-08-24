# SPDX-License-Identifier: BSD-3-Clause
"""The one interpolation step behind every collapsed photometry lookup (#1431).

When a component's photometry grid is pre-integrated over filters, the axes the
user fixed are collapsed out and the survivors are interpolated at runtime.
Nine ``*_phot_collapsed`` closures did that, and each spelled the interpolation
itself slightly differently -- two kernels, and a scalar-template guard present
in four of them and absent in five.

Each closure keeps its own name, signature and scale factor. Those are not
boilerplate: the name reaches JAX profiles and tracebacks, ``agn_torus_frac``
is keyword-only in the calling contract of the five torus models, and the scale
genuinely differs (``10**agn_log_lbol`` for a disc, times ``agn_torus_frac``
for a torus, times ``_LSUN_ERG`` again for SKIRTOR_mean_3p, and ``L_absorbed``
for dust). Only the middle step is shared.

Notes
-----
The ``if not axes`` guard is an optimization, not a correctness guard. Both
:func:`interp_nd_triweight` and :func:`interp_nd_pchip` already return
``grid_phot`` unchanged for an empty axis tuple -- measured, which is what makes
it safe to give all nine the same treatment when only four had the guard. It is
kept because skipping the call is cheaper to trace than proving the kernel is a
no-op.
"""

from __future__ import annotations

from tengri.utils.grid_interp import interp_nd_pchip, interp_nd_triweight

__all__ = ["interp_collapsed"]

#: Interpolation kernels a pre-integrated photometry grid may be read with.
#: ``triweight`` needs bin edges; ``pchip`` works from the axis values alone.
KERNELS = ("triweight", "pchip")


def interp_collapsed(grid_phot, axes, free_axis_values, *, kernel, edges=None):
    """Read a pre-integrated photometry grid at the surviving axes' values.

    Parameters
    ----------
    grid_phot: ndarray, shape (*axis_lengths, n_filters)
        Filter-integrated grid, normalized per unit scale [erg/s/Hz].
    axes: tuple of array_like
        Node values of the axes that remain free. Empty when every axis was
        fixed, in which case ``grid_phot`` is already the answer.
    free_axis_values: tuple
        Runtime value for each surviving axis, in ``axes`` order.
    kernel: {'triweight', 'pchip'}
        Interpolation kernel. Keyword-only and required -- defaulting it would
        let a caller silently change the interpolation their component uses.
    edges: tuple of array_like, optional
        Bin edges per axis. Required by ``triweight``, unused by ``pchip``.

    Returns
    -------
    ndarray, shape (n_filters,)
        Interpolated photometry, still per unit scale. The caller multiplies in
        its own luminosity scale.

    Raises
    ------
    ValueError
        If ``kernel`` is unknown, or ``triweight`` is asked for without
        ``edges``. Both would otherwise fail far from the cause -- a bad kernel
        by silently using the other one, a missing ``edges`` inside the kernel.

    Notes
    -----
    JIT/grad/vmap safe: every branch here is on a Python-level structure
    (``axes`` length, ``kernel`` string) that is static at trace time.
    """
    if kernel not in KERNELS:
        raise ValueError(
            f"unknown interpolation kernel {kernel!r}; expected one of {list(KERNELS)}"
        )
    if not axes:
        return grid_phot
    if kernel == "pchip":
        return interp_nd_pchip(grid_phot, axes, tuple(free_axis_values))
    if edges is None:
        raise ValueError("the 'triweight' kernel needs bin edges; got edges=None")
    return interp_nd_triweight(grid_phot, axes, edges, tuple(free_axis_values))
