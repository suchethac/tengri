# SPDX-License-Identifier: BSD-3-Clause
"""SKIRTOR-averaged (mean_1p) clumpy torus library (Stalevski et al. 2016).

One-parameter torus library interpolated differentiably in
``agn_incl_skirtor`` (inclination, degrees) over a precomputed HDF5 grid
derived from AGNfitter-rX's ``SKIRTOR_mean_1p.pickle``.

This is the coarsest of the three SKIRTOR-AGNfitter reductions tengri ships:
:mod:`tengri.components.agn.skirtor_agnfitter` (``SKIRTOR_mean_3p``, three
axes) is the AGNfitter-faithful default; :mod:`tengri.components.agn.skirtor_agnfitter_2p`
(``SKIRTOR_mean_2p``) adds the opening-angle axis; this module averages over
both opening angle and optical depth, retaining only inclination. All three
share the same inclination axis convention and extent (0-90 deg).

Axis semantics
--------------
``agn_incl_skirtor``
    Inclination angle measured from the pole [deg]. Range: 0-90 deg.
    0 = face-on, 90 = edge-on.

Runtime normalization
---------------------
Template is shape-only. At runtime the module divides by the trapezoidal
integral over frequency and multiplies by ``L_bol * agn_torus_frac``,
mirroring :mod:`tengri.components.agn.skirtor_agnfitter`.

References
----------
.. [1] M. Stalevski et al., "3D radiative transfer modeling of the dusty
   torus around AGN, the influence of clumping," MNRAS, 420, 2756 (2012).
   arXiv:1109.1286. https://doi.org/10.1111/j.1365-2966.2011.19775.x
.. [2] M. Stalevski, C. Ricci, Y. Ueda, P. Lira, J. Fritz, and M. Baes,
   "The dust covering factor in active galactic nuclei," MNRAS, 458,
   2288 (2016). arXiv:1602.06954. bibcode:2016MNRAS.458.2288S.
   https://doi.org/10.1093/mnras/stw444
.. [3] L. N. Martinez-Ramirez, et al., "AGNFITTER-RX: Modeling the
   radio-to-X-ray spectral energy distributions of AGNs," A&A 688, A46
   (2024). arXiv:2405.12111.
"""

from __future__ import annotations

import functools
from collections.abc import Callable

import jax.numpy as jnp
import numpy as np

from tengri.components.agn._params import DEFAULT_AGN_LOG_LBOL
from tengri.components.agn._template_grid import TorusTemplateGrid, torus_lnu_from_grid

__all__ = [
    "create_skirtor_agnfitter_1p_from_grid",
    "skirtor_agnfitter_1p_sed",
]


def _load_skirtor_agnfitter_1p_arrays(grid_path: str) -> dict:
    """Load raw numpy arrays from the ``SKIRTOR_mean_1p`` grid HDF5.

    Parameters
    ----------
    grid_path : str
        Path to ``skirtor_mean1p_torus_grid.h5``.

    Returns
    -------
    dict
        Keys ``incl_axis``, ``wavelength``, and ``template``.

    Notes
    -----
    **JIT-compatible**: no, performs HDF5 I/O at grid-load time.
    """
    import h5py

    with h5py.File(grid_path, "r") as f:
        g = f["skirtor_mean1p"]
        return {
            "incl_axis": np.asarray(g["incl_axis"][:], dtype=np.float64),
            "wavelength": np.asarray(g["wavelength"][:], dtype=np.float64),
            "template": np.asarray(g["template"][:], dtype=np.float64),
        }


@functools.cache
def load_skirtor_agnfitter_1p_grid(grid_path: str) -> TorusTemplateGrid:
    """Load a ``SKIRTOR_mean_1p`` grid HDF5 into a :class:`TorusTemplateGrid` pytree.

    Parameters
    ----------
    grid_path : str
        Path to ``skirtor_mean1p_torus_grid.h5``.

    Returns
    -------
    TorusTemplateGrid
        Template arrays with a single ``incl`` axis (degrees), as numpy
        arrays.

    Raises
    ------
    FileNotFoundError
        If ``grid_path`` does not exist.
    KeyError
        If the grid file is missing the expected ``/skirtor_mean1p``
        datasets.

    Notes
    -----
    **JIT-compatible**: no, performs HDF5 I/O. Call outside the trace and
    pass the result in as an argument.
    """
    raw = _load_skirtor_agnfitter_1p_arrays(grid_path)
    return TorusTemplateGrid(
        template=np.asarray(raw["template"]),
        axes=(np.asarray(raw["incl_axis"]),),
        wave_grid=np.asarray(raw["wavelength"]),
    )


def skirtor_agnfitter_1p_sed_from_grid(
    grid: TorusTemplateGrid,
    wavelength: jnp.ndarray,
    agn_log_lbol: float = DEFAULT_AGN_LOG_LBOL,
    agn_incl_skirtor: float = 30.0,
    agn_torus_frac: float = 0.5,
    **_kwargs,
) -> jnp.ndarray:
    r"""``SKIRTOR_mean_1p`` torus SED at a single inclination.

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Rest-frame wavelength grid. [Å]
    agn_log_lbol : float, optional
        ``log10(L_bol / L_sun)``. Default 10.0.
    agn_incl_skirtor : float, optional
        Inclination angle [deg]. Default 30.0.
    agn_torus_frac : float, optional
        Fraction of L_bol reprocessed by the torus. Default 0.5.

    Returns
    -------
    ndarray, shape (n_wave,)
        Spectral luminosity density. [erg/s/Hz]

    Notes
    -----
    .. math::

        L_\nu(\lambda) = L_{\rm bol}\,f_{\rm torus}\,
                         \frac{T(\lambda;\,i)}{\int T(\nu;\,i)\,\mathrm{d}\nu}

    where :math:`i` is the inclination angle.

    **JIT-compatible**: yes.

    **Approximation**: the grid is drawn from the ``SKIRTOR_mean_1p``
    library (Stalevski et al. 2016 [1]_, [2]_) as packaged by
    AGNfitter-rX [3]_, which averages out opening angle, clumpiness
    (p, q), and optical depth of the full SKIRTOR parameter space.
    """
    return torus_lnu_from_grid(
        grid,
        wavelength,
        (agn_incl_skirtor,),
        agn_log_lbol=agn_log_lbol,
        agn_torus_frac=agn_torus_frac,
    )


def create_skirtor_agnfitter_1p_from_grid(grid_path: str) -> Callable:
    """Load a ``SKIRTOR_mean_1p`` grid and bind it to the SED evaluator.

    Parameters
    ----------
    grid_path : str
        Path to ``skirtor_mean1p_torus_grid.h5``.

    Returns
    -------
    callable
    """
    return functools.partial(
        skirtor_agnfitter_1p_sed_from_grid, load_skirtor_agnfitter_1p_grid(grid_path)
    )


_GRID_SEARCH_PATHS: tuple[str, ...] = (
    "data/skirtor_mean1p_torus_grid.h5",
    "skirtor_mean1p_torus_grid.h5",
)

_NOT_FOUND_MSG = (
    "SKIRTOR_mean_1p torus grid not found. Build it with: "
    "python scripts/build_agnfitter_torus_reductions.py --which skirtor_mean1p"
)


def _find_skirtor_agnfitter_1p_grid() -> str:
    from tengri._data_setup import require_data

    return require_data(_GRID_SEARCH_PATHS[-1], _NOT_FOUND_MSG)


def load_skirtor_agnfitter_1p_default_grid() -> TorusTemplateGrid:
    """Load the packaged ``SKIRTOR_mean_1p`` grid pytree (discovery + cache).

    This is the ``template_loader`` the torus block registers.

    Returns
    -------
    TorusTemplateGrid

    Raises
    ------
    FileNotFoundError
        If no ``SKIRTOR_mean_1p`` grid HDF5 is present on disk.
    """
    return load_skirtor_agnfitter_1p_grid(_find_skirtor_agnfitter_1p_grid())


@functools.cache
def _load_skirtor_agnfitter_1p_default() -> Callable:
    return create_skirtor_agnfitter_1p_from_grid(_find_skirtor_agnfitter_1p_grid())


def skirtor_agnfitter_1p_sed(
    *args, _template: TorusTemplateGrid | None = None, **kwargs
) -> jnp.ndarray:
    """``SKIRTOR_mean_1p`` torus (auto-loaded from the packaged HDF5 grid).

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Rest-frame wavelength. [Å]
    agn_log_lbol : float, optional
        ``log10(L_bol / L_sun)``. Default 11.0.
    agn_incl_skirtor : float, optional
        Inclination angle [deg]. Default 30.0.
    agn_torus_frac : float, optional
        Torus reprocessing fraction. Default 0.5.
    **kwargs
        Accepted and ignored for unified-dispatch compatibility.

    Returns
    -------
    ndarray, shape (n_wave,)
        Torus SED [erg/s/Hz].

    Notes
    -----
    This function auto-discovers the grid file from the package data
    directory or the current working directory. To use a non-standard grid
    location, call :func:`create_skirtor_agnfitter_1p_from_grid` directly.
    """
    if _template is not None:
        return skirtor_agnfitter_1p_sed_from_grid(_template, *args, **kwargs)
    fn = _load_skirtor_agnfitter_1p_default()
    return fn(*args, **kwargs)
