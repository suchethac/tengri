# SPDX-License-Identifier: BSD-3-Clause
"""Nenkova et al. (2008) CLUMPY AGN torus from AGNfitter-rX.

One-parameter semi-empirical torus library keyed on inclination angle
(``agn_cos_inc``). The grid (9 bins from 10° to 90° inclination, averaged
over CLUMPY model parameters) is interpolated with node-exact monotone-cubic
(PCHIP) interpolation so gradients flow cleanly through ``agn_cos_inc``
during HMC / geoVI / MAP inference.

Grid provenance
---------------
Template data published with AGNfitter-rX (Martínez-Ramírez et al. 2024)
— see ``scripts/build_nk08_agnfitter_grid.py``. The AGNfitter-rX pickle
stores per-inclination ``log10(nu / Hz)`` and ``F_nu`` arrays; the build
script converts to ascending-wavelength [Å] and common-grid, then emits
``nenkova_agnfitter_torus_grid.h5``. This runtime module only consumes
the HDF5, not the pickle.

Runtime normalization
---------------------
The stored template is *shape-only*. At runtime,
:func:`create_nenkova_agnfitter_from_grid` divides by the trapezoidal
integral over frequency and multiplies by ``L_bol * agn_torus_frac``,
matching :mod:`tengri.components.agn.silva04`.

References
----------
.. [1] M. Nenkova et al., "Revisiting the AGN torus with MIDI and VISIR
   Herschel observations," ApJ 685, 160 (2008). arXiv:0806.1512.
   DOI: 10.1088/0004-637X/685/1/160.
.. [2] L. N. Martínez-Ramírez et al., "AGNfitter-rx: Modeling the
   radio-to-X-ray spectral energy distributions of AGNs," A&A 688, A46
   (2024). arXiv:2405.12111. DOI: 10.1051/0004-6361/202449329.
"""

from __future__ import annotations

import functools
from collections.abc import Callable

import jax.numpy as jnp
import numpy as np

from tengri.components.agn._params import DEFAULT_AGN_COS_INC, DEFAULT_AGN_LOG_LBOL
from tengri.components.agn._template_grid import TorusTemplateGrid, torus_lnu_from_grid

__all__ = [
    "create_nenkova_agnfitter_from_grid",
    "nenkova_agnfitter_sed",
]


def _load_nenkova_agnfitter_arrays(grid_path: str) -> dict:
    """Load raw numpy arrays from a Nenkova AGNfitter grid HDF5.

    Parameters
    ----------
    grid_path : str
        Path to ``nenkova_agnfitter_torus_grid.h5`` produced by
        ``scripts/build_nk08_agnfitter_grid.py``.

    Returns
    -------
    dict
        Keys ``incl_axis`` (n_incl,) in degrees, ``wavelength`` (n_wave,),
        and ``template`` (n_incl, n_wave).

    Notes
    -----
    **JIT-compatible**: no — performs HDF5 I/O at grid-load time.
    """
    import h5py

    with h5py.File(grid_path, "r") as f:
        g = f["nenkova_agnfitter"]
        return {
            "incl_axis": np.asarray(g["incl_axis"][:], dtype=np.float64),
            "wavelength": np.asarray(g["wavelength"][:], dtype=np.float64),
            "template": np.asarray(g["template"][:], dtype=np.float64),
        }


@functools.cache
def load_nenkova_agnfitter_grid(grid_path: str) -> TorusTemplateGrid:
    """Load a Nenkova AGNfitter grid HDF5 into a :class:`TorusTemplateGrid` pytree.

    Parameters
    ----------
    grid_path : str
        Path to ``nenkova_agnfitter_torus_grid.h5``.

    Returns
    -------
    TorusTemplateGrid
        Template arrays with a single ``cos_inc`` axis, as numpy arrays.

    Raises
    ------
    FileNotFoundError
        If ``grid_path`` does not exist.
    KeyError
        If the grid file is missing the expected ``/nenkova_agnfitter``
        datasets.

    Notes
    -----
    **JIT-compatible**: no — performs HDF5 I/O. Call outside the trace and
    pass the result in as an argument.
    """
    raw = _load_nenkova_agnfitter_arrays(grid_path)
    grid_np = np.asarray(raw["template"], dtype=np.float64)
    wave_np = np.asarray(raw["wavelength"], dtype=np.float64)
    incl_deg_np = np.asarray(raw["incl_axis"], dtype=np.float64)

    # Convert inclination (deg, ascending) to cos(incl), which is descending.
    # Reorder both axis and template so cos(incl) is ascending (consistent
    # with cat3d_wind pattern).
    cos_inc_axis_np = np.cos(np.deg2rad(incl_deg_np))
    order = np.argsort(cos_inc_axis_np)
    cos_inc_axis_np = cos_inc_axis_np[order]
    template_reordered_np = grid_np[order]
    return TorusTemplateGrid(
        template=template_reordered_np,
        axes=(cos_inc_axis_np,),
        wave_grid=wave_np,
    )


def nenkova_agnfitter_sed_from_grid(
    grid: TorusTemplateGrid,
    wavelength: jnp.ndarray,
    agn_log_lbol: float = DEFAULT_AGN_LOG_LBOL,
    agn_cos_inc: float = DEFAULT_AGN_COS_INC,
    agn_torus_frac: float = 0.5,
    **_kwargs,
) -> jnp.ndarray:
    r"""Nenkova+08 (AGNfitter-rX) torus SED at a single inclination.

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Rest-frame wavelength grid. [Å]
    agn_log_lbol : float, optional
        Bolometric luminosity, ``log10(L_bol / L_sun)``. Default 10.0.
    agn_cos_inc : float, optional
        Cosine of inclination (1 = face-on). Default 0.5.
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
                         \frac{T(\lambda,\,\cos i)}
                              {\int T(\nu,\,\cos i)\,\mathrm{d}\nu}

    where :math:`T` is the tabulated template (node-exact at grid points
    via monotone-cubic interpolation) and the integral is evaluated on
    the (sorted) frequency grid corresponding to ``wavelength``.

    **JIT-compatible**: yes.

    **Approximation**: the template is semi-empirical (Nenkova et al.
    2008 [1]_), based on CLUMPY radiative-transfer models averaged
    over secondary parameters. For studies requiring the full parameter
    space (N_0, σ, τ_V, Y, q), use the original CLUMPY library
    directly, not this averaged projection.
    """
    return torus_lnu_from_grid(
        grid,
        wavelength,
        (agn_cos_inc,),
        agn_log_lbol=agn_log_lbol,
        agn_torus_frac=agn_torus_frac,
    )


def create_nenkova_agnfitter_from_grid(grid_path: str) -> Callable:
    """Load a Nenkova AGNfitter grid and bind it to the SED evaluator.

    Retained for callers holding the historical closure API; new code should
    prefer :func:`load_nenkova_agnfitter_grid` plus
    :func:`nenkova_agnfitter_sed_from_grid`, which keeps the grid threadable.

    Parameters
    ----------
    grid_path : str
        Path to ``nenkova_agnfitter_torus_grid.h5``.

    Returns
    -------
    callable
    """
    return functools.partial(
        nenkova_agnfitter_sed_from_grid, load_nenkova_agnfitter_grid(grid_path)
    )


_GRID_SEARCH_PATHS: tuple[str, ...] = (
    "data/nenkova_agnfitter_torus_grid.h5",
    "nenkova_agnfitter_torus_grid.h5",
)

_NOT_FOUND_MSG = (
    "Nenkova AGNfitter torus grid not found. "
    "Build it with: python scripts/build_nk08_agnfitter_grid.py "
    "--input /tmp/AGNfitter-rX/models/TORUS/NK0_mean_1p.pickle"
)


def _find_nenkova_agnfitter_grid() -> str:
    from tengri._data_setup import require_data

    # _GRID_SEARCH_PATHS[-1] is the bare filename; require_data searches every
    # directory the old parents[4] walk reached, plus $TENGRI_DATA_DIR (#1431),
    # and re-raises _NOT_FOUND_MSG verbatim when the grid is absent.
    return require_data(_GRID_SEARCH_PATHS[-1], _NOT_FOUND_MSG)


def load_nenkova_agnfitter_default_grid() -> TorusTemplateGrid:
    """Load the packaged Nenkova AGNfitter grid pytree (discovery + cache).

    This is the ``template_loader`` the torus block registers.

    Returns
    -------
    TorusTemplateGrid

    Raises
    ------
    FileNotFoundError
        If no Nenkova AGNfitter grid HDF5 is present on disk.
    """
    return load_nenkova_agnfitter_grid(_find_nenkova_agnfitter_grid())


@functools.cache
def _load_nenkova_agnfitter_default() -> Callable:
    return create_nenkova_agnfitter_from_grid(_find_nenkova_agnfitter_grid())


def nenkova_agnfitter_sed(
    *args, _template: TorusTemplateGrid | None = None, **kwargs
) -> jnp.ndarray:
    """Nenkova+08 AGN torus (AGNfitter-rX, auto-loaded from tabulated templates).

    Wraps :func:`create_nenkova_agnfitter_from_grid` with on-disk grid discovery.

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Rest-frame wavelength grid. [Å]
    agn_log_lbol : float, optional
        ``log10(L_bol / L_sun)``. Default 10.0.
    agn_cos_inc : float, optional
        Cosine of inclination (1 = face-on). Default 0.5.
    agn_torus_frac : float, optional
        Torus reprocessing fraction. Default 0.5.
    **kwargs
        Accepted and ignored for unified-dispatch compatibility.

    Returns
    -------
    ndarray, shape (n_wave,)
        Spectral luminosity density. [erg/s/Hz]

    Raises
    ------
    FileNotFoundError
        If no Nenkova AGNfitter grid HDF5 is present on disk.

    References
    ----------
    .. [1] M. Nenkova et al., "Revisiting the AGN torus with MIDI and VISIR
       Herschel observations," ApJ 685, 160 (2008). arXiv:0806.1512.
       DOI: 10.1088/0004-637X/685/1/160.
    .. [2] L. N. Martínez-Ramírez et al., "AGNfitter-rx: Modeling the
       radio-to-X-ray spectral energy distributions of AGNs," A&A 688, A46
       (2024). arXiv:2405.12111. DOI: 10.1051/0004-6361/202449329.
    """
    if _template is not None:
        return nenkova_agnfitter_sed_from_grid(_template, *args, **kwargs)
    return _load_nenkova_agnfitter_default()(*args, **kwargs)
