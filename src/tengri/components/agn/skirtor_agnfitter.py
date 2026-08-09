# SPDX-License-Identifier: BSD-3-Clause
"""SKIRTOR-averaged (mean_3p) clumpy torus library (Stalevski et al. 2016).

Three-parameter torus library interpolated differentiably in
``(agn_oa_skirtor, agn_incl_skirtor, agn_tv_skirtor)`` over a precomputed
HDF5 grid derived from AGNfitter-rX's ``SKIRTOR_mean_3p.pickle``.

This module is distinct from :mod:`tengri.components.agn.skirtor`, which uses
tengri's full-grid SKIRTOR implementation. The SKIRTOR_mean_3p variant
shipped in AGNfitter-rX averages over the clumpiness parameters (p, q) and
radial index, retaining only the three-parameter sub-library with:

- half-opening angle (``agn_oa_skirtor``) [deg]
- inclination angle (``agn_incl_skirtor``) [deg]
- equatorial optical depth (``agn_tv_skirtor``)

This is the **AGNfitter-faithful** choice, where the default full-grid
``skirtor`` (X-CIGALE-faithful) differs in peak wavelength and other properties
(see issue #614 and #592).

Axis semantics
--------------
``agn_oa_skirtor``
    Half-opening angle of the torus [deg]. Range: 10–80 deg.
    This is the OpenAngle parameter of Stalevski et al. (2016).
``agn_incl_skirtor``
    Inclination angle measured from the pole [deg]. Range: 0–90 deg.
    0 = face-on, 90 = edge-on.
``agn_tv_skirtor``
    Equatorial optical depth at 9.7 µm (τ_9.7). Range: 3–11.
    Controls the torus optical thickness.

Runtime normalization
---------------------
Template is shape-only.  At runtime the module divides by the trapezoidal
integral over frequency and multiplies by ``L_bol * agn_torus_frac``,
mirroring :mod:`tengri.components.agn.skirtor` and
:mod:`tengri.components.agn.silva04`.

References
----------
.. [1] M. Stalevski et al., "3D radiative transfer modeling of the dusty
   torus around AGN — the influence of clumping," MNRAS, 420, 2756 (2012).
   arXiv:1109.1286. https://doi.org/10.1111/j.1365-2966.2011.19775.x
.. [2] M. Stalevski et al., "The dust covering factor in AGN — combining the
   IR torus emission with polar dust component," MNRAS, 458, 2288 (2016).
   arXiv:1602.01954. https://doi.org/10.1093/mnras/stw444
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
    "create_skirtor_agnfitter_from_grid",
    "skirtor_agnfitter_sed",
]


def _load_skirtor_agnfitter_arrays(grid_path: str) -> dict:
    """Load raw numpy arrays from the SKIRTOR_mean_3p grid HDF5.

    Parameters
    ----------
    grid_path : str
        Path to ``skirtor_mean3p_torus_grid.h5``.

    Returns
    -------
    dict
        Keys: ``oa_axis``, ``incl_axis``, ``tv_axis``, ``wavelength``,
        and ``template``.

    Notes
    -----
    **JIT-compatible**: no — performs HDF5 I/O at grid-load time.

    Templates may be stored as float32 or float64; they are cast to
    float64 for monotone-cubic interpolation.
    """
    import h5py

    with h5py.File(grid_path, "r") as f:
        g = f["skirtor_mean3p"]
        # Load template as-is (may be float32), then cast to float64 for interpolation
        template = np.asarray(g["template"][:], dtype=np.float64)
        return {
            "oa_axis": np.asarray(g["oa_axis"][:], dtype=np.float64),
            "incl_axis": np.asarray(g["incl_axis"][:], dtype=np.float64),
            "tv_axis": np.asarray(g["tv_axis"][:], dtype=np.float64),
            "wavelength": np.asarray(g["wavelength"][:], dtype=np.float64),
            "template": template,
        }


@functools.cache
def load_skirtor_agnfitter_grid(grid_path: str) -> TorusTemplateGrid:
    """Load a SKIRTOR_mean_3p grid HDF5 into a :class:`TorusTemplateGrid` pytree.

    Parameters
    ----------
    grid_path : str
        Path to ``skirtor_mean3p_torus_grid.h5``.

    Returns
    -------
    TorusTemplateGrid
        Template arrays with axes ``(oa, incl, tau_V)``, as numpy arrays.

    Raises
    ------
    FileNotFoundError
        If ``grid_path`` does not exist.
    KeyError
        If the grid file is missing any expected dataset under
        ``/skirtor_mean3p``.

    Notes
    -----
    **JIT-compatible**: no — performs HDF5 I/O. Call outside the trace and
    pass the result in as an argument.
    """
    raw = _load_skirtor_agnfitter_arrays(grid_path)
    return TorusTemplateGrid(
        template=np.asarray(raw["template"]),
        axes=(
            np.asarray(raw["oa_axis"]),
            np.asarray(raw["incl_axis"]),
            np.asarray(raw["tv_axis"]),
        ),
        wave_grid=np.asarray(raw["wavelength"]),
    )


def skirtor_agnfitter_sed_from_grid(
    grid: TorusTemplateGrid,
    wavelength: jnp.ndarray,
    agn_log_lbol: float = DEFAULT_AGN_LOG_LBOL,
    agn_oa_skirtor: float = 40.0,
    agn_incl_skirtor: float = 30.0,
    agn_tv_skirtor: float = 7.0,
    agn_torus_frac: float = 0.5,
    **_kwargs,
) -> jnp.ndarray:
    r"""SKIRTOR_mean_3p torus SED at a single (oa, incl, tv) node.

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Rest-frame wavelength grid. [Å]
    agn_log_lbol : float, optional
        ``log10(L_bol / L_sun)``. Default 11.0.
    agn_oa_skirtor : float, optional
        Half-opening angle [deg]. Default 40.0.
    agn_incl_skirtor : float, optional
        Inclination angle [deg]. Default 30.0.
    agn_tv_skirtor : float, optional
        Equatorial optical depth τ_9.7. Default 7.0.
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
                         \frac{T(\lambda;\,\theta,\,i,\,\tau_{9.7})}
                              {\int T(\nu;\,\theta,\,i,\,\tau_{9.7})
                               \,\mathrm{d}\nu}

    where θ is the half-opening angle, i is inclination, and τ_9.7 is
    the equatorial optical depth.

    **JIT-compatible**: yes.

    **Approximation**: the grid is drawn from the ``SKIRTOR_mean_3p``
    library (Stalevski et al. 2016 [1]_, [2]_) as packaged by
    AGNfitter-rX [3]_, which averages out secondary parameters
    (clumpiness p, q, radial index) of the full SKIRTOR parameter space.
    This averaged-grid variant differs from tengri's default full-grid
    SKIRTOR implementation (which follows X-CIGALE conventions). For
    AGNfitter-faithful torus modeling, use this component; for X-CIGALE
    faithful modeling, use the default SKIRTOR.
    """
    return torus_lnu_from_grid(
        grid,
        wavelength,
        (agn_oa_skirtor, agn_incl_skirtor, agn_tv_skirtor),
        agn_log_lbol=agn_log_lbol,
        agn_torus_frac=agn_torus_frac,
    )


def create_skirtor_agnfitter_from_grid(grid_path: str) -> Callable:
    """Load a SKIRTOR_mean_3p grid and bind it to the SED evaluator.

    Retained for callers holding the historical closure API; new code should
    prefer :func:`load_skirtor_agnfitter_grid` plus
    :func:`skirtor_agnfitter_sed_from_grid`, which keeps the grid threadable.

    Parameters
    ----------
    grid_path : str
        Path to ``skirtor_mean3p_torus_grid.h5``.

    Returns
    -------
    callable
    """
    return functools.partial(
        skirtor_agnfitter_sed_from_grid, load_skirtor_agnfitter_grid(grid_path)
    )


_GRID_SEARCH_PATHS: tuple[str, ...] = (
    "data/skirtor_mean3p_torus_grid.h5",
    "skirtor_mean3p_torus_grid.h5",
)

_NOT_FOUND_MSG = (
    "SKIRTOR_mean_3p torus grid not found. Build it with: "
    "python scripts/build_skirtor_mean3p_grid.py "
    "--input /tmp/AGNfitter-rX/models/TORUS/SKIRTOR_mean_3p.pickle"
)


def _find_skirtor_agnfitter_grid() -> str:
    from tengri._data_setup import data_path

    # _GRID_SEARCH_PATHS[-1] is the bare filename; data_path searches every
    # directory the old parents[4] walk reached, plus $TENGRI_DATA_DIR (#1431).
    try:
        return str(data_path(_GRID_SEARCH_PATHS[-1]))
    except FileNotFoundError:
        raise FileNotFoundError(_NOT_FOUND_MSG) from None


def load_skirtor_agnfitter_default_grid() -> TorusTemplateGrid:
    """Load the packaged SKIRTOR_mean_3p grid pytree (discovery + cache).

    This is the ``template_loader`` the torus block registers.

    Returns
    -------
    TorusTemplateGrid

    Raises
    ------
    FileNotFoundError
        If no SKIRTOR_mean_3p grid HDF5 is present on disk.
    """
    return load_skirtor_agnfitter_grid(_find_skirtor_agnfitter_grid())


@functools.cache
def _load_skirtor_agnfitter_default() -> Callable:
    return create_skirtor_agnfitter_from_grid(_find_skirtor_agnfitter_grid())


def skirtor_agnfitter_sed(
    *args, _template: TorusTemplateGrid | None = None, **kwargs
) -> jnp.ndarray:
    """SKIRTOR_mean_3p torus (auto-loaded from the packaged HDF5 grid).

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Rest-frame wavelength. [Å]
    agn_log_lbol : float, optional
        ``log10(L_bol / L_sun)``. Default 11.0.
    agn_oa_skirtor : float, optional
        Half-opening angle [deg]. Default 40.0.
    agn_incl_skirtor : float, optional
        Inclination angle [deg]. Default 30.0.
    agn_tv_skirtor : float, optional
        Equatorial optical depth τ_9.7. Default 7.0.
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
    location, call :func:`create_skirtor_agnfitter_from_grid` directly.
    """
    if _template is not None:
        return skirtor_agnfitter_sed_from_grid(_template, *args, **kwargs)
    fn = _load_skirtor_agnfitter_default()
    return fn(*args, **kwargs)
