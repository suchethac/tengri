# SPDX-License-Identifier: BSD-3-Clause
"""Nenkova et al. (2008) CLUMPY AGN torus, three-parameter AGNfitter-rX reduction.

Three-parameter torus library keyed on inclination angle (``agn_cos_inc``),
half-opening angle (``agn_oa_nenkova``), and equatorial optical depth
(``agn_tv_nenkova``), interpolated node-exactly (monotone cubic / PCHIP) so
gradients flow cleanly through every axis during HMC / geoVI / MAP
inference. This is the ``NK0_mean_3p`` reduction from AGNfitter-rX -- the
full three-parameter sub-library, one axis wider than
:mod:`tengri.components.agn.nenkova_agnfitter_2p` (``NK0_mean_2p``, which
shares the same inclination and opening-angle extents).

Grid provenance
---------------
Template data published with AGNfitter-rX (Martínez-Ramírez et al. 2024):
see ``scripts/build_agnfitter_torus_reductions.py``. The AGNfitter-rX pickle
(``NK0_mean_3p.pickle``, a pandas DataFrame) stores one shared
``log10(nu / Hz)`` axis and a per-(inclination, opening-angle, optical-depth)
``F_nu`` row; the build script converts to ascending wavelength [Å] at
native resolution (the upstream axis is common to every row, so no
resampling is performed) and reshapes into a rectangular
``(n_incl, n_oa, n_tv, n_wave)`` grid. This runtime module only consumes the
HDF5, not the pickle.

Axis semantics
--------------
``agn_cos_inc``
    Cosine of inclination (1 = face-on, 0 = edge-on); shared with every
    other cos-inclination-parametrized torus in tengri.
``agn_oa_nenkova``
    Half-opening angle of the CLUMPY torus [deg]. Range: 15-70 deg.
``agn_tv_nenkova``
    Equatorial optical depth. Range: 10-300 (``NK0_mean_3p`` grid extent).

Runtime normalization
---------------------
The stored template is *shape-only*. At runtime the module divides by the
trapezoidal integral over frequency and multiplies by
``L_bol * agn_torus_frac``, matching
:mod:`tengri.components.agn.nenkova_agnfitter`.

References
----------
.. [1] M. Nenkova, M. M. Sirocky, R. Nikutta, Z. Ivezic, and M. Elitzur,
   "AGN Dusty Tori. II. Observational Implications of Clumpiness," ApJ 685,
   160 (2008). doi:10.1086/590483. arXiv:0806.0512. bibcode:2008ApJ...685..160N.
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
    "create_nenkova_agnfitter_3p_from_grid",
    "nenkova_agnfitter_3p_sed",
]

_DEFAULT_AGN_OA_NENKOVA = 40.0
_DEFAULT_AGN_TV_NENKOVA = 60.0


def _load_nenkova_agnfitter_3p_arrays(grid_path: str) -> dict:
    """Load raw numpy arrays from a ``NK0_mean_3p`` grid HDF5.

    Parameters
    ----------
    grid_path : str
        Path to ``nenkova_agnfitter_3p_torus_grid.h5`` produced by
        ``scripts/build_agnfitter_torus_reductions.py``.

    Returns
    -------
    dict
        Keys ``incl_axis`` (n_incl,) in degrees, ``oa_axis`` (n_oa,) in
        degrees, ``tv_axis`` (n_tv,), ``wavelength`` (n_wave,), and
        ``template`` (n_incl, n_oa, n_tv, n_wave).

    Notes
    -----
    **JIT-compatible**: no, performs HDF5 I/O at grid-load time.
    """
    import h5py

    with h5py.File(grid_path, "r") as f:
        g = f["nenkova_agnfitter_3p"]
        return {
            "incl_axis": np.asarray(g["incl_axis"][:], dtype=np.float64),
            "oa_axis": np.asarray(g["oa_axis"][:], dtype=np.float64),
            "tv_axis": np.asarray(g["tv_axis"][:], dtype=np.float64),
            "wavelength": np.asarray(g["wavelength"][:], dtype=np.float64),
            "template": np.asarray(g["template"][:], dtype=np.float64),
        }


@functools.cache
def load_nenkova_agnfitter_3p_grid(grid_path: str) -> TorusTemplateGrid:
    """Load a ``NK0_mean_3p`` grid HDF5 into a :class:`TorusTemplateGrid` pytree.

    Parameters
    ----------
    grid_path : str
        Path to ``nenkova_agnfitter_3p_torus_grid.h5``.

    Returns
    -------
    TorusTemplateGrid
        Template arrays with axes ``(cos_inc, oa, tv)``, as numpy arrays.

    Raises
    ------
    FileNotFoundError
        If ``grid_path`` does not exist.
    KeyError
        If the grid file is missing the expected ``/nenkova_agnfitter_3p``
        datasets.

    Notes
    -----
    **JIT-compatible**: no, performs HDF5 I/O. Call outside the trace and
    pass the result in as an argument.
    """
    raw = _load_nenkova_agnfitter_3p_arrays(grid_path)
    grid_np = np.asarray(raw["template"], dtype=np.float64)
    wave_np = np.asarray(raw["wavelength"], dtype=np.float64)
    incl_deg_np = np.asarray(raw["incl_axis"], dtype=np.float64)
    oa_np = np.asarray(raw["oa_axis"], dtype=np.float64)
    tv_np = np.asarray(raw["tv_axis"], dtype=np.float64)

    # Convert inclination (deg, ascending) to cos(incl), which is descending;
    # reorder both the axis and the template's leading axis so cos(incl) is
    # ascending (matches nenkova_agnfitter.py / cat3d_wind.py convention).
    cos_inc_axis_np = np.cos(np.deg2rad(incl_deg_np))
    order = np.argsort(cos_inc_axis_np)
    cos_inc_axis_np = cos_inc_axis_np[order]
    template_reordered_np = grid_np[order]
    return TorusTemplateGrid(
        template=template_reordered_np,
        axes=(cos_inc_axis_np, oa_np, tv_np),
        wave_grid=wave_np,
    )


def nenkova_agnfitter_3p_sed_from_grid(
    grid: TorusTemplateGrid,
    wavelength: jnp.ndarray,
    agn_log_lbol: float = DEFAULT_AGN_LOG_LBOL,
    agn_cos_inc: float = DEFAULT_AGN_COS_INC,
    agn_oa_nenkova: float = _DEFAULT_AGN_OA_NENKOVA,
    agn_tv_nenkova: float = _DEFAULT_AGN_TV_NENKOVA,
    agn_torus_frac: float = 0.5,
    **_kwargs,
) -> jnp.ndarray:
    r"""``NK0_mean_3p`` (AGNfitter-rX) torus SED at a single (incl, oa, tv) node.

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Rest-frame wavelength grid. [Å]
    agn_log_lbol : float, optional
        Bolometric luminosity, ``log10(L_bol / L_sun)``. Default 10.0.
    agn_cos_inc : float, optional
        Cosine of inclination (1 = face-on). Default matches CIGALE i=30.
    agn_oa_nenkova : float, optional
        Half-opening angle [deg]. Default 40.0.
    agn_tv_nenkova : float, optional
        Equatorial optical depth. Default 60.0.
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
                         \frac{T(\lambda,\,\cos i,\,\theta_{\rm oa},\,\tau_V)}
                              {\int T(\nu,\,\cos i,\,\theta_{\rm oa},\,\tau_V)\,\mathrm{d}\nu}

    where :math:`T` is the tabulated template (node-exact at grid points
    via monotone-cubic interpolation) and the integral is evaluated on
    the (sorted) frequency grid corresponding to ``wavelength``.

    **JIT-compatible**: yes.

    **Approximation**: the template is semi-empirical (Nenkova et al.
    2008 [1]_), based on CLUMPY radiative-transfer models. This is the
    full three-parameter AGNfitter-rX sub-library (retains N_0-sigma
    averaging only); for the complete five-parameter CLUMPY space, use the
    original library directly.
    """
    return torus_lnu_from_grid(
        grid,
        wavelength,
        (agn_cos_inc, agn_oa_nenkova, agn_tv_nenkova),
        agn_log_lbol=agn_log_lbol,
        agn_torus_frac=agn_torus_frac,
    )


def create_nenkova_agnfitter_3p_from_grid(grid_path: str) -> Callable:
    """Load a ``NK0_mean_3p`` grid and bind it to the SED evaluator.

    Parameters
    ----------
    grid_path : str
        Path to ``nenkova_agnfitter_3p_torus_grid.h5``.

    Returns
    -------
    callable
    """
    return functools.partial(
        nenkova_agnfitter_3p_sed_from_grid, load_nenkova_agnfitter_3p_grid(grid_path)
    )


_GRID_SEARCH_PATHS: tuple[str, ...] = (
    "data/nenkova_agnfitter_3p_torus_grid.h5",
    "nenkova_agnfitter_3p_torus_grid.h5",
)

_NOT_FOUND_MSG = (
    "Nenkova AGNfitter 3p torus grid not found. "
    "Build it with: python scripts/build_agnfitter_torus_reductions.py --which nk08_3p"
)


def _find_nenkova_agnfitter_3p_grid() -> str:
    from tengri._data_setup import require_data

    return require_data(_GRID_SEARCH_PATHS[-1], _NOT_FOUND_MSG)


def load_nenkova_agnfitter_3p_default_grid() -> TorusTemplateGrid:
    """Load the packaged ``NK0_mean_3p`` grid pytree (discovery + cache).

    This is the ``template_loader`` the torus block registers.

    Returns
    -------
    TorusTemplateGrid

    Raises
    ------
    FileNotFoundError
        If no ``NK0_mean_3p`` grid HDF5 is present on disk.
    """
    return load_nenkova_agnfitter_3p_grid(_find_nenkova_agnfitter_3p_grid())


@functools.cache
def _load_nenkova_agnfitter_3p_default() -> Callable:
    return create_nenkova_agnfitter_3p_from_grid(_find_nenkova_agnfitter_3p_grid())


def nenkova_agnfitter_3p_sed(
    *args, _template: TorusTemplateGrid | None = None, **kwargs
) -> jnp.ndarray:
    """Nenkova+08 AGN torus, ``NK0_mean_3p`` reduction (auto-loaded from disk).

    Wraps :func:`create_nenkova_agnfitter_3p_from_grid` with on-disk grid
    discovery.

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Rest-frame wavelength grid. [Å]
    agn_log_lbol : float, optional
        ``log10(L_bol / L_sun)``. Default 10.0.
    agn_cos_inc : float, optional
        Cosine of inclination (1 = face-on). Default matches CIGALE i=30.
    agn_oa_nenkova : float, optional
        Half-opening angle [deg]. Default 40.0.
    agn_tv_nenkova : float, optional
        Equatorial optical depth. Default 60.0.
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
        If no ``NK0_mean_3p`` grid HDF5 is present on disk.

    References
    ----------
    .. [1] M. Nenkova, M. M. Sirocky, R. Nikutta, Z. Ivezic, and M. Elitzur,
       "AGN Dusty Tori. II. Observational Implications of Clumpiness," ApJ 685,
       160 (2008). doi:10.1086/590483. arXiv:0806.0512. bibcode:2008ApJ...685..160N.
    .. [2] L. N. Martínez-Ramírez et al., "AGNfitter-rx: Modeling the
       radio-to-X-ray spectral energy distributions of AGNs," A&A 688, A46
       (2024). arXiv:2405.12111.
    """
    if _template is not None:
        return nenkova_agnfitter_3p_sed_from_grid(_template, *args, **kwargs)
    return _load_nenkova_agnfitter_3p_default()(*args, **kwargs)
