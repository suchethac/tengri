# SPDX-License-Identifier: BSD-3-Clause
"""CAT3D-Wind clumpy-disc-plus-polar-wind AGN torus, low-wind-fraction reduction
(Hönig & Kishimoto 2017).

Three-parameter torus library interpolated differentiably in
``(agn_cos_inc, agn_a_cat3d_lowfwd, agn_fwd_cat3d_lowfwd)`` over a
precomputed HDF5 grid derived from AGNfitter-rX's ``CAT3D_mean_3p.pickle``.

AGNfitter-rX's CAT3D_mean_3p pickle concatenates two disjoint
three-parameter sub-libraries at different rows: rows 0-209 (this module)
cover a low polar-wind mass fraction range (``fwd`` 0.15-0.75) at a wider
radial power-law-index range (``a`` -3 to -0.5); rows 210-377
(:mod:`tengri.components.agn.cat3d_wind`, the reduction AGNfitter-rX's own
fitter code selects by default) cover a high wind-fraction range (``fwd``
1.0-2.25) at a narrower radial-index range (``a`` -3 to -1.5). The two
sub-libraries do not overlap in ``(a, fwd)`` and are therefore vendored as
two independent grids rather than one, each a full Cartesian rectangle in
its own right (unlike the high-``fwd`` sub-library, this low-``fwd`` one is
a complete Cartesian product -- no nearest-neighbor cell filling is needed).

Axis semantics
--------------
``agn_cos_inc``
    Cosine of inclination (1 = face-on, 0 = edge-on). The grid's native
    ``incl_axis`` is stored in degrees; this module converts at load time,
    matching :mod:`tengri.components.agn.cat3d_wind`.
``agn_a_cat3d_lowfwd``
    Radial power-law index of the clumpy-cloud distribution
    (Hönig & Kishimoto 2017 parameter ``a``). Range: -3.0 to -0.5.
``agn_fwd_cat3d_lowfwd``
    Polar-wind mass fraction (``fwd``). Range: 0.15 to 0.75.

Runtime normalization
---------------------
Template is shape-only. At runtime the module divides by the trapezoidal
integral over frequency and multiplies by ``L_bol * agn_torus_frac``,
mirroring :mod:`tengri.components.agn.cat3d_wind`.

Interpolation
-------------
Node-exact monotone cubic (PCHIP, :func:`tengri.utils.grid_interp.interp_nd_pchip`),
matching :mod:`tengri.components.agn.cat3d_wind`.

References
----------
.. [1] S. F. Hönig & M. Kishimoto, "Dusty winds in active galactic nuclei: reconciling
   observations with models," ApJL 838,
   L20 (2017). arXiv:1702.08691.
.. [2] L. N. Martínez-Ramírez, G. Calistro Rivera, E. Lusso, et al.,
   "AGNfitter-rx: Modeling the radio-to-X-ray spectral energy
   distributions of AGNs," A&A 688, A46 (2024). arXiv:2405.12111.
   DOI: 10.1051/0004-6361/202449329.
"""

from __future__ import annotations

import functools
from collections.abc import Callable

import jax.numpy as jnp
import numpy as np

from tengri.components.agn._params import DEFAULT_AGN_COS_INC, DEFAULT_AGN_LOG_LBOL
from tengri.components.agn._template_grid import TorusTemplateGrid, torus_lnu_from_grid

__all__ = [
    "cat3d_wind_lowfwd_sed",
    "create_cat3d_wind_lowfwd_from_grid",
]

_DEFAULT_AGN_A_CAT3D_LOWFWD = -2.0
_DEFAULT_AGN_FWD_CAT3D_LOWFWD = 0.45


def _load_cat3d_lowfwd_arrays(grid_path: str) -> dict:
    """Load raw numpy arrays from the CAT3D-Wind low-fwd grid HDF5."""
    import h5py

    with h5py.File(grid_path, "r") as f:
        g = f["cat3d_wind_lowfwd"]
        return {
            "incl_axis": np.asarray(g["incl_axis"][:], dtype=np.float64),
            "a_axis": np.asarray(g["a_axis"][:], dtype=np.float64),
            "fwd_axis": np.asarray(g["fwd_axis"][:], dtype=np.float64),
            "wavelength": np.asarray(g["wavelength"][:], dtype=np.float64),
            "template": np.asarray(g["template"][:], dtype=np.float64),
        }


@functools.cache
def load_cat3d_wind_lowfwd_grid(grid_path: str) -> TorusTemplateGrid:
    """Load a CAT3D-Wind low-fwd grid HDF5 into a :class:`TorusTemplateGrid` pytree.

    Parameters
    ----------
    grid_path : str
        Path to ``cat3d_wind_lowfwd_torus_grid.h5``.

    Returns
    -------
    TorusTemplateGrid
        Template arrays with axes ``(cos_inc, a, f_wd)``, as numpy arrays.

    Raises
    ------
    FileNotFoundError
        If ``grid_path`` does not exist.
    KeyError
        If the grid file is missing any expected dataset under
        ``/cat3d_wind_lowfwd``.

    Notes
    -----
    **JIT-compatible**: no, performs HDF5 I/O. Call outside the trace and
    pass the result in as an argument.
    """
    raw = _load_cat3d_lowfwd_arrays(grid_path)

    # Convert native inclination axis (degrees, ascending) to cos(incl),
    # which is tengri's canonical inclination parameterization. The
    # template's leading axis must follow suit, which means reversing it
    # because cos(incl) is *descending* as incl ascends.
    incl_deg = raw["incl_axis"]
    cos_inc_axis = np.cos(np.deg2rad(incl_deg))
    order = np.argsort(cos_inc_axis)
    cos_inc_axis = cos_inc_axis[order]
    template_reordered = raw["template"][order]

    return TorusTemplateGrid(
        template=np.asarray(template_reordered),
        axes=(
            np.asarray(cos_inc_axis),
            np.asarray(raw["a_axis"]),
            np.asarray(raw["fwd_axis"]),
        ),
        wave_grid=np.asarray(raw["wavelength"]),
    )


def cat3d_wind_lowfwd_sed_from_grid(
    grid: TorusTemplateGrid,
    wavelength: jnp.ndarray,
    agn_log_lbol: float = DEFAULT_AGN_LOG_LBOL,
    agn_cos_inc: float = DEFAULT_AGN_COS_INC,
    agn_a_cat3d_lowfwd: float = _DEFAULT_AGN_A_CAT3D_LOWFWD,
    agn_fwd_cat3d_lowfwd: float = _DEFAULT_AGN_FWD_CAT3D_LOWFWD,
    agn_torus_frac: float = 0.5,
    **_kwargs,
) -> jnp.ndarray:
    r"""CAT3D-Wind low-fwd torus SED at a single ``(cos_inc, a, f_wd)``.

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Rest-frame wavelength grid. [Å]
    agn_log_lbol : float, optional
        ``log10(L_bol / L_sun)``. Default 10.0.
    agn_cos_inc : float, optional
        Cosine of inclination (1 = face-on). Default matches CIGALE i=30.
    agn_a_cat3d_lowfwd : float, optional
        Radial power-law index of the clumpy-cloud distribution. Default -2.0.
    agn_fwd_cat3d_lowfwd : float, optional
        Polar-wind mass fraction. Default 0.45.
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
                         \frac{T(\lambda;\,\cos i,\,a,\,f_{\rm wd})}
                              {\int T(\nu;\,\cos i,\,a,\,f_{\rm wd})
                               \,\mathrm{d}\nu}

    **JIT-compatible**: yes.

    **Approximation**: the grid is drawn from rows 0-209 of the
    ``CAT3D_mean_3p`` library (Hönig & Kishimoto 2017 [1]_) as packaged by
    AGNfitter-rX [2]_, which averages out secondary parameters (N_0, tau_V,
    sigma) of the full Hönig & Kishimoto 2017 parameter space.

    **Grid completeness**: unlike
    :mod:`tengri.components.agn.cat3d_wind`'s high-``fwd`` sub-library, this
    low-``fwd`` sub-library is a full Cartesian product of the three axes --
    no nearest-neighbor cell filling was needed at build time.
    """
    return torus_lnu_from_grid(
        grid,
        wavelength,
        (agn_cos_inc, agn_a_cat3d_lowfwd, agn_fwd_cat3d_lowfwd),
        agn_log_lbol=agn_log_lbol,
        agn_torus_frac=agn_torus_frac,
    )


def create_cat3d_wind_lowfwd_from_grid(grid_path: str) -> Callable:
    """Load a CAT3D-Wind low-fwd grid and bind it to the SED evaluator.

    Parameters
    ----------
    grid_path : str
        Path to ``cat3d_wind_lowfwd_torus_grid.h5``.

    Returns
    -------
    callable
    """
    return functools.partial(
        cat3d_wind_lowfwd_sed_from_grid, load_cat3d_wind_lowfwd_grid(grid_path)
    )


_GRID_SEARCH_PATHS: tuple[str, ...] = (
    "data/cat3d_wind_lowfwd_torus_grid.h5",
    "cat3d_wind_lowfwd_torus_grid.h5",
)

_NOT_FOUND_MSG = (
    "CAT3D-Wind low-fwd torus grid not found. Build it with: "
    "python scripts/build_agnfitter_torus_reductions.py --which cat3d_lowfwd"
)


def _find_cat3d_wind_lowfwd_grid() -> str:
    from tengri._data_setup import require_data

    return require_data(_GRID_SEARCH_PATHS[-1], _NOT_FOUND_MSG)


def load_cat3d_wind_lowfwd_default_grid() -> TorusTemplateGrid:
    """Load the packaged CAT3D-Wind low-fwd grid pytree (discovery + cache).

    This is the ``template_loader`` the torus block registers.

    Returns
    -------
    TorusTemplateGrid

    Raises
    ------
    FileNotFoundError
        If no CAT3D-Wind low-fwd grid HDF5 is present on disk.
    """
    return load_cat3d_wind_lowfwd_grid(_find_cat3d_wind_lowfwd_grid())


@functools.cache
def _load_cat3d_wind_lowfwd_default() -> Callable:
    return create_cat3d_wind_lowfwd_from_grid(_find_cat3d_wind_lowfwd_grid())


def cat3d_wind_lowfwd_sed(
    *args, _template: TorusTemplateGrid | None = None, **kwargs
) -> jnp.ndarray:
    """CAT3D-Wind low-fwd torus (auto-loaded from tabulated templates).

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Rest-frame wavelength grid. [Å]
    agn_log_lbol : float, optional
        ``log10(L_bol / L_sun)``. Default 10.0.
    agn_cos_inc : float, optional
        Cosine of inclination (1 = face-on). Default matches CIGALE i=30.
    agn_a_cat3d_lowfwd : float, optional
        Radial power-law index. Default -2.0.
    agn_fwd_cat3d_lowfwd : float, optional
        Polar-wind mass fraction. Default 0.45.
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
        If no CAT3D-Wind low-fwd grid HDF5 is present on disk.

    References
    ----------
    .. [1] S. F. Hönig & M. Kishimoto, ApJL 838, L20 (2017). arXiv:1702.08691.
    .. [2] L. N. Martínez-Ramírez et al., A&A 688, A46 (2024). arXiv:2405.12111.
    """
    if _template is not None:
        return cat3d_wind_lowfwd_sed_from_grid(_template, *args, **kwargs)
    return _load_cat3d_wind_lowfwd_default()(*args, **kwargs)
