# SPDX-License-Identifier: BSD-3-Clause
"""CAT3D-Wind clumpy-disc-plus-polar-wind AGN torus (Hönig & Kishimoto 2017).

Three-parameter torus library interpolated differentiably in
``(agn_cos_inc, agn_a_cat3d, agn_fwd_cat3d)`` over a precomputed HDF5 grid
derived from AGNfitter-rX's ``CAT3D_mean_3p.pickle``.

Axis semantics
--------------
``agn_cos_inc``
    Cosine of inclination (1 = face-on, 0 = edge-on).  The grid's native
    ``incl_axis`` is stored in degrees; this module converts at load time
    so the runtime axis matches tengri's naming convention and SKIRTOR's
    ``agn_cos_inc`` usage.
``agn_a_cat3d``
    Radial power-law index of the clumpy-cloud distribution
    (Hönig & Kishimoto 2017 parameter ``a``).  Typical range −2.5 to −0.5;
    exact bounds follow the populated grid.
``agn_fwd_cat3d``
    Polar-wind mass fraction (``fwd``); the rest is in the mid-plane
    clumpy disc.

Runtime normalization
---------------------
Template is shape-only.  At runtime the module divides by the trapezoidal
integral over frequency and multiplies by ``L_bol * agn_torus_frac``,
mirroring :mod:`tengri.components.agn.skirtor` and
:mod:`tengri.components.agn.silva04`.

Grid completeness
-----------------
AGNfitter-rX's library is not a full Cartesian product of the three
axes; ``scripts/build_cat3d_wind_grid.py`` fills the missing cells with
their nearest-neighbor populated value at build time so the interpolant
has support over the full grid box.

Interpolation
-------------
Node-exact monotone cubic (PCHIP, :func:`tengri.utils.grid_interp.interp_nd_pchip`):
it reproduces every tabulated AGNfitter template at the grid nodes while
keeping C¹-continuous gradients. The C²-smooth triweight *smoother* used
elsewhere averages neighboring nodes, which smeared this torus's mid-IR
peak by tens of percent (median ~30%) — the same peak-smear that moved
:mod:`tengri.components.agn.slone_netzer` to node-exact interpolation.
Monotone cubic is shape-preserving, so it does not overshoot on the
nearest-neighbor-filled grid.

References
----------
.. [1] S. F. Hönig & M. Kishimoto, "The dusty heart of nearby active
   galaxies. II. From clumpy torus models to a unified model," ApJL 838,
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

from tengri._deprecated import deprecated_alias
from tengri.components.agn._params import DEFAULT_AGN_COS_INC, DEFAULT_AGN_LOG_LBOL
from tengri.components.agn._phys import (
    bolometric_integral_nu as _bolometric_integral_nu,
    wavelength_to_nu as _wavelength_to_nu,
)
from tengri.utils.grid_interp import interp_nd_pchip, resample_template
from tengri.utils.physics_constants import L_SUN as _LSUN_ERG

__all__ = [
    "cat3d_wind_analytic",
    "cat3d_wind_sed",
    "create_cat3d_wind_from_grid",
]


def _load_cat3d_arrays(grid_path: str) -> dict:
    """Load raw numpy arrays from the CAT3D-Wind grid HDF5."""
    import h5py

    with h5py.File(grid_path, "r") as f:
        g = f["cat3d_wind"]
        return {
            "incl_axis": np.asarray(g["incl_axis"][:], dtype=np.float64),
            "a_axis": np.asarray(g["a_axis"][:], dtype=np.float64),
            "fwd_axis": np.asarray(g["fwd_axis"][:], dtype=np.float64),
            "wavelength": np.asarray(g["wavelength"][:], dtype=np.float64),
            "template": np.asarray(g["template"][:], dtype=np.float64),
        }


def create_cat3d_wind_from_grid(grid_path: str) -> Callable:
    """Load CAT3D-Wind grid and return a JAX-native interpolation closure.

    Parameters
    ----------
    grid_path : str
        Path to ``cat3d_wind_torus_grid.h5``.

    Returns
    -------
    callable
        ``fn(wavelength, agn_log_lbol, agn_cos_inc, agn_a_cat3d,
        agn_fwd_cat3d, agn_torus_frac, **_) -> L_nu [erg/s/Hz]``.

    Raises
    ------
    FileNotFoundError
        If ``grid_path`` does not exist.
    KeyError
        If the grid file is missing any expected dataset under
        ``/cat3d_wind``.

    Notes
    -----
    **JIT-compatible**: yes — pure ``jnp`` and monotone-cubic interpolation.

    **Gradient-safe**: yes — node-exact PCHIP gives C¹-continuous gradients
    across the three parameter axes.
    """
    raw = _load_cat3d_arrays(grid_path)

    # Convert native inclination axis (degrees, ascending) to cos(incl),
    # which is tengri's canonical inclination parameterization.  The
    # template's leading axis must follow suit, which means reversing it
    # because cos(incl) is *descending* as incl ascends.
    incl_deg = raw["incl_axis"]
    cos_inc_axis = np.cos(np.deg2rad(incl_deg))
    order = np.argsort(cos_inc_axis)
    cos_inc_axis = cos_inc_axis[order]
    template_reordered = raw["template"][order]

    grid_jax = jnp.asarray(template_reordered)
    wave_grid = jnp.asarray(raw["wavelength"])
    axes = (
        jnp.asarray(cos_inc_axis),
        jnp.asarray(raw["a_axis"]),
        jnp.asarray(raw["fwd_axis"]),
    )

    def cat3d_wind_grid(
        wavelength: jnp.ndarray,
        agn_log_lbol: float = DEFAULT_AGN_LOG_LBOL,
        agn_cos_inc: float = DEFAULT_AGN_COS_INC,
        agn_a_cat3d: float = -2.0,
        agn_fwd_cat3d: float = 1.0,
        agn_torus_frac: float = 0.5,
        **_kwargs,
    ) -> jnp.ndarray:
        r"""CAT3D-Wind torus SED at a single ``(cos_inc, a, f_wd)``.

        Parameters
        ----------
        wavelength : array_like, shape (n_wave,)
            Rest-frame wavelength grid. [Å]
        agn_log_lbol : float, optional
            ``log10(L_bol / L_sun)``. Default 10.0.
        agn_cos_inc : float, optional
            Cosine of inclination (1 = face-on). Default 0.5.
        agn_a_cat3d : float, optional
            Radial power-law index of the clumpy-cloud distribution
            (Hönig & Kishimoto 2017 ``a``). Default −2.0.
        agn_fwd_cat3d : float, optional
            Polar-wind mass fraction. Default 1.0.
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

        **Approximation**: the grid is drawn from the ``CAT3D_mean_3p``
        library (Hönig & Kishimoto 2017 [1]_) as packaged by AGNfitter-rX
        [2]_, which averages out secondary parameters (N_0, τ_V, σ) of
        the full Hönig & Kishimoto 2017 parameter space. For torus
        studies where those parameters are scientifically important, use
        the full CAT3D-Wind library directly, not this three-parameter
        projection.

        **Grid completeness**: cells absent from the upstream library are
        filled with the nearest populated cell at build time (see
        ``scripts/build_cat3d_wind_grid.py``).
        """
        template = interp_nd_pchip(
            grid_jax,
            axes,
            (agn_cos_inc, agn_a_cat3d, agn_fwd_cat3d),
        )
        sed = resample_template(wavelength, wave_grid, template, left=0.0, right=0.0)
        nu = _wavelength_to_nu(wavelength)
        integral_safe = _bolometric_integral_nu(sed, nu, floor=1e-100)
        l_scale = 10.0**agn_log_lbol * _LSUN_ERG * agn_torus_frac
        return l_scale * sed / integral_safe

    return cat3d_wind_grid


_GRID_SEARCH_PATHS: tuple[str, ...] = (
    "data/cat3d_wind_torus_grid.h5",
    "cat3d_wind_torus_grid.h5",
)

_NOT_FOUND_MSG = (
    "CAT3D-Wind torus grid not found. Build it with: "
    "python scripts/build_cat3d_wind_grid.py "
    "--input /tmp/AGNfitter-rX/models/TORUS/CAT3D_mean_3p.pickle"
)


def _find_cat3d_grid() -> str:
    from tengri._data_setup import data_path

    # _GRID_SEARCH_PATHS[-1] is the bare filename; data_path searches every
    # directory the old parents[4] walk reached, plus $TENGRI_DATA_DIR (#1431).
    try:
        return str(data_path(_GRID_SEARCH_PATHS[-1]))
    except FileNotFoundError:
        raise FileNotFoundError(_NOT_FOUND_MSG) from None


@functools.cache
def _load_cat3d_default() -> Callable:
    return create_cat3d_wind_from_grid(_find_cat3d_grid())


def cat3d_wind_sed(*args, **kwargs) -> jnp.ndarray:
    """CAT3D-Wind torus (auto-loaded from the packaged HDF5 grid).

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Rest-frame wavelength. [Å]
    agn_log_lbol : float, optional
        ``log10(L_bol / L_sun)``. Default 10.0.
    agn_cos_inc : float, optional
        Cosine of inclination. Default 0.5.
    agn_a_cat3d : float, optional
        Radial power-law index. Default −2.0.
    agn_fwd_cat3d : float, optional
        Wind fraction. Default 1.0.
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
        If no CAT3D-Wind grid HDF5 is present on disk.
    """
    return _load_cat3d_default()(*args, **kwargs)


# Deprecated: "_analytic" was a misnomer — this is grid interpolation, not a
# closed-form model. Use cat3d_wind_sed. Alias removed in v1.0.
cat3d_wind_analytic = deprecated_alias(
    cat3d_wind_sed, old_name="cat3d_wind_analytic", new_name="cat3d_wind_sed"
)
