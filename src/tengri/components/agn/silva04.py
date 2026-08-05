# SPDX-License-Identifier: BSD-3-Clause
"""Silva, Maiolino & Granato (2004) smooth AGN torus.

One-parameter semi-empirical torus library keyed on hydrogen column density
(``log10(N_H / cm^-2)``).  The grid (5 bins in Silva+04) is interpolated with
a C²-continuous triweight kernel so gradients flow cleanly through
``agn_log_nh_silva`` during HMC / geoVI / MAP inference.

Grid provenance
---------------
Template data published with AGNfitter (Calistro Rivera et al. 2016) — see
``scripts/build_silva04_grid.py``.  The AGN-fitter pickle stores per-bin
``log10(nu)`` and ``F_nu`` arrays; the build script converts to
ascending-wavelength [Å] and common-grid, then emits ``silva04_torus_grid.h5``.
This runtime module only consumes the HDF5, not the pickle.

Runtime normalization
---------------------
The stored template is *shape-only*.  At runtime, :func:`create_silva04_from_grid`
divides by the trapezoidal integral over frequency and multiplies by
``L_bol * agn_torus_frac``, matching :mod:`tengri.components.agn.skirtor`.

References
----------
.. [1] L. Silva, R. Maiolino & G. L. Granato, "The nature of the Compton-thick
   AGN in NGC 1068 and implications for the cosmic X-ray background," MNRAS
   355, 973 (2004).  arXiv:astro-ph/0403425.  Citation details (volume, DOI)
   must be verified against the original paper before publication.
.. [2] G. Calistro Rivera et al., "AGNfitter: a Bayesian MCMC approach to
   fitting spectral energy distributions of AGNs," ApJ 833, 98 (2016).
   arXiv:1606.05648.
"""

from __future__ import annotations

import functools
from collections.abc import Callable

import jax.numpy as jnp
import numpy as np

from tengri._deprecated import deprecated_alias
from tengri.components.agn._params import DEFAULT_AGN_LOG_LBOL
from tengri.components.agn._phys import (
    bolometric_integral_nu as _bolometric_integral_nu,
    wavelength_to_nu as _wavelength_to_nu,
)
from tengri.utils.grid_interp import interp_nd_triweight, resample_template
from tengri.utils.physics_constants import L_SUN as _LSUN_ERG

__all__ = [
    "create_silva04_from_grid",
    "silva04_analytic",
    "silva04_sed",
]


def _load_silva04_arrays(grid_path: str) -> dict:
    """Load raw numpy arrays from a Silva+04 grid HDF5.

    Parameters
    ----------
    grid_path : str
        Path to ``silva04_torus_grid.h5`` produced by
        ``scripts/build_silva04_grid.py``.

    Returns
    -------
    dict
        Keys ``log_nh_axis`` (n_nh,), ``wavelength`` (n_wave,), and
        ``template`` (n_nh, n_wave).

    Notes
    -----
    **JIT-compatible**: no — performs HDF5 I/O at grid-load time.
    """
    import h5py

    with h5py.File(grid_path, "r") as f:
        g = f["silva04"]
        return {
            "log_nh_axis": np.asarray(g["log_nh_axis"][:], dtype=np.float64),
            "wavelength": np.asarray(g["wavelength"][:], dtype=np.float64),
            "template": np.asarray(g["template"][:], dtype=np.float64),
        }


def create_silva04_from_grid(grid_path: str) -> Callable:
    """Load Silva+04 grid and return a JAX-native interpolation closure.

    Parameters
    ----------
    grid_path : str
        Path to ``silva04_torus_grid.h5``.

    Returns
    -------
    callable
        Function ``fn(wavelength, agn_log_lbol, agn_log_nh_silva,
        agn_torus_frac, **_) -> L_nu [erg/s/Hz]``.

    Raises
    ------
    FileNotFoundError
        If ``grid_path`` does not exist.
    KeyError
        If the grid file is missing the expected ``/silva04`` datasets.

    Notes
    -----
    **JIT-compatible**: yes — the returned closure uses only ``jnp`` and
    triweight interpolation.

    **Gradient-safe**: yes — triweight kernel is C²-continuous in
    ``agn_log_nh_silva``.
    """
    # Keep the captured grid arrays as ``np.ndarray`` rather than
    # ``jnp.ndarray``. If this loader is first invoked inside a JIT trace
    # (e.g. via ``@functools.cache`` on ``_load_silva04_default``) and we
    # convert to JAX here, any ``jnp`` ops on those arrays — including
    # ``edges_for_grid`` — produce Tracers that the returned closure
    # captures. The cache then immortalizes a poisoned closure, leaking
    # tracers as ``UnexpectedTracerError`` on subsequent out-of-trace
    # calls. ``jnp.asarray`` of a numpy array inside the closure body is
    # safe in either context: a DeviceArray when called eagerly, a JIT
    # constant when called under trace.
    raw = _load_silva04_arrays(grid_path)
    grid_np = np.asarray(raw["template"], dtype=np.float64)
    wave_np = np.asarray(raw["wavelength"], dtype=np.float64)
    log_nh_np = np.asarray(raw["log_nh_axis"], dtype=np.float64)
    # ``edges_for_grid`` uses ``jnp.concatenate``; running it on a numpy
    # array still yields a JAX array, so compute the equivalent in pure
    # numpy to keep the precompute fully concrete.
    half_lo = (log_nh_np[1] - log_nh_np[0]) / 2.0
    half_hi = (log_nh_np[-1] - log_nh_np[-2]) / 2.0
    mid = 0.5 * (log_nh_np[1:] + log_nh_np[:-1])
    edges_np = np.concatenate([[log_nh_np[0] - half_lo], mid, [log_nh_np[-1] + half_hi]])

    def silva04_grid(
        wavelength: jnp.ndarray,
        agn_log_lbol: float = DEFAULT_AGN_LOG_LBOL,
        agn_log_nh_silva: float = 23.0,
        agn_torus_frac: float = 0.5,
        **_kwargs,
    ) -> jnp.ndarray:
        r"""Silva+04 torus SED at a single ``log10(N_H)``.

        Parameters
        ----------
        wavelength : array_like, shape (n_wave,)
            Rest-frame wavelength grid. [Å]
        agn_log_lbol : float, optional
            Bolometric luminosity, ``log10(L_bol / L_sun)``. Default 10.0.
        agn_log_nh_silva : float, optional
            Hydrogen column density, ``log10(N_H / cm^-2)``. Valid over the
            grid extent (Silva+04 bins typically 22–25). Default 23.0.
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
                             \frac{T(\lambda,\,\log N_H)}
                                  {\int T(\nu,\,\log N_H)\,\mathrm{d}\nu}

        where :math:`T` is the tabulated template and the integral is
        evaluated on the (sorted) frequency grid corresponding to
        ``wavelength``.

        **JIT-compatible**: yes.

        **Approximation**: the template is semi-empirical (Silva, Maiolino &
        Granato 2004 [1]_); it assumes smooth-dust geometry and is not a
        full 3D radiative-transfer solution.  For silicate-feature–level
        accuracy, use SKIRTOR ([2]_) instead.
        """
        grid_jax = jnp.asarray(grid_np)
        log_nh_axis = jnp.asarray(log_nh_np)
        wave_grid = jnp.asarray(wave_np)
        edges = (jnp.asarray(edges_np),)
        template = interp_nd_triweight(grid_jax, (log_nh_axis,), edges, (agn_log_nh_silva,))
        sed = resample_template(wavelength, wave_grid, template, left=0.0, right=0.0)
        nu = _wavelength_to_nu(wavelength)
        integral_safe = _bolometric_integral_nu(sed, nu, floor=1e-100)
        l_scale = 10.0**agn_log_lbol * _LSUN_ERG * agn_torus_frac
        return l_scale * sed / integral_safe

    return silva04_grid


_GRID_SEARCH_PATHS: tuple[str, ...] = (
    "data/silva04_torus_grid.h5",
    "silva04_torus_grid.h5",
)

_NOT_FOUND_MSG = (
    "Silva+04 torus grid not found. "
    "Build it with: python scripts/build_silva04_grid.py "
    "--input /tmp/AGNfitter/models/TORUS/S04.pickle"
)


def _find_silva04_grid() -> str:
    from tengri._data_setup import data_path

    # _GRID_SEARCH_PATHS[-1] is the bare filename; data_path searches every
    # directory the old parents[4] walk reached, plus $TENGRI_DATA_DIR (#1431).
    try:
        return str(data_path(_GRID_SEARCH_PATHS[-1]))
    except FileNotFoundError:
        raise FileNotFoundError(_NOT_FOUND_MSG) from None


@functools.cache
def _load_silva04_default() -> Callable:
    return create_silva04_from_grid(_find_silva04_grid())


def silva04_sed(*args, **kwargs) -> jnp.ndarray:
    """Silva+04 smooth AGN torus (auto-loaded from tabulated templates).

    Wraps :func:`create_silva04_from_grid` with on-disk grid discovery.

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Rest-frame wavelength grid. [Å]
    agn_log_lbol : float, optional
        ``log10(L_bol / L_sun)``. Default 10.0.
    agn_log_nh_silva : float, optional
        ``log10(N_H / cm^-2)``. Default 23.0.
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
        If no Silva+04 grid HDF5 is present on disk.
    """
    return _load_silva04_default()(*args, **kwargs)


# Deprecated: "_analytic" was a misnomer — this is grid interpolation, not a
# closed-form model. Use silva04_sed. Alias removed in v1.0.
silva04_analytic = deprecated_alias(
    silva04_sed, old_name="silva04_analytic", new_name="silva04_sed"
)
