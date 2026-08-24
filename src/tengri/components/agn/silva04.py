# SPDX-License-Identifier: BSD-3-Clause
"""Silva, Maiolino & Granato (2004) smooth AGN torus.

One-parameter semi-empirical torus library keyed on hydrogen column density
(``log10(N_H / cm^-2)``).  The grid (5 bins in Silva+04) is interpolated with
a C²-continuous triweight kernel so gradients flow cleanly through
``agn_log_nh_silva`` during HMC / geoVI / MAP inference.

Grid provenance
---------------
Template data published with AGNfitter (Calistro Rivera et al. 2016): see
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
from typing import NamedTuple

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
    "Silva04Grid",
    "create_silva04_from_grid",
    "load_silva04_grid",
    "silva04_analytic",
    "silva04_sed",
    "silva04_sed_from_grid",
]


class Silva04Grid(NamedTuple):
    """Silva+04 template arrays, as a JAX pytree.

    Carrying the grid as a pytree rather than closing over it is what makes
    threading possible: a closure's captured arrays are concrete at trace
    time and freeze into the graph as ``Constant`` ops, whereas a pytree can
    be passed as a traced **argument** of the jitted forward model.

    Attributes
    ----------
    template : ndarray, shape (n_nh, n_wave)
        Tabulated torus templates [arbitrary units; normalized on use].
    log_nh_axis : ndarray, shape (n_nh,)
        Grid axis, :math:`\\log_{10}(N_H / {\\rm cm}^{-2})`.
    edges : ndarray, shape (n_nh + 1,)
        Triweight-kernel bin edges derived from ``log_nh_axis``.
    wave_grid : ndarray, shape (n_wave,)
        Template rest-frame wavelength grid [Angstrom].
    """

    template: jnp.ndarray
    log_nh_axis: jnp.ndarray
    edges: jnp.ndarray
    wave_grid: jnp.ndarray


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
    **JIT-compatible**: no, performs HDF5 I/O at grid-load time.
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
    **JIT-compatible**: yes, the returned closure uses only ``jnp`` and
    triweight interpolation.

    **Gradient-safe**: yes, triweight kernel is C²-continuous in
    ``agn_log_nh_silva``.
    """
    # Keep the captured grid arrays as ``np.ndarray`` rather than
    # ``jnp.ndarray``. If this loader is first invoked inside a JIT trace
    # (e.g. via ``@functools.cache`` on ``_load_silva04_default``) and we
    # convert to JAX here, any ``jnp`` ops on those arrays: including
    # ``edges_for_grid``: produce Tracers that the returned closure
    # captures. The cache then immortalizes a poisoned closure, leaking
    # tracers as ``UnexpectedTracerError`` on subsequent out-of-trace
    # calls. ``jnp.asarray`` of a numpy array inside the closure body is
    # safe in either context: a DeviceArray when called eagerly, a JIT
    # constant when called under trace.
    return functools.partial(silva04_sed_from_grid, load_silva04_grid(grid_path))


@functools.cache
def load_silva04_grid(grid_path: str) -> Silva04Grid:
    """Load a Silva+04 grid HDF5 into a :class:`Silva04Grid` pytree.

    Parameters
    ----------
    grid_path : str
        Path to ``silva04_torus_grid.h5``.

    Returns
    -------
    Silva04Grid
        Template arrays as **numpy** arrays.

    Notes
    -----
    **JIT-compatible**: no, performs HDF5 I/O. Call it outside the trace
    and pass the result in as an argument.

    Leaves stay ``np.ndarray`` rather than ``jnp.ndarray`` on purpose. If
    this loader were first invoked inside a JIT trace and converted here,
    any ``jnp`` op on those arrays would produce Tracers that the
    ``functools.cache`` then immortalizes, leaking them as
    ``UnexpectedTracerError`` on later out-of-trace calls. ``jnp.asarray``
    at the point of use is safe in either context.
    """
    raw = _load_silva04_arrays(grid_path)
    log_nh_np = np.asarray(raw["log_nh_axis"], dtype=np.float64)
    # ``edges_for_grid`` uses ``jnp.concatenate``; running it on a numpy
    # array still yields a JAX array, so compute the equivalent in pure
    # numpy to keep the precompute fully concrete.
    half_lo = (log_nh_np[1] - log_nh_np[0]) / 2.0
    half_hi = (log_nh_np[-1] - log_nh_np[-2]) / 2.0
    mid = 0.5 * (log_nh_np[1:] + log_nh_np[:-1])
    edges_np = np.concatenate([[log_nh_np[0] - half_lo], mid, [log_nh_np[-1] + half_hi]])
    return Silva04Grid(
        template=np.asarray(raw["template"], dtype=np.float64),
        log_nh_axis=log_nh_np,
        edges=edges_np,
        wave_grid=np.asarray(raw["wavelength"], dtype=np.float64),
    )


def silva04_sed_from_grid(
    grid: Silva04Grid,
    wavelength: jnp.ndarray,
    agn_log_lbol: float = DEFAULT_AGN_LOG_LBOL,
    agn_log_nh_silva: float = 23.0,
    agn_torus_frac: float = 0.5,
    **_kwargs,
) -> jnp.ndarray:
    r"""Silva+04 torus SED at a single ``log10(N_H)``.

    Parameters
    ----------
    grid : Silva04Grid
        Template arrays. Passing these as an **argument** (rather than
        closing over them) is what lets the forward model thread the
        library through ``jax.jit`` as a ``Parameter`` instead of baking
        ~2 MB into the graph as ``Constant`` ops.
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

    **JIT-compatible**: yes. Differentiable in ``agn_log_nh_silva``
    (triweight kernel is C²-continuous).

    **Approximation**: the template is semi-empirical (Silva, Maiolino &
    Granato 2004 [1]_); it assumes smooth-dust geometry and is not a
    full 3D radiative-transfer solution.  For silicate-feature–level
    accuracy, use SKIRTOR ([2]_) instead.
    """
    template = interp_nd_triweight(
        jnp.asarray(grid.template),
        (jnp.asarray(grid.log_nh_axis),),
        (jnp.asarray(grid.edges),),
        (agn_log_nh_silva,),
    )
    sed = resample_template(wavelength, jnp.asarray(grid.wave_grid), template, left=0.0, right=0.0)
    nu = _wavelength_to_nu(wavelength)
    integral_safe = _bolometric_integral_nu(sed, nu, floor=1e-100)
    l_scale = 10.0**agn_log_lbol * _LSUN_ERG * agn_torus_frac
    return l_scale * sed / integral_safe


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
    from tengri._data_setup import require_data

    # _GRID_SEARCH_PATHS[-1] is the bare filename; require_data searches every
    # directory the old parents[4] walk reached, plus $TENGRI_DATA_DIR (#1431),
    # and re-raises _NOT_FOUND_MSG verbatim when the grid is absent.
    return require_data(_GRID_SEARCH_PATHS[-1], _NOT_FOUND_MSG)


@functools.cache
def _load_silva04_default() -> Callable:
    return create_silva04_from_grid(_find_silva04_grid())


def load_silva04_default_grid() -> Silva04Grid:
    """Load the packaged Silva+04 grid pytree (discovery + cache).

    This is the ``template_loader`` the torus block registers, so the
    forward model can hoist the library out of the trace.

    Returns
    -------
    Silva04Grid

    Raises
    ------
    FileNotFoundError
        If no Silva+04 grid HDF5 is present on disk.
    """
    return load_silva04_grid(_find_silva04_grid())


def silva04_sed(*args, _template: Silva04Grid | None = None, **kwargs) -> jnp.ndarray:
    """Silva+04 smooth AGN torus (auto-loaded from tabulated templates).

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
    _template : Silva04Grid, optional
        Pre-loaded grid, threaded in as a JIT argument by the forward
        model. When ``None`` (default) the packaged grid is loaded from
        disk and (if this call happens under trace) baked into the
        graph as constants.
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
    if _template is not None:
        return silva04_sed_from_grid(_template, *args, **kwargs)
    return _load_silva04_default()(*args, **kwargs)


# Deprecated: "_analytic" was a misnomer; this is grid interpolation, not a
# closed-form model. Use silva04_sed. Alias removed in v1.0.
silva04_analytic = deprecated_alias(
    silva04_sed, old_name="silva04_analytic", new_name="silva04_sed"
)
