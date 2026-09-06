# SPDX-License-Identifier: BSD-3-Clause
r"""AGNfitter-rX's Kubota & Done (2018) grid-tabulated accretion-disc library.

Two node-exact template grids, interpolated from HDF5 files built by
``scripts/build_kd18_grid.py`` from AGNfitter-rX's ``KD18.pickle`` /
``KD18_warmInd.pickle``:

- :func:`kd18_agnfitter_sed` -- 2-D grid over ``(log M_BH, log Mdot/Mdot_Edd)``.
- :func:`kd18_agnfitter_warmindex_sed` -- 3-D grid adding the warm
  Comptonization spectral index (AGNfitter-rX's ``warmIndex``).

A one-time measurement compared the two pickles at each candidate
``warmIndex`` value (matched node-by-node on ``(logBHmass, logEddra)``): even
the closest match disagreed by up to a factor of 1.27 at some wavelength, so
the two are vendored and interpolated as independent grids rather than one
grid at a fixed ``warmIndex`` (see ``scripts/build_kd18_grid.py`` module
docstring for the measurement).

Both templates already carry AGNfitter-rX's hot corona (the wavelength axis
extends to :math:`\lambda \approx 0.06` A, hard X-ray): the ``KD18`` and
``KD18_warmIndex`` branches of AGNfitter-rX never call ``add_xrays``.
tengri's own composable ``xray`` group must not be layered on top of either
block for the same reason (see
``check_disc_xray_double_count``).

Uses the same template library as AGNfitter-rX (Martinez-Ramirez et al. 2024
[2]_); validated against its output.

References
----------
.. [1] A. Kubota and C. Done, "A physical interpretation of the hard X-ray
   excess in low-luminosity AGN," MNRAS, 480, 1247 (2018).
   doi:10.1093/mnras/sty1890. arXiv:1804.02334.
.. [2] L. N. Martinez-Ramirez, et al., "AGNFITTER-RX: Modeling the
   radio-to-X-ray spectral energy distributions of AGNs," A&A, 688, A46
   (2024). doi:10.1051/0004-6361/202449329. arXiv:2405.12111.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import NamedTuple

import numpy as np
from jax import numpy as jnp

from tengri.components.agn._params import DEFAULT_AGN_LOG_LBOL, DEFAULT_AGN_LOG_MBH
from tengri.components.agn._phys import (
    bolometric_integral_nu as _bolometric_integral_nu,
    wavelength_to_nu as _wavelength_to_nu,
)
from tengri.utils.grid_interp import resample_template
from tengri.utils.physics_constants import L_SUN as _LSUN_ERG

__all__ = [
    "KD18AGNfitterGrid",
    "KD18AGNfitterWarmIndexGrid",
    "create_kd18_agnfitter_from_grid",
    "create_kd18_agnfitter_warmindex_from_grid",
    "kd18_agnfitter_grid_support",
    "kd18_agnfitter_sed",
    "kd18_agnfitter_sed_from_grid",
    "kd18_agnfitter_warmindex_grid_support",
    "kd18_agnfitter_warmindex_sed",
    "kd18_agnfitter_warmindex_sed_from_grid",
    "load_kd18_agnfitter_default_grid",
    "load_kd18_agnfitter_grid",
    "load_kd18_agnfitter_warmindex_default_grid",
    "load_kd18_agnfitter_warmindex_grid",
]

#: log_edd default: the shared ``agn_log_ledd`` declared default (-1.0) falls
#: inside both grids' native extent ([-1.5, 0]), unlike the SN12 case, so no
#: block-local pin is needed -- kept as a literal here to match the existing
#: ``kubota_done_disc_block`` signature style rather than adding a second
#: shared-default constant next to :data:`DEFAULT_AGN_LOG_MBH`.
_DEFAULT_AGN_LOG_LEDD = -1.0


def _renormalize(sed: jnp.ndarray, wavelength: jnp.ndarray, agn_log_lbol: float) -> jnp.ndarray:
    r"""Renormalize a shape-only :math:`F_\nu` template to ``agn_log_lbol``.

    .. math::

        L_\nu(\lambda) = L_{\rm bol}\,
                         \frac{T(\lambda)}{\int T(\nu)\,\mathrm{d}\nu}

    with :math:`L_{\rm bol} = 10^{\rm agn\_log\_lbol}\,L_\odot`. Mirrors
    :func:`tengri.components.agn.slone_netzer.slone_netzer_sed_from_grid`: the
    vendored KD18 template's own absolute flux calibration (a fixed distance,
    AGNfitter-rX's ``BB=0`` free-normalization-off convention) is discarded --
    only its *shape* is kept, and ``agn_log_lbol`` sets the physical
    normalization at runtime, exactly as every other tengri AGN disc block
    does.

    Parameters
    ----------
    sed : array_like, shape (n_wave,)
        Node-interpolated template, resampled onto ``wavelength``. [F_nu, unnormalized]
    wavelength : array_like, shape (n_wave,)
        Rest-frame wavelength grid. [Angstrom]
    agn_log_lbol : float
        :math:`\log_{10}(L_{\rm bol}/L_\odot)`.

    Returns
    -------
    ndarray, shape (n_wave,)
        :math:`L_\nu` [erg/s/Hz].

    Notes
    -----
    **JIT-compatible**: yes.
    """
    nu = _wavelength_to_nu(wavelength)
    l_scale = 10.0**agn_log_lbol * _LSUN_ERG
    if wavelength.dtype == jnp.float32:
        # Float32 (#1206): same two-trap factorization as slone_netzer_sed_from_grid
        # (peak-factor the integrand so no ~1e45 erg/s intermediate overflows).
        import jax

        peak = jax.lax.stop_gradient(jnp.max(jnp.abs(sed)))
        peak = jnp.where(peak > 0.0, peak, 1.0)
        hat_int = _bolometric_integral_nu(sed / peak, nu, floor=1e-30)
        return (l_scale / hat_int) * (sed / peak)
    integral_safe = _bolometric_integral_nu(sed, nu, floor=1e-100)
    return l_scale * sed / integral_safe


# ─────────────────────────────────────────────────────────────────────────
# 2-D grid: (log_mbh, log_edd)
# ─────────────────────────────────────────────────────────────────────────


def _load_kd18_agnfitter_arrays(grid_path: str) -> dict:
    """Load raw numpy arrays from the KD18-agnfitter grid HDF5."""
    import h5py

    with h5py.File(grid_path, "r") as f:
        g = f["kd18_agnfitter"]
        return {
            "log_mbh": np.asarray(g["log_mbh"][:], dtype=np.float64),
            "log_edd": np.asarray(g["log_edd"][:], dtype=np.float64),
            "wavelength": np.asarray(g["wavelength"][:], dtype=np.float64),
            "template": np.asarray(g["template"][:], dtype=np.float64),
        }


class KD18AGNfitterGrid(NamedTuple):
    """AGNfitter-rX KD18 disc template arrays, as a JAX pytree.

    Attributes
    ----------
    template : ndarray, shape (n_mbh, n_edd, n_wave)
        Tabulated disc SEDs [shape only; renormalized on use].
    wave_grid : ndarray, shape (n_wave,)
        Template rest-frame wavelength grid. [Angstrom]
    log_mbh : ndarray, shape (n_mbh,)
        Grid axis, :math:`\\log_{10}(M_{\\rm BH}/M_\\odot)`.
    log_edd : ndarray, shape (n_edd,)
        Grid axis, :math:`\\log_{10}(\\dot M/\\dot M_{\\rm Edd})`.
    """

    template: jnp.ndarray
    wave_grid: jnp.ndarray
    log_mbh: jnp.ndarray
    log_edd: jnp.ndarray


@functools.cache
def load_kd18_agnfitter_grid(grid_path: str) -> KD18AGNfitterGrid:
    """Load a KD18-agnfitter grid HDF5 into a :class:`KD18AGNfitterGrid` pytree.

    Notes
    -----
    **JIT-compatible**: no, performs HDF5 I/O. Call outside the trace.
    """
    raw = _load_kd18_agnfitter_arrays(grid_path)
    return KD18AGNfitterGrid(
        template=np.asarray(raw["template"]),
        wave_grid=np.asarray(raw["wavelength"]),
        log_mbh=np.asarray(raw["log_mbh"]),
        log_edd=np.asarray(raw["log_edd"]),
    )


_GRID_SEARCH_PATH = "kd18_agnfitter_disc_grid.h5"
_NOT_FOUND_MSG = (
    "AGNfitter-rX KD18 disc grid not found. Build it with: "
    "python scripts/build_kd18_grid.py "
    "--input /tmp/AGNfitter-rX/models/BBB/KD18.pickle"
)


def _find_grid() -> str:
    from tengri._data_setup import require_data

    return require_data(_GRID_SEARCH_PATH, _NOT_FOUND_MSG)


def load_kd18_agnfitter_default_grid() -> KD18AGNfitterGrid:
    """Load the packaged KD18-agnfitter grid pytree (discovery + cache).

    This is the ``template_loader`` the ``kd18_agnfitter`` disc block registers.

    Returns
    -------
    KD18AGNfitterGrid

    Raises
    ------
    FileNotFoundError
        If no KD18-agnfitter grid HDF5 is present on disk.
    """
    return load_kd18_agnfitter_grid(_find_grid())


def create_kd18_agnfitter_from_grid(grid_path: str) -> Callable:
    """Load the KD18-agnfitter grid and return a JAX-native interpolation closure.

    Returns
    -------
    callable
        ``fn(wavelength, agn_log_lbol, agn_log_mbh, agn_log_ledd, **_)
        -> L_nu [erg/s/Hz]``.

    Notes
    -----
    **JIT-compatible**: yes, pure ``jnp`` and node-exact bilinear interpolation
    (same rationale as :func:`tengri.components.agn.slone_netzer.create_slone_netzer_from_grid`:
    the disc peak wavelength varies strongly with accretion rate, so bilinear
    -- node-exact -- is used rather than a smooth kernel).
    """
    return functools.partial(kd18_agnfitter_sed_from_grid, load_kd18_agnfitter_grid(grid_path))


def kd18_agnfitter_sed_from_grid(
    grid: KD18AGNfitterGrid,
    wavelength: jnp.ndarray,
    agn_log_lbol: float = DEFAULT_AGN_LOG_LBOL,
    agn_log_mbh: float = DEFAULT_AGN_LOG_MBH,
    agn_log_ledd: float = _DEFAULT_AGN_LOG_LEDD,
    **_kwargs,
) -> jnp.ndarray:
    r"""AGNfitter-rX KD18 disc SED at a single ``(M_BH, Mdot/Mdot_Edd)``.

    Parameters
    ----------
    grid : KD18AGNfitterGrid
        Template arrays, passed as an argument so they thread through JIT.
    wavelength : array_like, shape (n_wave,)
        Rest-frame wavelength grid. [Angstrom]
    agn_log_lbol : float, optional
        ``log10(L_bol / L_sun)``. Default 11.0.
    agn_log_mbh : float, optional
        ``log10(M_BH / M_sun)``. Default 7.0 (on-grid: the KD18 grid's
        ``log_mbh`` axis is ``[6, 10]``, identical to the shared declaration).
    agn_log_ledd : float, optional
        ``log10(Mdot / Mdot_Edd)``. Default -1.0 (on-grid: the KD18 grid's
        ``log_edd`` axis is ``[-1.5, 0]``).

    Returns
    -------
    ndarray, shape (n_wave,)
        Spectral luminosity density. [erg/s/Hz]

    Notes
    -----
    **JIT-compatible**: yes.
    """
    grid_jax = jnp.asarray(grid.template)
    wave_grid = jnp.asarray(grid.wave_grid)
    mbh_ax = jnp.asarray(grid.log_mbh)
    edd_ax = jnp.asarray(grid.log_edd)
    m = jnp.clip(agn_log_mbh, mbh_ax[0], mbh_ax[-1])
    e = jnp.clip(agn_log_ledd, edd_ax[0], edd_ax[-1])
    i = jnp.clip(jnp.searchsorted(mbh_ax, m) - 1, 0, mbh_ax.shape[0] - 2)
    j = jnp.clip(jnp.searchsorted(edd_ax, e) - 1, 0, edd_ax.shape[0] - 2)
    fm = (m - mbh_ax[i]) / (mbh_ax[i + 1] - mbh_ax[i])
    fe = (e - edd_ax[j]) / (edd_ax[j + 1] - edd_ax[j])
    template = (
        (1.0 - fm) * (1.0 - fe) * grid_jax[i, j]
        + (1.0 - fm) * fe * grid_jax[i, j + 1]
        + fm * (1.0 - fe) * grid_jax[i + 1, j]
        + fm * fe * grid_jax[i + 1, j + 1]
    )
    sed = resample_template(wavelength, wave_grid, template, left=0.0, right=0.0)
    return _renormalize(sed, wavelength, agn_log_lbol)


@functools.cache
def _load_default() -> Callable:
    return create_kd18_agnfitter_from_grid(_find_grid())


@functools.cache
def kd18_agnfitter_grid_support() -> dict[str, tuple[float, float]]:
    r"""Parameter support of the shipped KD18-agnfitter grid, read from its own axes.

    Mirrors :func:`tengri.components.agn.slone_netzer.slone_netzer_grid_support`
    (#1586): the closure built by :func:`create_kd18_agnfitter_from_grid` clips
    both parameters onto these axes, so a value outside them collapses onto the
    edge node.

    Returns
    -------
    support : dict[str, tuple[float, float]]
        ``{'agn_log_mbh': (lo, hi), 'agn_log_ledd': (lo, hi)}``.

    Raises
    ------
    FileNotFoundError
        If the packaged grid is not installed.

    Notes
    -----
    **JIT-compatible**: not applicable; composition-time only. Cached.
    """
    raw = _load_kd18_agnfitter_arrays(_find_grid())
    return {
        "agn_log_mbh": (float(raw["log_mbh"][0]), float(raw["log_mbh"][-1])),
        "agn_log_ledd": (float(raw["log_edd"][0]), float(raw["log_edd"][-1])),
    }


def kd18_agnfitter_sed(*args, _template: KD18AGNfitterGrid | None = None, **kwargs) -> jnp.ndarray:
    """AGNfitter-rX KD18 disc (auto-loaded from the packaged HDF5 grid).

    Parameters
    ----------
    _template : KD18AGNfitterGrid, optional
        Pre-loaded grid, threaded in as a JIT argument by the forward model.
        When ``None`` (default) the packaged grid is loaded from disk.

    Returns
    -------
    ndarray, shape (n_wave,)
        Spectral luminosity density. [erg/s/Hz]
    """
    if _template is not None:
        return kd18_agnfitter_sed_from_grid(_template, *args, **kwargs)
    return _load_default()(*args, **kwargs)


# ─────────────────────────────────────────────────────────────────────────
# 3-D grid: (log_mbh, log_edd, gamma_warm)
# ─────────────────────────────────────────────────────────────────────────


def _load_kd18_agnfitter_warmindex_arrays(grid_path: str) -> dict:
    """Load raw numpy arrays from the KD18-agnfitter-warmindex grid HDF5."""
    import h5py

    with h5py.File(grid_path, "r") as f:
        g = f["kd18_agnfitter_warmindex"]
        return {
            "log_mbh": np.asarray(g["log_mbh"][:], dtype=np.float64),
            "log_edd": np.asarray(g["log_edd"][:], dtype=np.float64),
            "gamma_warm": np.asarray(g["gamma_warm"][:], dtype=np.float64),
            "wavelength": np.asarray(g["wavelength"][:], dtype=np.float64),
            "template": np.asarray(g["template"][:], dtype=np.float64),
        }


class KD18AGNfitterWarmIndexGrid(NamedTuple):
    """AGNfitter-rX KD18-warmIndex disc template arrays, as a JAX pytree.

    Attributes
    ----------
    template : ndarray, shape (n_mbh, n_edd, n_gamma_warm, n_wave)
        Tabulated disc SEDs [shape only; renormalized on use].
    wave_grid : ndarray, shape (n_wave,)
        Template rest-frame wavelength grid. [Angstrom]
    log_mbh : ndarray, shape (n_mbh,)
        Grid axis, :math:`\\log_{10}(M_{\\rm BH}/M_\\odot)`.
    log_edd : ndarray, shape (n_edd,)
        Grid axis, :math:`\\log_{10}(\\dot M/\\dot M_{\\rm Edd})`.
    gamma_warm : ndarray, shape (n_gamma_warm,)
        Grid axis, warm Comptonization spectral index (AGNfitter-rX
        ``warmIndex``; Kubota & Done 2018).
    """

    template: jnp.ndarray
    wave_grid: jnp.ndarray
    log_mbh: jnp.ndarray
    log_edd: jnp.ndarray
    gamma_warm: jnp.ndarray


@functools.cache
def load_kd18_agnfitter_warmindex_grid(grid_path: str) -> KD18AGNfitterWarmIndexGrid:
    """Load a KD18-agnfitter-warmindex grid HDF5 into a pytree.

    Notes
    -----
    **JIT-compatible**: no, performs HDF5 I/O. Call outside the trace.
    """
    raw = _load_kd18_agnfitter_warmindex_arrays(grid_path)
    return KD18AGNfitterWarmIndexGrid(
        template=np.asarray(raw["template"]),
        wave_grid=np.asarray(raw["wavelength"]),
        log_mbh=np.asarray(raw["log_mbh"]),
        log_edd=np.asarray(raw["log_edd"]),
        gamma_warm=np.asarray(raw["gamma_warm"]),
    )


_GRID_SEARCH_PATH_WARMINDEX = "kd18_agnfitter_warmindex_disc_grid.h5"
_NOT_FOUND_MSG_WARMINDEX = (
    "AGNfitter-rX KD18-warmIndex disc grid not found. Build it with: "
    "python scripts/build_kd18_grid.py "
    "--input-warmindex /tmp/AGNfitter-rX/models/BBB/KD18_warmInd.pickle"
)


def _find_grid_warmindex() -> str:
    from tengri._data_setup import require_data

    return require_data(_GRID_SEARCH_PATH_WARMINDEX, _NOT_FOUND_MSG_WARMINDEX)


def load_kd18_agnfitter_warmindex_default_grid() -> KD18AGNfitterWarmIndexGrid:
    """Load the packaged KD18-agnfitter-warmindex grid pytree (discovery + cache).

    This is the ``template_loader`` the ``kd18_agnfitter_warmindex`` disc block
    registers.

    Returns
    -------
    KD18AGNfitterWarmIndexGrid

    Raises
    ------
    FileNotFoundError
        If no KD18-agnfitter-warmindex grid HDF5 is present on disk.
    """
    return load_kd18_agnfitter_warmindex_grid(_find_grid_warmindex())


def create_kd18_agnfitter_warmindex_from_grid(grid_path: str) -> Callable:
    """Load the KD18-agnfitter-warmindex grid and return a JAX-native closure.

    Returns
    -------
    callable
        ``fn(wavelength, agn_log_lbol, agn_log_mbh, agn_log_ledd,
        agn_gamma_warm, **_) -> L_nu [erg/s/Hz]``.

    Notes
    -----
    **JIT-compatible**: yes, pure ``jnp`` and node-exact trilinear
    interpolation (the 3-axis analog of
    :func:`create_kd18_agnfitter_from_grid`'s bilinear).
    """
    return functools.partial(
        kd18_agnfitter_warmindex_sed_from_grid, load_kd18_agnfitter_warmindex_grid(grid_path)
    )


def kd18_agnfitter_warmindex_sed_from_grid(
    grid: KD18AGNfitterWarmIndexGrid,
    wavelength: jnp.ndarray,
    agn_log_lbol: float = DEFAULT_AGN_LOG_LBOL,
    agn_log_mbh: float = DEFAULT_AGN_LOG_MBH,
    agn_log_ledd: float = _DEFAULT_AGN_LOG_LEDD,
    agn_gamma_warm: float = 2.5,
    **_kwargs,
) -> jnp.ndarray:
    r"""AGNfitter-rX KD18-warmIndex disc SED at a single grid point.

    Parameters
    ----------
    grid : KD18AGNfitterWarmIndexGrid
        Template arrays, passed as an argument so they thread through JIT.
    wavelength : array_like, shape (n_wave,)
        Rest-frame wavelength grid. [Angstrom]
    agn_log_lbol : float, optional
        ``log10(L_bol / L_sun)``. Default 11.0.
    agn_log_mbh : float, optional
        ``log10(M_BH / M_sun)``. Default 7.0 (on-grid).
    agn_log_ledd : float, optional
        ``log10(Mdot / Mdot_Edd)``. Default -1.0 (on-grid).
    agn_gamma_warm : float, optional
        Warm Comptonization spectral index (AGNfitter-rX ``warmIndex``).
        Default 2.5 (the shared ``agn_gamma_warm`` declared default; on-grid,
        the KD18-warmIndex axis is ``[1.5, 4.0]``).

    Returns
    -------
    ndarray, shape (n_wave,)
        Spectral luminosity density. [erg/s/Hz]

    Notes
    -----
    **JIT-compatible**: yes. Trilinear interpolation over the three grid
    axes: node-exact at any grid corner, same rationale as the 2-D bilinear
    case (a smooth kernel is not node-exact and would smear the disc peak).
    """
    grid_jax = jnp.asarray(grid.template)
    wave_grid = jnp.asarray(grid.wave_grid)
    mbh_ax = jnp.asarray(grid.log_mbh)
    edd_ax = jnp.asarray(grid.log_edd)
    warm_ax = jnp.asarray(grid.gamma_warm)

    m = jnp.clip(agn_log_mbh, mbh_ax[0], mbh_ax[-1])
    e = jnp.clip(agn_log_ledd, edd_ax[0], edd_ax[-1])
    w = jnp.clip(agn_gamma_warm, warm_ax[0], warm_ax[-1])
    i = jnp.clip(jnp.searchsorted(mbh_ax, m) - 1, 0, mbh_ax.shape[0] - 2)
    j = jnp.clip(jnp.searchsorted(edd_ax, e) - 1, 0, edd_ax.shape[0] - 2)
    k = jnp.clip(jnp.searchsorted(warm_ax, w) - 1, 0, warm_ax.shape[0] - 2)
    fm = (m - mbh_ax[i]) / (mbh_ax[i + 1] - mbh_ax[i])
    fe = (e - edd_ax[j]) / (edd_ax[j + 1] - edd_ax[j])
    fw = (w - warm_ax[k]) / (warm_ax[k + 1] - warm_ax[k])

    # Trilinear: 8 corners of the (mbh, edd, gamma_warm) cell.
    c000 = grid_jax[i, j, k]
    c001 = grid_jax[i, j, k + 1]
    c010 = grid_jax[i, j + 1, k]
    c011 = grid_jax[i, j + 1, k + 1]
    c100 = grid_jax[i + 1, j, k]
    c101 = grid_jax[i + 1, j, k + 1]
    c110 = grid_jax[i + 1, j + 1, k]
    c111 = grid_jax[i + 1, j + 1, k + 1]
    template = (
        (1.0 - fm) * (1.0 - fe) * (1.0 - fw) * c000
        + (1.0 - fm) * (1.0 - fe) * fw * c001
        + (1.0 - fm) * fe * (1.0 - fw) * c010
        + (1.0 - fm) * fe * fw * c011
        + fm * (1.0 - fe) * (1.0 - fw) * c100
        + fm * (1.0 - fe) * fw * c101
        + fm * fe * (1.0 - fw) * c110
        + fm * fe * fw * c111
    )
    sed = resample_template(wavelength, wave_grid, template, left=0.0, right=0.0)
    return _renormalize(sed, wavelength, agn_log_lbol)


@functools.cache
def _load_default_warmindex() -> Callable:
    return create_kd18_agnfitter_warmindex_from_grid(_find_grid_warmindex())


@functools.cache
def kd18_agnfitter_warmindex_grid_support() -> dict[str, tuple[float, float]]:
    r"""Parameter support of the shipped KD18-agnfitter-warmindex grid.

    Mirrors :func:`kd18_agnfitter_grid_support` (#1586), with the added
    ``agn_gamma_warm`` axis.

    Returns
    -------
    support : dict[str, tuple[float, float]]
        ``{'agn_log_mbh': (lo, hi), 'agn_log_ledd': (lo, hi),
        'agn_gamma_warm': (lo, hi)}``.

    Raises
    ------
    FileNotFoundError
        If the packaged grid is not installed.

    Notes
    -----
    **JIT-compatible**: not applicable; composition-time only. Cached.
    """
    raw = _load_kd18_agnfitter_warmindex_arrays(_find_grid_warmindex())
    return {
        "agn_log_mbh": (float(raw["log_mbh"][0]), float(raw["log_mbh"][-1])),
        "agn_log_ledd": (float(raw["log_edd"][0]), float(raw["log_edd"][-1])),
        "agn_gamma_warm": (float(raw["gamma_warm"][0]), float(raw["gamma_warm"][-1])),
    }


def kd18_agnfitter_warmindex_sed(
    *args, _template: KD18AGNfitterWarmIndexGrid | None = None, **kwargs
) -> jnp.ndarray:
    """AGNfitter-rX KD18-warmIndex disc (auto-loaded from the packaged HDF5 grid).

    Parameters
    ----------
    _template : KD18AGNfitterWarmIndexGrid, optional
        Pre-loaded grid, threaded in as a JIT argument by the forward model.
        When ``None`` (default) the packaged grid is loaded from disk.

    Returns
    -------
    ndarray, shape (n_wave,)
        Spectral luminosity density. [erg/s/Hz]
    """
    if _template is not None:
        return kd18_agnfitter_warmindex_sed_from_grid(_template, *args, **kwargs)
    return _load_default_warmindex()(*args, **kwargs)
