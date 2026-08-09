# SPDX-License-Identifier: BSD-3-Clause
r"""Slone & Netzer (2012) alpha-disc accretion-disk model.

The Slone & Netzer (2012 [1]_) optically-thick, geometrically-thin alpha-disc
templates, interpolated over ``(log M_BH, log Mdot/Mdot_Edd)`` from a
precomputed HDF5 grid built from AGNfitter-rX's ``SN12.pickle`` by
``scripts/build_slone_netzer_grid.py``.

Uses the same template library as AGNfitter-rX (Martinez-Ramirez et al. 2024
[2]_); validated against its output. AGNfitter-rX's ``MODEL_AGNfitter.BBB``
uses this same library. This is the
fourth of AGNfitter-rX's four accretion-disk libraries (alongside Richards+2006
``richards2006``, Kubota & Done 2018 ``multicolor``/``kubota_done``, and
Temple+2021 ``qsogen``).

Eddington axis
--------------
The SN12 SED stores 12 accretion-rate columns; AGNfitter-rX labels them with
``logEddra-values[:12]`` ∈ ``[-4.0, -1.96]`` (the build script reproduces this
exactly). The grid therefore spans the sub-Eddington regime; the template is
shape-only and renormalized to ``agn_log_lbol`` at runtime.

References
----------
.. [1] O. Slone and H. Netzer, "The effect of disc winds on the optical and
   ultraviolet emission lines of active galactic nuclei," MNRAS, 426, 656
   (2012). doi:10.1111/j.1365-2966.2012.21699.x.
.. [2] L. N. Martinez-Ramirez, et al., "AGNFITTER-RX: Modeling the
   radio-to-X-ray spectral energy distributions of AGNs," A&A, 688, A46
   (2024). doi:10.1051/0004-6361/202449329. arXiv:2405.12111.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from tengri.components.agn._phys import bolometric_integral_nu as _bolometric_integral_nu
from tengri.utils.grid_interp import resample_template
from tengri.utils.physics_constants import L_SUN as _LSUN_ERG

__all__ = [
    "SloneNetzerGrid",
    "create_slone_netzer_from_grid",
    "load_slone_netzer_default_grid",
    "load_slone_netzer_grid",
    "slone_netzer_sed",
    "slone_netzer_sed_from_grid",
]

from tengri.components.agn._params import DEFAULT_AGN_LOG_LBOL
from tengri.utils.physics_constants import C_AA as _C_AA_PER_S


def _wavelength_to_nu(wavelength: jnp.ndarray) -> jnp.ndarray:
    """Convert wavelength [Å] to frequency [Hz]."""
    return _C_AA_PER_S / wavelength


def _load_slone_netzer_arrays(grid_path: str) -> dict:
    """Load raw numpy arrays from the Slone & Netzer grid HDF5."""
    import h5py

    with h5py.File(grid_path, "r") as f:
        g = f["slone_netzer"]
        return {
            "log_mbh": np.asarray(g["log_mbh"][:], dtype=np.float64),
            "log_edd": np.asarray(g["log_edd"][:], dtype=np.float64),
            "wavelength": np.asarray(g["wavelength"][:], dtype=np.float64),
            "template": np.asarray(g["template"][:], dtype=np.float64),
        }


def create_slone_netzer_from_grid(grid_path: str) -> Callable:
    """Load the SN12 grid and return a JAX-native interpolation closure.

    Parameters
    ----------
    grid_path : str
        Path to ``slone_netzer_disc_grid.h5``.

    Returns
    -------
    callable
        ``fn(wavelength, agn_log_lbol, agn_log_mbh, agn_log_ledd, **_)
        -> L_nu [erg/s/Hz]``.

    Raises
    ------
    FileNotFoundError
        If ``grid_path`` does not exist.
    KeyError
        If the grid file is missing any dataset under ``/slone_netzer``.

    Notes
    -----
    **JIT-compatible**: yes — pure ``jnp`` and node-exact bilinear interpolation.

    **Gradient-safe**: yes — bilinear interpolation is piecewise-linear (C⁰)
    across both parameter axes with finite gradients inside every cell. Linear
    (rather than a smooth triweight kernel) is required for fidelity: the SN12
    templates' peak shifts strongly with accretion rate, and a smoothing kernel
    is not node-exact (cf. #583).
    """
    return functools.partial(slone_netzer_sed_from_grid, load_slone_netzer_grid(grid_path))


class SloneNetzerGrid(NamedTuple):
    """Slone & Netzer (2012) disc template arrays, as a JAX pytree.

    Carried as a pytree rather than closed over so the forward model can pass
    the library into ``jax.jit`` as an argument; a closure's captured arrays
    are concrete at trace time and freeze into the graph as ``Constant`` ops.

    Attributes
    ----------
    template : ndarray, shape (n_mbh, n_edd, n_wave)
        Tabulated disc SEDs [shape only; renormalized on use].
    wave_grid : ndarray, shape (n_wave,)
        Template rest-frame wavelength grid [Angstrom].
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
def load_slone_netzer_grid(grid_path: str) -> SloneNetzerGrid:
    """Load an SN12 grid HDF5 into a :class:`SloneNetzerGrid` pytree.

    Parameters
    ----------
    grid_path : str
        Path to ``slone_netzer_disc_grid.h5``.

    Returns
    -------
    SloneNetzerGrid

    Notes
    -----
    **JIT-compatible**: no — performs HDF5 I/O. Call outside the trace.
    """
    raw = _load_slone_netzer_arrays(grid_path)
    return SloneNetzerGrid(
        template=np.asarray(raw["template"]),  # (n_mbh, n_edd, n_wave)
        wave_grid=np.asarray(raw["wavelength"]),
        log_mbh=np.asarray(raw["log_mbh"]),
        log_edd=np.asarray(raw["log_edd"]),
    )


def load_slone_netzer_default_grid() -> SloneNetzerGrid:
    """Load the packaged SN12 grid pytree (discovery + cache).

    This is the ``template_loader`` the SN12 disc block registers.

    Returns
    -------
    SloneNetzerGrid

    Raises
    ------
    FileNotFoundError
        If no SN12 grid HDF5 is present on disk.
    """
    return load_slone_netzer_grid(_find_grid())


def slone_netzer_sed_from_grid(
    grid: SloneNetzerGrid,
    wavelength: jnp.ndarray,
    agn_log_lbol: float = DEFAULT_AGN_LOG_LBOL,
    # Deliberately NOT DEFAULT_AGN_LOG_MBH: the SN12 grid's log_mbh axis
    # starts at 7.4, so the declared 7.0 would be silently clipped by the
    # jnp.clip below. 8.6 is the grid's center node.
    agn_log_mbh: float = 8.6,
    agn_log_ledd: float = -2.0,
    **_kwargs,
) -> jnp.ndarray:
    r"""Slone & Netzer (2012) disc SED at a single ``(M_BH, Mdot/Mdot_Edd)``.

    Parameters
    ----------
    grid : SloneNetzerGrid
        Template arrays, passed as an argument so they thread through JIT.
    wavelength : array_like, shape (n_wave,)
        Rest-frame wavelength grid. [Å]
    agn_log_lbol : float, optional
        ``log10(L_bol / L_sun)``. Default 11.0.
    agn_log_mbh : float, optional
        ``log10(M_BH / M_sun)``. Default 8.6.
    agn_log_ledd : float, optional
        ``log10(Mdot / Mdot_Edd)``. Default −2.0.

    Returns
    -------
    ndarray, shape (n_wave,)
        Spectral luminosity density. [erg/s/Hz]

    Notes
    -----
    .. math::

        L_\nu(\lambda) = L_{\rm bol}\,
                         \frac{T(\lambda;\,\log M_{\rm BH},\,\log\dot m)}
                              {\int T(\nu;\,\log M_{\rm BH},\,\log\dot m)
                               \,\mathrm{d}\nu}

    with :math:`L_{\rm bol} = 10^{\rm agn\_log\_lbol}\,L_\odot`. The
    template is shape-only; ``agn_log_lbol`` sets the normalization.

    **JIT-compatible**: yes.
    """
    grid_jax = jnp.asarray(grid.template)
    wave_grid = jnp.asarray(grid.wave_grid)
    mbh_ax = jnp.asarray(grid.log_mbh)
    edd_ax = jnp.asarray(grid.log_edd)
    # Node-exact bilinear interpolation over (log_mbh, log_edd). The SN12
    # templates' peak wavelength varies strongly with accretion rate, so a
    # smooth triweight kernel (which is not node-exact) smears the peak
    # across neighboring nodes by 30-50%; bilinear reproduces the library
    # templates at grid nodes exactly (cf. the DL14 WavePrecomp fix #583).
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
    nu = _wavelength_to_nu(wavelength)
    integral_safe = _bolometric_integral_nu(sed, nu, floor=1e-100)
    l_scale = 10.0**agn_log_lbol * _LSUN_ERG
    return l_scale * sed / integral_safe


_GRID_SEARCH_PATHS: tuple[str, ...] = (
    "data/slone_netzer_disc_grid.h5",
    "slone_netzer_disc_grid.h5",
)

_NOT_FOUND_MSG = (
    "Slone & Netzer (2012) disc grid not found. Build it with: "
    "python scripts/build_slone_netzer_grid.py "
    "--input /tmp/AGNfitter-rX/models/BBB/SN12.pickle"
)


def _find_grid() -> str:
    from tengri._data_setup import require_data

    # _GRID_SEARCH_PATHS[-1] is the bare filename; require_data searches every
    # directory the old parents[4] walk reached, plus $TENGRI_DATA_DIR (#1431),
    # and re-raises _NOT_FOUND_MSG verbatim when the grid is absent.
    return require_data(_GRID_SEARCH_PATHS[-1], _NOT_FOUND_MSG)


@functools.cache
def _load_default() -> Callable:
    return create_slone_netzer_from_grid(_find_grid())


@functools.cache
def slone_netzer_grid_support() -> dict[str, tuple[float, float]]:
    r"""Parameter support of the shipped SN12 grid, read from its own axes.

    A parameter declaration records one support — its prior. A block that
    interpolates a template library carries a *second*, implicit one: the
    extent of the axes it interpolates over. The closure built by
    :func:`create_slone_netzer_from_grid` clips both parameters onto these
    axes, so a value outside them collapses onto the edge node — the SED is
    bit-identical and the gradient is exactly zero, with no NaN, warning or
    error to reveal it (#1586).

    This accessor exposes that second support so a caller can compare it
    against a declared prior at composition time instead of discovering the
    clip empirically.

    Returns
    -------
    support : dict[str, tuple[float, float]]
        ``{'agn_log_mbh': (lo, hi), 'agn_log_ledd': (lo, hi)}`` — inclusive
        bounds, both dimensionless. ``agn_log_mbh`` is
        :math:`\log_{10}(M_{\rm BH}/M_\odot)`, ``agn_log_ledd`` is
        :math:`\log_{10}(\dot m/\dot m_{\rm Edd})`.

    Raises
    ------
    FileNotFoundError
        If the packaged grid is not installed.

    Notes
    -----
    **JIT-compatible**: not applicable — pure Python/NumPy, called at
    composition time only. Cached, so the grid is read once per process.

    The bounds are **read from the file**, and taken as ``axis[0]`` /
    ``axis[-1]`` so they are exactly the arguments the ``jnp.clip`` pair in
    :func:`create_slone_netzer_from_grid` uses. A hand-copied literal would
    silently go stale if the packaged grid were ever rebuilt on new axes.
    """
    raw = _load_slone_netzer_arrays(_find_grid())
    return {
        "agn_log_mbh": (float(raw["log_mbh"][0]), float(raw["log_mbh"][-1])),
        "agn_log_ledd": (float(raw["log_edd"][0]), float(raw["log_edd"][-1])),
    }


def slone_netzer_sed(*args, _template: SloneNetzerGrid | None = None, **kwargs) -> jnp.ndarray:
    """Slone & Netzer (2012) disc (auto-loaded from the packaged HDF5 grid).

    Parameters
    ----------
    _template : SloneNetzerGrid, optional
        Pre-loaded grid, threaded in as a JIT argument by the forward model.
        When ``None`` (default) the packaged grid is loaded from disk and —
        if this call happens under trace — baked into the graph as constants.

    Returns
    -------
    ndarray, shape (n_wave,)
        Spectral luminosity density. [erg/s/Hz]
    """
    if _template is not None:
        return slone_netzer_sed_from_grid(_template, *args, **kwargs)
    return _load_default()(*args, **kwargs)
