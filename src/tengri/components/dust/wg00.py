# SPDX-License-Identifier: BSD-3-Clause
"""Witt & Gordon (2000) radiative-transfer attenuation: vendored-grid runtime.

Loads the WG00 Monte-Carlo attenuation tables (vendored from FSPS into
``data/wg00_attenuation_grid.h5`` by ``scripts/build_wg00_grid.py``) and
interpolates the effective attenuation optical depth :math:`A(\\lambda; \\tau_V)`
with a pure-JAX triweight kernel in :math:`\\tau_V`, so ``τ_V`` is a fully
differentiable, JIT/vmap-safe *fitted* parameter: matching the SKIRTOR /
Nenkova / Silva+04 paths.

Unlike a fixed ``k(λ)`` law scaled by ``τ_V``, WG00's curve *shape* depends on
``τ_V`` (high-τ sightlines self-shield → grayer effective attenuation), so the
full ``A(λ; τ_V)`` table is interpolated directly and applied as ``exp(-A)``.

Data source: Witt & Gordon 2000 (ApJ 528, 799), as reformatted and distributed
by FSPS (Conroy & Gunn 2010) in ``$SPS_HOME/dust/alldirty_{h,c}.dat``: the same
tables FSPS reads for ``dust_type=3``.
"""

from __future__ import annotations

import functools
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np

from tengri.utils.grid_interp import interp_nd_triweight
from tengri.utils.interpolation import edges_for_grid

__all__ = [
    "WG00_DUST_CURVES",
    "WG00_GEOMETRIES",
    "WG00_STRUCTURES",
    "create_wg00_from_grid",
    "wg00_attenuation",
]

WG00_DUST_CURVES: tuple[str, ...] = ("mw", "smc")
WG00_GEOMETRIES: tuple[str, ...] = ("dusty", "shell", "cloudy")
WG00_STRUCTURES: tuple[str, ...] = ("homogeneous", "clumpy")

_GRID_SEARCH_PATHS: tuple[str, ...] = (
    "data/wg00_attenuation_grid.h5",
    "wg00_attenuation_grid.h5",
)

_NOT_FOUND_MSG = (
    "Witt & Gordon (2000) attenuation grid not found. "
    "Build it with: python scripts/build_wg00_grid.py --download "
    '(or --input-dir "$SPS_HOME/dust").'
)


def _load_wg00_arrays(grid_path: str) -> dict:
    """Load raw numpy arrays from a vendored WG00 attenuation HDF5 grid.

    Parameters
    ----------
    grid_path : str
        Path to ``wg00_attenuation_grid.h5`` from ``scripts/build_wg00_grid.py``.

    Returns
    -------
    dict
        ``wavelength`` (n_wave,), ``tau_v_axis`` (n_tau,), and ``a_lambda``
        (2, 2, 3, n_tau, n_wave) keyed by (structure, dust, geometry, τ, λ).

    Notes
    -----
    **JIT-compatible**: no, performs HDF5 I/O at grid-load time.
    """
    import h5py

    with h5py.File(grid_path, "r") as f:
        g = f["wg00"]
        return {
            "wavelength": np.asarray(g["wavelength"][:], dtype=np.float64),
            "tau_v_axis": np.asarray(g["tau_v_axis"][:], dtype=np.float64),
            "a_lambda": np.asarray(g["a_lambda"][:], dtype=np.float64),
        }


def _select_indices(dust_curve: str, geometry: str, structure: str) -> tuple[int, int, int]:
    """Resolve structural-selector strings to static grid indices."""
    try:
        return (
            WG00_STRUCTURES.index(structure),
            WG00_DUST_CURVES.index(dust_curve),
            WG00_GEOMETRIES.index(geometry),
        )
    except ValueError as exc:
        raise ValueError(
            f"Invalid WG00 selector: dust_curve={dust_curve!r} "
            f"(choose {WG00_DUST_CURVES}), geometry={geometry!r} "
            f"(choose {WG00_GEOMETRIES}), structure={structure!r} "
            f"(choose {WG00_STRUCTURES})."
        ) from exc


def create_wg00_from_grid(
    grid_path: str,
    *,
    dust_curve: str = "mw",
    geometry: str = "shell",
    structure: str = "homogeneous",
) -> Callable:
    r"""Load the WG00 grid and return a JAX-native ``A(λ; τ_V)`` interpolator.

    Parameters
    ----------
    grid_path : str
        Path to ``wg00_attenuation_grid.h5``.
    dust_curve : {"mw", "smc"}
        Underlying dust grain population.
    geometry : {"dusty", "shell", "cloudy"}
        Large-scale star-dust geometry (Witt & Gordon 2000).
    structure : {"homogeneous", "clumpy"}
        Local density structure.

    Returns
    -------
    callable
        ``fn(wavelength, tau_v) -> A(λ)``: the effective attenuation optical
        depth (apply as ``exp(-A)``), interpolated in ``τ_V`` with a triweight
        kernel and onto the requested wavelength grid.

    Raises
    ------
    FileNotFoundError
        If ``grid_path`` does not exist.
    ValueError
        If a structural selector is not one of the tabulated options.

    Notes
    -----
    **JIT-compatible**: yes, the returned closure uses only ``jnp`` and a
    triweight interpolation kernel; the structural selectors are resolved to
    static indices at closure-build time.

    **Gradient-safe**: yes, the triweight kernel is C²-continuous in ``τ_V``.

    **Wavelength range**: the WG00 tables span 1000–30001 Å. Below 1000 Å the
    FUV value ``A(1000 Å)`` is held (a conservative extrapolation; those photons
    are dominated by the Lyman continuum, which the consumer masks out of the
    dust-absorbed pool); above 30001 Å the dust is taken transparent
    (``A → 0``), the physical IR limit.
    """
    is_, id_, ig_ = _select_indices(dust_curve, geometry, structure)
    raw = _load_wg00_arrays(grid_path)

    with jax.ensure_compile_time_eval():
        # Static structural slice → (n_tau, n_wave) table of A(λ; τ_V).
        table = jnp.asarray(raw["a_lambda"][is_, id_, ig_])
        wave_grid = jnp.asarray(raw["wavelength"])
        tau_axis = jnp.asarray(raw["tau_v_axis"])
        edges = (edges_for_grid(tau_axis),)

    def wg00_curve(wavelength: jnp.ndarray, tau_v: float) -> jnp.ndarray:
        r"""Effective attenuation optical depth A(λ) at a single τ_V.

        Parameters
        ----------
        wavelength : array_like, shape (n_wave,)
            Rest-frame wavelength grid. [Å]
        tau_v : float
            V-band optical depth. Interpolated over the tabulated range
            [0.25, 10.0]; clamped at the edges. [dimensionless]

        Returns
        -------
        ndarray, shape (n_wave,)
            Effective attenuation optical depth A(λ) (apply as ``exp(-A)``).
        """
        a_at_tau = interp_nd_triweight(table, (tau_axis,), edges, (tau_v,))
        return jnp.interp(wavelength, wave_grid, a_at_tau, left=a_at_tau[0], right=0.0)

    return wg00_curve


def _find_wg00_grid() -> str:
    """Locate the vendored WG00 grid relative to the package root or CWD."""
    from tengri._data_setup import require_data

    # _GRID_SEARCH_PATHS[-1] is the bare filename; require_data searches every
    # directory the old parents[4] walk reached, plus $TENGRI_DATA_DIR (#1431),
    # and re-raises _NOT_FOUND_MSG verbatim when the grid is absent.
    return require_data(_GRID_SEARCH_PATHS[-1], _NOT_FOUND_MSG)


@functools.cache
def _load_wg00_default(dust_curve: str, geometry: str, structure: str) -> Callable:
    return create_wg00_from_grid(
        _find_wg00_grid(),
        dust_curve=dust_curve,
        geometry=geometry,
        structure=structure,
    )


def wg00_attenuation(
    wavelength: jnp.ndarray,
    tau_v: float,
    *,
    dust_curve: str = "mw",
    geometry: str = "shell",
    structure: str = "homogeneous",
) -> jnp.ndarray:
    r"""Witt & Gordon (2000) effective attenuation A(λ; τ_V) from vendored tables.

    Interpolates the WG00 Monte-Carlo radiative-transfer attenuation grid
    (FSPS ``dust_type=3``) in ``τ_V`` with a pure-JAX triweight kernel.

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Rest-frame wavelength grid. [Å]
    tau_v : float
        V-band optical depth (tabulated range 0.25–10.0). [dimensionless]
    dust_curve : {"mw", "smc"}, optional
        Dust grain population. Default ``"mw"``.
    geometry : {"dusty", "shell", "cloudy"}, optional
        Large-scale star-dust geometry. Default ``"shell"`` (foreground screen).
    structure : {"homogeneous", "clumpy"}, optional
        Local density structure. Default ``"homogeneous"``.

    Returns
    -------
    ndarray, shape (n_wave,)
        Effective attenuation optical depth A(λ) (apply as ``exp(-A)``).

    Raises
    ------
    FileNotFoundError
        If no vendored WG00 grid HDF5 is present. Build it with
        ``scripts/build_wg00_grid.py``.

    Notes
    -----
    **JIT-compatible**: yes, loads the vendored grid once via a cached closure,
    then interpolates with pure JAX. **Gradient-safe**: yes, ``τ_V`` is a
    differentiable, traceable parameter.

    References
    ----------
    .. [1] A. N. Witt and K. D. Gordon, "Multiple Scattering in Clumpy Media.
       II. Galactic Environments," ApJ, 528, 799 (2000).
       https://doi.org/10.1086/308197
    """
    fn = _load_wg00_default(dust_curve, geometry, structure)
    return fn(wavelength, tau_v)
