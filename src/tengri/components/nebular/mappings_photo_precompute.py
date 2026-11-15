# SPDX-License-Identifier: BSD-3-Clause
"""Precompute adapter for MAPPINGS V photoionization grid (Flury et al. 2024).

Implements :class:`~tengri.forward.precompute.protocol.PrecomputeModule` for
the Flury et al. (2024) MAPPINGS V photoionization grids with stellar (SB99/BPASS)
and AGN (OPTXAGNF) modes. This adapter wraps stellar mode with 4D grid axes:

    (neb_logZ_gas, log_age, neb_logU, neb_logn)

with discrete stellar_lib (SB99 or BPASS) selected at precompute time.

Lines are preintegrated through filter curves via a precomputed ``(n_lines, n_filt)``
weight matrix, following the canonical CLOUDY pattern.

References
----------
.. [1] S. R. Flury et al., "The MAPPINGS V Survey: Expanding the reach of
       photoionization models to 3D nebular grids," arXiv:2412.06763 (2024).
.. [2] R. J. R. Sutherland and M. A. Dopita, "Spectral Synthesis Modeling
       of AGN Heating in Starburst and Post-Starburst Galaxies," ApJS, 229,
       34 (2017). https://doi.org/10.3847/1538-4365/aa6541

"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from tengri.components.nebular._constants import (
    _LOG10_ZSUN,
    _LSUN_ERG,
)
from tengri.components.nebular._params import DEFAULT_NEB_FESC
from tengri.components.nebular._shared import _interp_index_weight
from tengri.components.nebular.mappings_photo import (
    _DEFAULT_GRID_PATH,
    _load_stellar_grid,
)

# Continuous axes of the stellar MAPPINGS V grid. (The discrete
# stellar_lib axis (sb99 or bpass) is selected at build time.)
AXIS_PARAMS: tuple[str, ...] = (
    "neb_logZ_gas",
    "log_age",
    "neb_logU",
    "neb_logn",
)

# "log_age" and "neb_logn" are internal grid-axis labels, not declared user
# parameters. They are set at precompute time (log_age from the SSP grid; neb_logn
# from the precompute config or as defaults passed by the backend). They should
# never be collapsed by user declaration. See issue #1827.
INTERNAL_AXES: frozenset[str] = frozenset({"log_age", "neb_logn"})


def precompute(
    filter_waves: list,
    filter_trans: list,
    redshift: float,
    parameters: Any,
    *,
    grid_path: str | Path | None = None,
    stellar_lib: str = "sb99",
    density_structure: str = "cpr",
) -> dict:
    """Build preintegrated MAPPINGS V stellar grid, auto-collapsing Fixed axes.

    Loads the HDF5 grid for the selected stellar library (SB99 or BPASS),
    then preintegrates emission lines through filter curves.

    Parameters
    ----------
    filter_waves : list[ndarray]
        Wavelength grid per filter [Angstrom], observed frame.
    filter_trans : list[ndarray]
        Transmission per filter (0–1).
    redshift : float
        Source redshift. [dimensionless]
    parameters : Parameters or None
        Parameters spec, used to detect Fixed-axis parameters.
    grid_path : str or Path, keyword-only
        Path to ``flury2024_grids.h5``. If None, defaults to
        ``data/flury2024_grids.h5`` relative to package root.
    stellar_lib : str, keyword-only
        Stellar library: ``"sb99"`` or ``"bpass"``. Default: ``"sb99"``.
    density_structure : str, keyword-only
        Density structure: ``"cpr"`` (isobaric) or ``"cdn"`` (isochoric).
        Default: ``"cpr"``.

    Returns
    -------
    dict
        Keys:

        - ``line_wavelengths``: (n_lines,) array [Angstrom].
        - ``grid_axes``: tuple of full grid axes (jax arrays).
        - ``axes``: tuple of remaining (non-collapsed) grid axes.
        - ``logHB_per_logq``: grid of Hβ per-ionizing-photon luminosities.
        - ``line_ratios``: grid of line-to-Hβ ratios.
        - ``zo_axis``: original solar-relative metallicity axis.
        - ``_collapsed_axes``: dict of collapsed axis indices and values.

    References
    ----------
    .. [1] S. R. Flury et al., "The MAPPINGS V Survey," arXiv:2412.06763 (2024).

    Notes
    -----
    **JIT-compatible**: no, this is a build-time function using NumPy.

    """
    if grid_path is None:
        grid_path = _DEFAULT_GRID_PATH

    loaded = _load_stellar_grid(grid_path, model=stellar_lib, density=density_structure)
    grid = loaded.grid

    # Build the (n_lines, n_filt) filter projection matrix
    n_lines = len(grid.line_wavelengths)
    n_filt = len(filter_waves)
    line_weight_matrix = np.zeros((n_lines, n_filt), dtype=np.float64)

    for i_line, line_wave_aa in enumerate(grid.line_wavelengths):
        # Redshift line wavelength to observed frame
        line_wave_obs = line_wave_aa * (1.0 + redshift)

        # Integrate line (delta function) through each filter
        for i_filt, (fw, ft) in enumerate(zip(filter_waves, filter_trans)):
            fw = np.asarray(fw)
            ft = np.asarray(ft)
            line_weight_matrix[i_line, i_filt] = np.interp(
                line_wave_obs, fw, ft, left=0.0, right=0.0
            )

    # Grid axes: (ζ_O, log_age, logU, logn)
    # Convert ζ_O back to absolute log10(Z) for AXIS_PARAMS consistency
    # float64: the stored axis is float32 and numpy's float32 log10 is platform-dependent (#3067).
    logZ_axis = np.log10(np.asarray(grid.zo_axis, dtype=np.float64)) + _LOG10_ZSUN

    axes_np = (
        logZ_axis,
        grid.log_age_yr_axis,
        grid.logU_axis,
        grid.logn_axis,
    )

    axes_jax = tuple(jnp.asarray(ax) for ax in axes_np)

    # Auto-collapse: identify Fixed axes
    fixed: dict[int, float] = {}
    if parameters is not None:
        for i, pname in enumerate(AXIS_PARAMS):
            if parameters.is_fixed(pname):
                fixed[i] = float(parameters.fixed_value(pname))

    remaining_axes = tuple(ax for i, ax in enumerate(axes_jax) if i not in fixed)

    return {
        "line_wavelengths": jnp.asarray(grid.line_wavelengths),
        "line_weight_matrix": jnp.asarray(line_weight_matrix),
        "logHB_per_logq": grid.logHB_per_logq,
        "line_ratios": grid.line_ratios,
        "grid_axes": axes_jax,
        "axes": remaining_axes,
        "zo_axis": jnp.asarray(grid.zo_axis),
        "sfh_idx_cont": loaded.sfh_idx_cont,
        "_collapsed_axes": fixed,
    }


def _multilinear_4d(data: jnp.ndarray, axes: tuple, point: tuple) -> jnp.ndarray:
    """Multilinear interpolation over four leading axes; trailing dimensions pass through.

    Parameters
    ----------
    data : ndarray, shape (N_0, N_1, N_2, N_3, ...)
        Table on the four axes.
    axes : tuple of ndarray
        Sorted grid points of each leading axis.
    point : tuple of float
        Query coordinates, one per axis.

    Returns
    -------
    ndarray
        Interpolated value, shape ``data.shape[4:]``.

    """
    brackets = [_interp_index_weight(v, ax) for v, ax in zip(point, axes)]
    total = 0.0
    for corner in itertools.product((0, 1), repeat=4):
        weight = 1.0
        index = []
        for (idx, w), c in zip(brackets, corner):
            weight = weight * (w if c else 1.0 - w)
            index.append(idx + c)
        total = total + weight * data[tuple(index)]
    return total


def build_lookup(preint: dict, **kwargs: Any) -> dict:
    """Build the runtime MAPPINGS V stellar emission lookup.

    Returns a callable with signature::

        fn(log_qh, *free_axis_vals, neb_fesc=0.0) -> (line_wavelengths, line_lum)

    Returns
    -------
    dict
        Keys:

        - ``predict_lines``: JIT-compiled callable.
        - ``line_wavelengths``: line vacuum wavelengths [Angstrom].

    Notes
    -----
    **JIT-compatible**: yes, the returned function is fully JAX-native.

    """
    line_wavelengths = preint["line_wavelengths"]
    logHB_per_logq = preint["logHB_per_logq"]
    line_ratios = preint["line_ratios"]
    zo_axis = preint["zo_axis"]
    grid_axes = preint["grid_axes"]
    collapsed_axes = preint.get("_collapsed_axes", {})
    sfh_idx_cont = preint["sfh_idx_cont"]

    @jax.jit
    def predict_lines(log_qh, *free_axis_vals, neb_fesc=DEFAULT_NEB_FESC):
        """Compute MAPPINGS V stellar NLR emission lines via linear interpolation.

        Parameters
        ----------
        log_qh : float
            log10(Q_H [photons/s]).
        *free_axis_vals : tuple of float
            Per remaining axis values (after collapse).
        neb_fesc : float
            Ionizing photon escape fraction [0, 1]. Default 0.0.

        Returns
        -------
        wavelengths : ndarray, shape (n_lines,)
            Line vacuum wavelengths [Angstrom].
        luminosities : ndarray, shape (n_lines,)
            Line luminosities [L_sun], scaled by Q_H and escape fraction.

        """
        # Reconstruct full grid point from free params and fixed values
        full_point = []
        free_idx = 0
        for axis_idx in range(len(grid_axes)):
            if axis_idx in collapsed_axes:
                full_point.append(collapsed_axes[axis_idx])
            else:
                full_point.append(free_axis_vals[free_idx])
                free_idx += 1

        # The table is on ζ_O; the axis handed to the kernel is absolute log10(Z).
        zo_val = 10.0 ** (full_point[0] - _LOG10_ZSUN)
        point = (zo_val, full_point[1], full_point[2], full_point[3])

        # Cont SFH row of the (N_z, N_a, N_s, N_u, N_n[, N_lines]) tables.
        logHB_slice = logHB_per_logq[:, :, sfh_idx_cont]
        ratios_slice = line_ratios[:, :, sfh_idx_cont]
        table_axes = (zo_axis, grid_axes[1], grid_axes[2], grid_axes[3])

        logHB_interp = _multilinear_4d(logHB_slice, table_axes, point)
        ratios_interp = _multilinear_4d(ratios_slice, table_axes, point)

        # Convert to luminosity: L_Hβ = 10^{logHB_per_logq} × Q_H
        l_hb_erg = (10.0**logHB_interp) * (10.0**log_qh)
        line_lum = ratios_interp * l_hb_erg * (1.0 - neb_fesc) / _LSUN_ERG

        return line_wavelengths, line_lum

    return {
        "predict_lines": predict_lines,
        "line_wavelengths": line_wavelengths,
    }
