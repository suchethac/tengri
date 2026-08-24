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

from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from tengri.components.nebular._constants import (
    _LOG10_ZSUN,
    _LSUN_ERG,
)
from tengri.components.nebular.mappings_photo import (
    _DEFAULT_GRID_PATH,
    _load_stellar_grid,
)

# Continuous axes of the stellar MAPPINGS V grid. (The discrete
# stellar_lib axis: sb99 or bpass: is selected at build time.)
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
    filter_waves: list[ndarray]
        Wavelength grid per filter [Angstrom], observed frame.
    filter_trans: list[ndarray]
        Transmission per filter (0–1).
    redshift: float
        Source redshift. [dimensionless]
    parameters: Parameters or None
        Parameters spec, used to detect Fixed-axis parameters.
    grid_path: str or Path, keyword-only
        Path to ``flury2024_grids.h5``. If None, defaults to
        ``data/flury2024_grids.h5`` relative to package root.
    stellar_lib: str, keyword-only
        Stellar library: ``"sb99"`` or ``"bpass"``. Default: ``"sb99"``.
    density_structure: str, keyword-only
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

    grid = _load_stellar_grid(grid_path, model=stellar_lib, density=density_structure)

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
    logZ_axis = np.log10(grid.zo_axis) + _LOG10_ZSUN

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
        "_collapsed_axes": fixed,
    }


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

    @jax.jit
    def predict_lines(log_qh, *free_axis_vals, neb_fesc=0.0):
        """Compute MAPPINGS V stellar NLR emission lines via linear interpolation.

        Parameters
        ----------
        log_qh: float
            log10(Q_H [photons/s]).
        *free_axis_vals: tuple of float
            Per remaining axis values (after collapse).
        neb_fesc: float
            Ionizing photon escape fraction [0, 1]. Default 0.0.

        Returns
        -------
        wavelengths: ndarray, shape (n_lines,)
            Line vacuum wavelengths [Angstrom].
        luminosities: ndarray, shape (n_lines,)
            Line luminosities [L_sun], scaled by Q_H and escape fraction.

        """
        from tengri.components.nebular._shared import _interp_index_weight

        # Reconstruct full grid point from free params and fixed values
        full_point = []
        free_idx = 0
        for axis_idx in range(len(grid_axes)):
            if axis_idx in collapsed_axes:
                full_point.append(collapsed_axes[axis_idx])
            else:
                full_point.append(free_axis_vals[free_idx])
                free_idx += 1

        # Convert absolute log10(Z) back to ζ_O for interpolation
        logZ_val = full_point[0]
        zo_val = 10.0 ** (logZ_val - jnp.log10(_LOG10_ZSUN))
        full_point = (zo_val, *tuple(full_point[1:]))

        # Linear interpolation on 4D grid
        # Grid shape: (N_z, N_a, N_s, N_u, N_n) where N_s is discrete (sfh)
        # Select cont SFH (index 1)
        logHB_slice = logHB_per_logq[:, :, 1, :, :]  # (N_z, N_a, N_u, N_n)
        ratios_slice = line_ratios[:, :, 1, :, :, :]  # (N_z, N_a, N_u, N_n, nl)

        # Interpolation on 4 axes: zo, age, logU, logn
        iz, wz = _interp_index_weight(full_point[0], zo_axis)
        ia, wa = _interp_index_weight(full_point[1], grid_axes[1])
        iu, wu = _interp_index_weight(full_point[2], grid_axes[2])
        in_, wn = _interp_index_weight(full_point[3], grid_axes[3])

        # Bilinear per pair: interp over logn, then logU, then age, then zo
        def _hb_at(iz_, ia_, iu_, in_):
            return logHB_slice[iz_, ia_, iu_, in_]

        hb_c00 = _hb_at(iz, ia, iu, in_) * (1 - wn) + _hb_at(iz, ia, iu, in_ + 1) * wn
        hb_c10 = _hb_at(iz, ia, iu + 1, in_) * (1 - wn) + _hb_at(iz, ia, iu + 1, in_ + 1) * wn
        hb_c01 = _hb_at(iz + 1, ia, iu, in_) * (1 - wn) + _hb_at(iz + 1, ia, iu, in_ + 1) * wn
        hb_c11 = (
            _hb_at(iz + 1, ia, iu + 1, in_) * (1 - wn) + _hb_at(iz + 1, ia, iu + 1, in_ + 1) * wn
        )

        hb_c0 = hb_c00 * (1 - wu) + hb_c10 * wu
        hb_c1 = hb_c01 * (1 - wu) + hb_c11 * wu

        hb_c00a = hb_c0 * (1 - wa)
        hb_c10a = hb_c1 * wa
        logHB_cu = hb_c00a + hb_c10a

        logHB_interp = logHB_cu * (1 - wz) + logHB_cu * wz

        # Same for line ratios with extra line dimension
        def _ratio_at(iz_, ia_, iu_, in_):
            return ratios_slice[iz_, ia_, iu_, in_, :]

        r_c00 = _ratio_at(iz, ia, iu, in_) * (1 - wn) + _ratio_at(iz, ia, iu, in_ + 1) * wn
        r_c10 = _ratio_at(iz, ia, iu + 1, in_) * (1 - wn) + _ratio_at(iz, ia, iu + 1, in_ + 1) * wn
        r_c01 = _ratio_at(iz + 1, ia, iu, in_) * (1 - wn) + _ratio_at(iz + 1, ia, iu, in_ + 1) * wn
        r_c11 = (
            _ratio_at(iz + 1, ia, iu + 1, in_) * (1 - wn)
            + _ratio_at(iz + 1, ia, iu + 1, in_ + 1) * wn
        )

        r_c0 = r_c00 * (1 - wu) + r_c10 * wu
        r_c1 = r_c01 * (1 - wu) + r_c11 * wu

        r_c00a = r_c0 * (1 - wa)
        r_c10a = r_c1 * wa
        ratios_cu = r_c00a + r_c10a

        ratios_interp = ratios_cu * (1 - wz) + ratios_cu * wz

        # Convert to luminosity: L_Hβ = 10^{logHB_per_logq} × Q_H
        l_hb_erg = (10.0**logHB_interp) * (10.0**log_qh)
        line_lum = ratios_interp * l_hb_erg * (1.0 - neb_fesc) / _LSUN_ERG

        return line_wavelengths, line_lum

    return {
        "predict_lines": predict_lines,
        "line_wavelengths": line_wavelengths,
    }
