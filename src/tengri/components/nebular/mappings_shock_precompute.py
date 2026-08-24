# SPDX-License-Identifier: BSD-3-Clause
"""Precompute adapter for MAPPINGS III+V shock emission model.

Implements :class:`~tengri.forward.precompute.protocol.PrecomputeModule` for
the Allen+2008 MAPPINGS III and Alarie+Morisset 2019 MAPPINGS V shock grids,
exposing a 4D preintegrated line-luminosity grid:

    (shock_velocity, shock_b_over_sqrt_n, shock_log_density, shock_abundance)

with discrete abundance selections and continuous velocity/B-field/density
interpolation via triweight kernel. Lines are preintegrated through filter
curves via a precomputed ``(n_lines, n_filt)`` weight matrix.

References
----------
.. [1] D. A. Allen et al., "The Distance and Metallicity of the Galaxy M33,"
       ApJS, 178, 20 (2008). https://doi.org/10.1086/589652
.. [2] R. J. R. Sutherland and M. A. Dopita, "Spectral Synthesis Modeling
       of AGN Heating in Starburst and Post-Starburst Galaxies," ApJS, 229,
       34 (2017). https://doi.org/10.3847/1538-4365/aa6541
.. [3] D. Alarie and C. Morisset, "Synthetic Narrow-Line Emission from a
       Large Grid of CLOUDY Models," Rev. Mex. Astron. Astrofis., 55, 279
       (2019). https://doi.org/10.22201/ia.01851101p.2019.55.02.14

"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from tengri.components.nebular._constants import (
    _LSUN_ERG,
)
from tengri.components.nebular.shock import (
    _load_mappings_grids,
    _resolve_abundance,
)
from tengri.utils.grid_interp import (
    interp_nd_triweight,
)
from tengri.utils.interpolation import edges_for_grid

# Axis parameters for shock grid.
# Note: shock_abundance is discrete and selected at build time.
AXIS_PARAMS: tuple[str, ...] = (
    "shock_velocity",
    "shock_b_over_sqrt_n",
    "shock_log_density",
)


def precompute(
    filter_waves: list,
    filter_trans: list,
    redshift: float,
    parameters: Any,
    *,
    shock_abundance: str = "solar",
    shock_component: str = "combined",
) -> dict:
    """Build preintegrated MAPPINGS shock grid, auto-collapsing Fixed axes.

    Loads the HDF5 grid (or hardcoded fallback), selects an abundance and
    component, then preintegrates emission lines through filter curves.

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
    shock_abundance: str, keyword-only
        Abundance set: ``"solar"``, ``"2xsolar"``, ``"dopita2005"``, ``"lmc"``,
        or ``"smc"``. Or full name from database. Default: ``"solar"``.
    shock_component: str, keyword-only
        Component: ``"shock"``, ``"precursor"``, or ``"combined"``.
        Default: ``"combined"``.

    Returns
    -------
    dict
        Keys:

        - ``line_wavelengths``: (n_lines,) array [Angstrom].
        - ``grid_axes``: tuple of full grid axes (jax arrays).
        - ``axes``: tuple of remaining (non-collapsed) grid axes.
        - ``line_ratios_grid``: grid of line-to-Hβ ratios.
        - ``_preint``: metadata (axes, edges).

    References
    ----------
    .. [1] D. A. Allen et al., "The Distance and Metallicity of the Galaxy M33,"
           ApJS, 178, 20 (2008).
    .. [3] D. Alarie and C. Morisset, "Synthetic Narrow-Line Emission from a
           Large Grid of CLOUDY Models," Rev. Mex. Astron. Astrofis., 55, 279
           (2019).

    Notes
    -----
    **JIT-compatible**: no, this is a build-time function using NumPy.

    The abundance and component are chosen at precompute time, not at runtime.
    Continuous parameters (velocity, B-field, density) use triweight
    interpolation at runtime.

    """
    grids = _load_mappings_grids()

    if grids is None or "mappings5" not in grids:
        raise ValueError(
            "MAPPINGS shock grid not found. Download via scripts/download_mappings_templates.py"
        )

    g = grids["mappings5"]

    # Resolve abundance and component indices
    i_abund = _resolve_abundance(shock_abundance, g["abundance_names"])

    component_map = {
        "shock": "shock_ratios",
        "precursor": "precursor_ratios",
        "combined": "combined_ratios",
    }
    ratio_array = g[component_map.get(shock_component, "combined_ratios")]
    # Shape: (N_abund, N_n, N_v, N_B, N_lines)

    # Build the (n_lines, n_filt) filter projection matrix
    n_lines = len(g["line_names"])
    n_filt = len(filter_waves)
    line_weight_matrix = np.zeros((n_lines, n_filt), dtype=np.float64)

    line_waves = np.asarray(g["line_wavelengths_aa"])
    for i_line, line_wave_aa in enumerate(line_waves):
        # Redshift line wavelength to observed frame
        line_wave_obs = line_wave_aa * (1.0 + redshift)

        # Integrate line (delta function) through each filter
        for i_filt, (fw, ft) in enumerate(zip(filter_waves, filter_trans)):
            fw = np.asarray(fw)
            ft = np.asarray(ft)
            line_weight_matrix[i_line, i_filt] = np.interp(
                line_wave_obs, fw, ft, left=0.0, right=0.0
            )

    # Grid axes: (velocity, B-field, log_density) from shock grid
    v_grid = np.asarray(g["velocities_kms"])
    b_grid = np.asarray(g["b_axis"])
    log_den_grid = np.asarray(g["log_density_cm3"])

    axes_np = (
        v_grid,
        b_grid,
        log_den_grid,
    )

    axes_jax = tuple(jnp.asarray(ax) for ax in axes_np)

    # Slice out abundance axis → (N_n, N_v, N_B, N_lines)
    # Transpose to (N_v, N_B, N_n, N_lines) for consistent axis ordering
    grid_abund = ratio_array[i_abund]  # (N_n, N_v, N_B, N_lines)
    grid_vbn = np.transpose(grid_abund, (1, 2, 0, 3))  # (N_v, N_B, N_n, N_lines)

    # Auto-collapse: identify Fixed axes
    fixed: dict[int, float] = {}
    if parameters is not None:
        for i, pname in enumerate(AXIS_PARAMS):
            if parameters.is_fixed(pname):
                fixed[i] = float(parameters.fixed_value(pname))

    remaining_axes = tuple(ax for i, ax in enumerate(axes_jax) if i not in fixed)

    return {
        "line_wavelengths": jnp.asarray(line_waves),
        "line_weight_matrix": jnp.asarray(line_weight_matrix),
        "line_ratios_grid": jnp.asarray(grid_vbn),
        "grid_axes": axes_jax,
        "axes": remaining_axes,
        "_preint_edges": tuple(edges_for_grid(np.asarray(ax)) for ax in axes_np),
        "_collapsed_axes": fixed,
    }


def build_lookup(preint: dict, **kwargs: Any) -> dict:
    """Build the runtime MAPPINGS shock emission lookup from preintegrated data.

    Returns a callable with signature::

        fn(l_shock_halpha, shock_velocity, shock_b_over_sqrt_n, shock_log_density)
            -> (line_wavelengths, line_lum)

    Returns
    -------
    dict
        Keys:

        - ``predict_lines``: JIT-compiled callable.
        - ``line_wavelengths``: line vacuum wavelengths [Angstrom].

    Notes
    -----
    **JIT-compatible**: yes, the returned function is fully JAX-native.

    **Gradient-safe**: yes, triweight interpolation is fully differentiable.

    """
    line_wavelengths = preint["line_wavelengths"]
    line_ratios_grid = preint["line_ratios_grid"]
    grid_axes = preint["grid_axes"]
    edges = preint["_preint_edges"]
    collapsed_axes = preint.get("_collapsed_axes", {})

    @jax.jit
    def predict_lines(l_shock_halpha, *free_axis_vals):
        """Compute MAPPINGS shock emission lines via triweight interpolation.

        Parameters
        ----------
        l_shock_halpha: float
            Total shock Hα luminosity [erg/s] (normalization anchor).
        *free_axis_vals: tuple of float
            Per remaining axis values (after collapse): velocity [km/s],
            B-field [μG], log_density [cm^-3].

        Returns
        -------
        wavelengths: ndarray, shape (n_lines,)
            Line vacuum wavelengths [Angstrom].
        luminosities: ndarray, shape (n_lines,)
            Line luminosities [L_sun], scaled by Hα and normalized to unity
            Hα ratio.

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

        # Clip to grid bounds
        full_point = tuple(
            jnp.clip(full_point[i], grid_axes[i][0], grid_axes[i][-1])
            for i in range(len(full_point))
        )

        # Triweight interpolation: all 3 axes jointly
        ratios_interp = interp_nd_triweight(line_ratios_grid, grid_axes, edges, full_point)
        # ratios_interp: shape (n_lines,)

        # Normalize to unity Hα ratio: Hα is typically the last line
        # Use Hα ratio as the denominator
        r_ha = ratios_interp[-1]  # Hα is typically last
        r_ha_safe = jnp.where(r_ha > 0.0, r_ha, jnp.ones_like(r_ha))

        # L_line = (ratio / r_ha) * l_shock_halpha
        line_lum = (ratios_interp / r_ha_safe) * l_shock_halpha / _LSUN_ERG

        return line_wavelengths, line_lum

    return {
        "predict_lines": predict_lines,
        "line_wavelengths": line_wavelengths,
    }
