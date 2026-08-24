# SPDX-License-Identifier: BSD-3-Clause
"""Precompute adapter for Feltre+2016 AGN NLR photoionization grid.

Implements :class:`~tengri.forward.precompute.protocol.PrecomputeModule` for
the Feltre, Charlot & Gutkin (2016) CLOUDY c13.03 AGN NLR photoionization grid,
exposing a 4D preintegrated line-luminosity grid:

    (neb_logZ_gas, agn_alpha_ion, neb_logU, neb_xid)

with 20 emission lines (O II, Hβ, O III, O I, N II, Hα, S II, NV, CIV, HeII,
and optical UV lines).

The dual (line_lum, continuum_phot) precompute follows the canonical CLOUDY
pattern: emission lines are projected through filter curves via a precomputed
``(n_lines, n_filt)`` weight matrix, enabling fast filter-integrated line flux
lookup at runtime.

References
----------
.. [1] A. Feltre, S. Charlot, and J. Gutkin, "Updated photoionization models
       of the diffuse ionized gas in galaxies," MNRAS, 456, 3354 (2016).
       https://doi.org/10.1093/mnras/stv2794

"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from tengri.components.nebular._constants import (
    _LSUN_ERG,
)
from tengri.components.nebular.agn_nebular import (
    _DEFAULT_FELTRE_GRID_PATH,
    _load_feltre_grid,
)
from tengri.utils.grid_interp import (
    PreintegratedGrid,
    interp_nd_triweight,
    slice_fixed_axes,
)
from tengri.utils.interpolation import edges_for_grid

# Axis parameters: ordered tuple matching the Feltre grid axes.
# neb_logZ_gas: absolute log10(Z), neb_logU: ionization parameter,
# neb_xid: dust-to-metal ratio, agn_alpha_ion: UV power-law slope.
AXIS_PARAMS: tuple[str, ...] = (
    "neb_logZ_gas",
    "agn_alpha_ion",
    "neb_logU",
    "neb_xid",
)


def precompute(
    filter_waves: list,
    filter_trans: list,
    redshift: float,
    parameters: Any,
    *,
    grid_path: str | Path | None = None,
) -> dict:
    """Build preintegrated Feltre+2016 NLR grid, auto-collapsing Fixed axes.

    Loads the HDF5 grid, builds a (line_wavelengths, logHB_per_logq, line_ratios)
    tuple, then preintegrates emission lines through filter curves via a
    ``(n_lines, n_filt)`` projection matrix. Auto-collapses any axis whose
    corresponding parameter is Fixed in ``parameters``.

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
        Path to ``feltre_grid.h5``. If None, defaults to
        ``data/feltre_grid.h5`` relative to package root.

    Returns
    -------
    dict
        Keys:

        - ``line_lum_grid``: line-luminosity grid after projecting through
          filters, shape (collapsed_dims, n_filters). Per-unit Q_H.
        - ``line_weight_matrix``: (n_lines, n_filters) projection matrix.
        - ``line_wavelengths``: (n_lines,) array.
        - ``axes``: tuple of remaining (non-collapsed) grid axes (jax arrays).
        - ``_preint``: internal PreintegratedGrid metadata (axes, edges).

    References
    ----------
    .. [1] A. Feltre, S. Charlot, and J. Gutkin, "Updated photoionization
           models of the diffuse ionized gas in galaxies," MNRAS, 456, 3354
           (2016). arXiv:1511.08217.
           https://doi.org/10.1093/mnras/stw044

    Notes
    -----
    **JIT-compatible**: no, this is a build-time function using NumPy.

    Emission lines are preintegrated by placing Gaussian profiles (or delta
    functions) at each line's vacuum wavelength and convolving with filter
    transmission curves. The resulting ``line_weight_matrix`` is shape
    (n_lines, n_filters) so that per-line luminosities can be rapidly
    projected: ``line_phot_per_filter = line_weight_matrix @ line_lum_array``.

    """
    if grid_path is None:
        grid_path = _DEFAULT_FELTRE_GRID_PATH

    grid = _load_feltre_grid(grid_path)

    # Build the (n_lines, n_filt) filter projection matrix
    # Each row integrates one line's Gaussian through the filters
    n_lines = len(grid.line_wavelengths_aa)
    n_filt = len(filter_waves)
    line_weight_matrix = np.zeros((n_lines, n_filt), dtype=np.float64)

    for i_line, line_wave_aa in enumerate(grid.line_wavelengths_aa):
        # Redshift line wavelength to observed frame
        line_wave_obs = line_wave_aa * (1.0 + redshift)

        # For each filter, integrate the line (delta or Gaussian) through the
        # transmission curve. Using delta function: integrate transmission at
        # the line wavelength via linear interpolation.
        for i_filt, (fw, ft) in enumerate(zip(filter_waves, filter_trans)):
            fw = np.asarray(fw)
            ft = np.asarray(ft)
            # Interpolate filter transmission at line wavelength
            line_weight_matrix[i_line, i_filt] = np.interp(
                line_wave_obs, fw, ft, left=0.0, right=0.0
            )

    # Now build the continuum photometry grid (though Feltre has no continuum)
    # We still follow the protocol by returning a dummy continuum (zeros)
    axes_np = (
        np.asarray(grid.logZ_axis),
        np.asarray(grid.alpha_axis),
        np.asarray(grid.logUs_axis),
        np.asarray(grid.xi_d_axis),
    )

    # Feltre has no continuum; use a dummy zero grid
    continuum_grid = np.zeros((*[ax.shape[0] for ax in axes_np[:-1]], n_filt), dtype=np.float64)

    preint = PreintegratedGrid(
        phot=continuum_grid,  # Dummy continuum (zeros)
        moment=None,
        axes=tuple(jnp.asarray(ax) for ax in axes_np),
        edges=tuple(edges_for_grid(np.asarray(ax)) for ax in axes_np),
        effective_wavelengths=jnp.zeros(n_filt),
        effective_wavelengths_rest=jnp.zeros(n_filt),
        log10_flux_scale=0.0,  # unit scale; the caller applies the cosmology
        n_filters=n_filt,
    )

    # Auto-collapse: identify Fixed axes
    fixed: dict[int, float] = {}
    for i, pname in enumerate(AXIS_PARAMS):
        if parameters is not None and parameters.is_fixed(pname):
            fixed[i] = float(parameters.fixed_value(pname))

    if fixed:
        collapsed = slice_fixed_axes(preint, fixed)
        remaining_axes = tuple(ax for i, ax in enumerate(preint.axes) if i not in fixed)
        return {
            "line_weight_matrix": jnp.asarray(line_weight_matrix),
            "line_wavelengths": jnp.asarray(grid.line_wavelengths_aa),
            "logHB_per_logq": grid.logHB_per_logq,
            "line_ratios": grid.line_ratios,
            "grid_axes": preint.axes,
            "axes": remaining_axes,
            "_preint": collapsed,
            "_collapsed_axes": fixed,
        }

    return {
        "line_weight_matrix": jnp.asarray(line_weight_matrix),
        "line_wavelengths": jnp.asarray(grid.line_wavelengths_aa),
        "logHB_per_logq": grid.logHB_per_logq,
        "line_ratios": grid.line_ratios,
        "grid_axes": preint.axes,
        "axes": preint.axes,
        "_preint": preint,
    }


def build_lookup(preint: dict, **kwargs: Any) -> dict:
    """Build the runtime Feltre NLR emission lookup from preintegrated data.

    Returns a callable with signature::

        fn(log_qh, neb_logZ_gas, agn_alpha_ion, neb_logU, neb_xid)
            -> (line_wavelengths, line_lum)

    Returns
    -------
    dict
        Keys:

        - ``predict_lines``: JIT-compiled callable returning
          (wavelengths, luminosities).
        - ``line_wavelengths``: line vacuum wavelengths [Angstrom].

    Notes
    -----
    **JIT-compatible**: yes, the returned function is fully JAX-native.

    **Gradient-safe**: yes, triweight interpolation is fully differentiable.

    """
    line_wavelengths = preint["line_wavelengths"]
    logHB_per_logq = preint["logHB_per_logq"]
    line_ratios = preint["line_ratios"]
    grid_axes = preint["grid_axes"]
    collapsed_axes = preint.get("_collapsed_axes", {})

    # Precompute edges for triweight interpolation
    edges = tuple(edges_for_grid(ax) for ax in grid_axes)

    # If axes were collapsed, build a mapping from remaining free params to grid indices
    fixed_idx_to_value = collapsed_axes

    @jax.jit
    def predict_lines(log_qh, *free_axis_vals, neb_fesc=0.0):
        """Compute Feltre NLR emission lines via triweight interpolation.

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
            if axis_idx in fixed_idx_to_value:
                full_point.append(fixed_idx_to_value[axis_idx])
            else:
                full_point.append(free_axis_vals[free_idx])
                free_idx += 1

        # Interpolate logHB_per_logq and line_ratios at the query point
        logHB_interp = interp_nd_triweight(logHB_per_logq, grid_axes, edges, tuple(full_point))
        ratios_interp = interp_nd_triweight(line_ratios, grid_axes, edges, tuple(full_point))

        # Convert to linear luminosity:
        # L_Hβ [erg/s] = 10^{logHB_per_logq} × Q_H
        # L_line = ratio × L_Hβ × (1 - fesc) / L_sun
        l_hb_erg = (10.0**logHB_interp) * (10.0**log_qh)
        line_lum = ratios_interp * l_hb_erg * (1.0 - neb_fesc) / _LSUN_ERG

        return line_wavelengths, line_lum

    return {
        "predict_lines": predict_lines,
        "line_wavelengths": line_wavelengths,
        "line_weight_matrix": preint["line_weight_matrix"],
    }
