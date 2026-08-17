# SPDX-License-Identifier: BSD-3-Clause
"""Precompute adapter for CB_19 (Charlot & Bruzual 2019) CLOUDY photoionization grid.

Implements :class:`~tengri.forward.precompute.protocol.PrecomputeModule` for
the CB_19 3MdB_17 CLOUDY c17.01 photoionization grid (2,358,330 models),
exposing a 7-axis line-luminosity grid:

    (log_OH, log_age, log_U, log_nH, log_CO, dNO, HbFrac)

The dual (line_lum, continuum_phot) precompute follows the canonical CLOUDY
pattern: emission lines are projected through filter curves via a precomputed
``(n_lines, n_filt)`` weight matrix, enabling fast filter-integrated line flux
lookup at runtime.

**Practical usage**: The full 7-axis grid is ~2.3M points. Memory is manageable
when Fixed axes are collapsed via :func:`slice_fixed_axes` (e.g., fix C/O, ΔN/O,
HbFrac to a nominal subset, leaving ~4 free axes). The ``precompute`` function
auto-detects Fixed axes in Parameters and collapses them before building the
photometry tensor.

References
----------
.. [1] Martinez-Paredes et al. 2023, "CLOUDY c17.01 photoionization models of
       stellar and AGN ionizing sources," MNRAS, arXiv:2308.05604.
       https://doi.org/10.1093/mnras/stad1859
.. [2] Osterbrock & Ferland 2006, "Astrophysics of Gaseous Nebulae and Active
       Galactic Nuclei," 2nd ed., University Science Books. Table 4.4.
.. [3] Byler et al. 2017, "The nebular emission line spectra of the young
       starburst, UV-bright star-forming galaxies," ApJ, 840, 44.

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
from tengri.components.nebular.cloudy_cb19 import (
    _DEFAULT_PATH,
    load_cb19_grid,
)
from tengri.forward.precompute.templates import collapse_fixed_axes
from tengri.utils.grid_interp import (
    PreintegratedGrid,
    interp_nd_triweight,
)
from tengri.utils.interpolation import edges_for_grid

# Axis parameters: ordered tuple matching the CB19 grid axes.
# log_OH: absolute log10(O/H) on CLOUDY c17.01 scale
# log_age: log10(age/yr)
# log_U: log10(ionization parameter)
# log_nH: log10(density / cm⁻³)
# log_CO: log10(C/O)
# dNO: ΔN/O abundance offset
# HbFrac: Hβ fraction (matter-bounded; 1.0 = radiation-bounded)
AXIS_PARAMS: tuple[str, ...] = (
    "log_OH_total",  # internal CB19 axis name
    "log_age_yr_ssp",  # internal axis name (depends on sed_type)
    "log_U",
    "log_nH",
    "log_CO",
    "dNO",
    "HbFrac",
)

# All seven axes are internal grid-axis labels, not user-facing parameters.
# The real grid decision is blocked by #1737 (placeholder tokenization).
# These are declared as internal to prevent spurious warnings when they
# don't match valid parameter names. See issue #1827.
INTERNAL_AXES: frozenset[str] = frozenset(AXIS_PARAMS)


def precompute(
    filter_waves: list,
    filter_trans: list,
    redshift: float,
    parameters: Any,
    *,
    filepath: str | Path | None = None,
    sed_type: str = "SSP",
    imf: str = "Kroupa01",
    mup: float = 100.0,
) -> dict:
    """Build preintegrated CB19 grid, auto-collapsing Fixed axes.

    Loads the CB19 HDF5 grid (or a fixed HbFrac slice), builds the
    ``(n_lines, n_filt)`` projection matrix by integrating each line's Gaussian
    profile through filter curves, then auto-collapses any axis whose
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
    filepath : str or Path, keyword-only
        Path to ``cb19_templates.h5``. If None, defaults to
        ``data/cb19_templates.h5`` relative to package root.
    sed_type : str, keyword-only
        "SSP" (single stellar population, default) or "CSF" (constant SFR).
    imf : str, keyword-only
        "Kroupa01" (default) or "x030".
    mup : float, keyword-only
        Upper stellar mass limit (100.0 or 300.0 M_sun).

    Returns
    -------
    dict
        Keys:

        - ``line_lum_grid``: line-luminosity grid after projecting through
          filters, shape (collapsed_dims..., n_filters). Per-unit Q_H.
        - ``line_weight_matrix``: (n_lines, n_filters) projection matrix.
        - ``line_wavelengths``: (n_lines,) array in vacuum [Angstrom].
        - ``axes``: tuple of remaining (non-collapsed) JAX arrays.
        - ``_preint``: internal PreintegratedGrid metadata.

    Notes
    -----
    **JIT-compatible**: no — this is a build-time function using NumPy.

    **Fixed axes**: The CB19 grid has 7 axes. To avoid memory explosion,
    practical use fixes several (typically C/O, ΔN/O, HbFrac). Pass
    Parameters with Fixed values, and this function auto-detects and
    collapses them via :func:`slice_fixed_axes`.

    **Line profiles**: Lines are integrated as Gaussians with rest-frame FWHM
    of ~50 km/s (narrow lines), or delta functions at line wavelength. The
    ``line_weight_matrix`` encodes the filter-integrated profile for each line.

    """
    if filepath is None:
        filepath = _DEFAULT_PATH

    grid = load_cb19_grid(
        filepath=filepath,
        sed_type=sed_type,
        imf=imf,
        mup=mup,
        hbfrac=1.0,  # Always load HbFrac=1 (radiation-bounded); collapse if needed
    )

    # Build the (n_lines, n_filt) filter projection matrix
    # Each row integrates one emission line (delta function) through filter curves
    n_lines = len(grid.line_wavelengths)
    n_filt = len(filter_waves)
    line_weight_matrix = np.zeros((n_lines, n_filt), dtype=np.float64)

    for i_line, line_wave_aa in enumerate(grid.line_wavelengths):
        # Redshift line wavelength to observed frame
        line_wave_obs = float(line_wave_aa) * (1.0 + redshift)

        # For each filter, integrate the line (delta function) through transmission
        for i_filt, (fw, ft) in enumerate(zip(filter_waves, filter_trans)):
            fw = np.asarray(fw)
            ft = np.asarray(ft)
            # Interpolate filter transmission at line wavelength (delta function)
            line_weight_matrix[i_line, i_filt] = np.interp(
                line_wave_obs, fw, ft, left=0.0, right=0.0
            )

    # Build grid axes from CB19 data
    axes_np = (
        np.asarray(grid.log_OH_grid),
        np.asarray(grid.log_age_grid),
        np.asarray(grid.log_U_grid),
        np.asarray(grid.log_nH_grid),
        np.asarray(grid.log_CO_grid),
        np.asarray(grid.dNO_grid),
        np.array([1.0]),  # HbFrac axis (fixed to 1.0 for now; extend if needed)
    )

    # CB19 has no nebular continuum; use a dummy zero grid
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
    collapsed, remaining_axes, fixed = collapse_fixed_axes(
        preint,
        AXIS_PARAMS,
        parameters,
        internal_axes=INTERNAL_AXES,
        origin="cb19_precompute",
    )

    if fixed:
        return {
            "line_weight_matrix": jnp.asarray(line_weight_matrix),
            "line_wavelengths": jnp.asarray(grid.line_wavelengths),
            "log_line_ratios": grid.log_line_ratios,
            "log_hb_per_qh": grid.log_hb_per_qh,
            "grid_axes": preint.axes,
            "axes": remaining_axes,
            "_preint": collapsed,
            "_collapsed_axes": fixed,
        }

    return {
        "line_weight_matrix": jnp.asarray(line_weight_matrix),
        "line_wavelengths": jnp.asarray(grid.line_wavelengths),
        "log_line_ratios": grid.log_line_ratios,
        "log_hb_per_qh": grid.log_hb_per_qh,
        "grid_axes": preint.axes,
        "axes": preint.axes,
        "_preint": preint,
    }


def build_lookup(preint: dict, **kwargs: Any) -> dict:
    """Build the runtime CB19 line-emission lookup from preintegrated data.

    Returns a callable with signature::

        fn(log_qh, *free_axis_vals) -> (line_wavelengths, line_lum)

    Returns
    -------
    dict
        Keys:

        - ``predict_lines``: JIT-compiled callable returning
          (wavelengths, luminosities).
        - ``line_wavelengths``: line vacuum wavelengths [Angstrom].

    Notes
    -----
    **JIT-compatible**: yes — the returned function is fully JAX-native.

    **Gradient-safe**: yes — triweight interpolation is fully differentiable.

    """
    line_wavelengths = preint["line_wavelengths"]
    log_line_ratios = preint["log_line_ratios"]
    log_hb_per_qh = preint["log_hb_per_qh"]
    grid_axes = preint["grid_axes"]
    collapsed_axes = preint.get("_collapsed_axes", {})

    # Precompute edges for triweight interpolation
    edges = tuple(edges_for_grid(ax) for ax in grid_axes)

    # If axes were collapsed, build a mapping from remaining free params to grid indices
    fixed_idx_to_value = collapsed_axes

    @jax.jit
    def predict_lines(log_qh, *free_axis_vals, neb_fesc=0.0):
        """Compute CB19 emission lines via triweight interpolation.

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

        # Interpolate log_line_ratios at the query point
        log_ratios_interp = interp_nd_triweight(
            log_line_ratios, grid_axes, edges, tuple(full_point)
        )

        # Convert to linear luminosity:
        # L_Hβ [erg/s] = 10^{log_hb_per_qh + log_qh} [erg/s]
        # L_line = 10^{log_ratio} × L_Hβ × (1 - fesc) / L_sun
        log_hb_erg = log_hb_per_qh + log_qh
        hb_erg = 10.0**log_hb_erg
        line_lum = (10.0**log_ratios_interp) * hb_erg * (1.0 - neb_fesc) / _LSUN_ERG

        return line_wavelengths, line_lum

    return {
        "predict_lines": predict_lines,
        "line_wavelengths": line_wavelengths,
    }
