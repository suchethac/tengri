# SPDX-License-Identifier: BSD-3-Clause
"""CLOUDY grid nebular emission backend.

Loads precomputed CLOUDY photoionization grids (from FSPS/Byler+2017 or
Synthesizer) and computes nebular emission as a function of ionization
parameter (logU) and gas metallicity (logZ_gas).

The physical pipeline:
1. SSP spectrum → integrate below 912 A → Q_H (ionizing photon rate)
2. Q_H × grid(logU, logZ, age) → line luminosities + nebular continuum
3. Apply dust (diffuse only, no birth cloud) to nebular emission
4. Add to stellar SED

Grid normalization
------------------
All grids are stored with luminosities per ionizing photon rate Q_H::

    L_line / Q_H  [L_⊙ · s]

At runtime Q_H is computed from the DSPS stellar spectrum by integrating below
912 Å.  This makes the grid independent of IMF or SFR normalization choices.

Note: the BEAGLE/Gutkin+2016 grids (``data/neogal/nebular_emission_Z*.txt``)
store luminosities per unit SFR [L_⊙ / (M_⊙ yr⁻¹)] under a 10^8-yr constant-
SFR assumption.  Those grids are used by ``cloudy_cb19.py`` with an explicit
L_Hβ / Q_H conversion; they are **not** directly compatible with this module.

Comparison with BEAGLE / Gutkin+2016
--------------------------------------
BEAGLE uses Gutkin, Charlot & Bruzual (2016) CLOUDY c13.03 grids computed
with BC03/CB19 ionizing SEDs (Chabrier IMF, constant SFR for 10^8 yr).
This backend (``CloudyGridBackend``) loads CLOUDY grids computed with BPASS
v2.1 binary stellar populations (Byler et al. 2017), re-normalizing Q_H to
the user's DSPS SSPs at runtime.

Key similarities and differences:

+----------------------------+---------------------------------+----------------------------+
| Feature                    | BEAGLE / Gutkin+2016            | CloudyGridBackend          |
+============================+=================================+============================+
| Ionizing SED (grid)        | BC03/CB19 (single star)         | BPASS v2.1 binary          |
+----------------------------+---------------------------------+----------------------------+
| Q_H normalization          | baked-in per SFR                | runtime from DSPS SSPs     |
+----------------------------+---------------------------------+----------------------------+
| Line ratio shape           | fixed to ionizing SED above     | fixed to BPASS v2.1        |
+----------------------------+---------------------------------+----------------------------+
| Age axis                   | absent (10^8 yr average)        | present (per-age grids)    |
+----------------------------+---------------------------------+----------------------------+
| N emission lines           | 18                              | ~128 (FSPS grids)          |
+----------------------------+---------------------------------+----------------------------+
| CLOUDY version             | c13.03                          | c17.01 (FSPS), c23.01      |
|                            |                                 | (Synthesizer test grids)   |
+----------------------------+---------------------------------+----------------------------+
| C/O, N/O axes              | C/O (9 pt), N/O fixed           | absent                     |
+----------------------------+---------------------------------+----------------------------+
| JAX / JIT                  | no                              | yes (triweight interp.)    |
+----------------------------+---------------------------------+----------------------------+

Warning: ``CloudyGridIonizingSpectrumWarning`` is raised when Q_H is re-
normalized but the ionizing SED shape (which drives line ratios) remains BPASS.
If ionizing SED shape variation is scientifically important, use
``CB19Backend`` (CB_19 3MdB_17) or ``CueBackend`` instead.

Relation to Synthesizer stellar grids
--------------------------------------
Synthesizer (Lovell et al. 2025; Roper et al. 2026) provides stellar photoionization grids
alongside its AGN grids.  The stellar grids (``test_grid_sfzh-*.hdf5``)
use CLOUDY c23.01 and cover HII region emission for various SSP models.
Their HDF5 schema uses the same ``axes/`` + ``lines/`` + ``spectra/``
structure as the AGN test grids in ``data/synthesizer_grids/``.

Key differences vs ``CloudyGridBackend`` (BPASS v2.1 / Byler+2017):

+----------------------------+---------------------------+---------------------------+
| Feature                    | CloudyGridBackend          | Synthesizer stellar grids |
+============================+===========================+===========================+
| CLOUDY version             | c17.01 (FSPS grids)       | c23.01 (10 yr newer)      |
+----------------------------+---------------------------+---------------------------+
| N emission lines           | ~128                      | 215                       |
+----------------------------+---------------------------+---------------------------+
| Full spectra stored        | no (lines only)           | yes (9244 λ pts)          |
+----------------------------+---------------------------+---------------------------+
| JAX / JIT in tengri        | yes (triweight interp.)   | not yet integrated        |
+----------------------------+---------------------------+---------------------------+

The tengri ``CloudyGridBackend`` can in principle load any grid with the
same axis / normalization convention.  Synthesizer stellar grids would
require a format adapter (``L_line`` stored in W, not L_⊙/Q_H) before
plugging in.  For age-dependent stellar nebular emission the preferred
tengri backend remains ``CB19Backend`` (c17.01, 2,358,330 models, Q_H-
normalized, JAX-compatible; see ``cloudy_cb19.py``).

References
----------

- Byler et al. 2017, ApJ, 840, 44
- Gutkin, Charlot & Bruzual 2016, MNRAS, 462, 1757
- Chevallard & Charlot 2016, MNRAS, 462, 1415 (BEAGLE)
- Lovell et al. 2025 (doi:10.33232/001c.145766) + Roper et al. 2026 (doi:10.21105/joss.09436)
- diffhtwo (ArgonneCPAC) for JAX grid interpolation patterns

"""

import os
import warnings
from typing import Any, NamedTuple

import h5py
import jax
import jax.numpy as jnp
import numpy as np

from tengri.components.nebular._constants import _LOG10_ZSUN, _LSUN_ERG
from tengri.components.nebular._recombination_coeffs import lyc_dust_escape_factor
from tengri.components.nebular._shared import (
    _interp_index_weight,
    _qh_bilinear,
    compute_qh,
    compute_qh_log10,
    render_nebular_lines,
    sanitize_qh_table,
)
from tengri.utils.interpolation import compute_grid_weights, edges_for_grid
from tengri.utils.scale import pow10

# ── Ionizing-spectrum warnings ────────────────────────────────────


class CloudyGridWNESSPWarning(UserWarning):
    """Deprecated alias retained for backwards compatibility.

    Superseded by ``CloudyGridWNESSPError`` which raises immediately. Kept
    so user code that filters this warning class continues to import.
    """


class CloudyGridWNESSPError(ValueError):
    """Raised when CloudyGridBackend is constructed with a wNE SSP.

    See :class:`tengri.components.nebular.cue.CueWNESSPError` for the same
    failure mode applied to the Cue backend, including which of the two
    checks ``TENGRI_ALLOW_WNE_CLOUDY_GRID=1`` reaches. In short (#1579): it
    downgrades the Q_H heuristic, whose false positives are routine, and
    **not** the ``nebular_included`` metadata check, which has none.

    Resolution: use a bare-stellar SSP, or keep this one and drop the
    ``neb={'type': 'cloudy'}`` group -- its baked-in nebular backend
    already models the lines.
    """


class CloudyGridIonizingSpectrumWarning(UserWarning):
    """Warning: CLOUDY grid ionizing SED shape is BPASS v2.1, not your DSPS SSPs.

    The CLOUDY photoionization grid was computed with BPASS v2.1 binary stellar
    populations as the ionizing source.  Q_H is re-normalized to your DSPS SSPs at
    runtime (correct for stellar mass accounting), but the *shape* of the EUV
    spectrum driving line ratios is fixed to BPASS.  If your SSPs have a
    significantly harder or softer ionizing SED (e.g., single-star models,
    stripped-star prescriptions, non-standard IMF), predicted line ratios will be
    biased.  Use CueBackend if ionizing SED shape variation matters for your science.
    """


# log10(Q_H) threshold for wNE SSP detection (linear, not log10 space).
# Normal bare O/B stars at < 10 Myr: Q_H per Msun ~ 10^47–10^50.
# wNE SSPs: Q_H ≈ 0 after internal absorption.
# Threshold 10^44 gives > 3 dex headroom below the physical floor.
_WNE_QH_THRESHOLD: float = 1e44

# Age cutoff (log10 yr) for young SSP bins used in the wNE check.
_YOUNG_LOG_AGE_MAX_WNE: float = 7.0  # 10 Myr


class CloudyGridData(NamedTuple):
    """Pre-loaded CLOUDY grid data."""

    # Lines
    line_wavelengths: jnp.ndarray  # (n_lines,) rest-frame Angstrom
    line_luminosity: jnp.ndarray  # (n_met, n_age, n_logU, n_lines) Lsun/Q_H
    line_log_met: jnp.ndarray  # (n_met_lines,) log10(Z)
    line_log_age: jnp.ndarray  # (n_age_lines,) log10(age/yr)
    line_log_U: jnp.ndarray  # (n_logU,) log10(U)

    # Continuum
    cont_wavelength: jnp.ndarray  # (n_wave_cont,) Angstrom
    cont_luminosity: jnp.ndarray  # (n_met, n_age, n_logU, n_wave) Lsun_Hz/Q_H
    cont_log_met: jnp.ndarray  # (n_met_cont,) log10(Z)
    cont_log_age: jnp.ndarray  # (n_age_cont,) log10(age/yr)
    cont_log_U: jnp.ndarray  # (n_logU,) shared with lines


def load_cloudy_grid(filepath: str) -> CloudyGridData:
    """Load a tengri-format CLOUDY grid HDF5 file.

    Following FSPS convention, stores luminosities in log10 space
    for interpolation accuracy. A floor of 10^{-95} prevents log(0).

    Metallicity axes are converted from log10(Z/Zsun) (FSPS convention
    in the HDF5 file) to absolute log10(Z) at load time, matching the
    SSP metallicity grid convention used by DSPS.

    Parameters
    ----------
    filepath : str
        Path to cloudy_grid_*.h5 file (from convert_fsps_cloudy_grid.py).

    Returns
    -------
    CloudyGridData
        Pre-loaded grid with line and continuum luminosities in log10 space.

    Notes
    -----
    **JIT-compatible**: no — HDF5 I/O is not JAX-compatible. Call once
    at model initialization and cache the result for repeated use.

    """
    _LOG_FLOOR = 1e-95  # FSPS convention to avoid log(0)

    with h5py.File(filepath, "r") as f:
        line_lum_raw = np.array(f["lines/luminosity"][:])
        cont_lum_raw = np.array(f["continuum/luminosity"][:])

        # Store in log10 space (FSPS convention for interpolation accuracy)
        line_lum_log = np.log10(line_lum_raw + _LOG_FLOOR)
        cont_lum_log = np.log10(cont_lum_raw + _LOG_FLOOR)

        # Convert metallicity from log10(Z/Zsun) → absolute log10(Z)
        line_log_met_abs = np.array(f["lines/axes/log_met"][:]) + _LOG10_ZSUN
        cont_log_met_abs = np.array(f["continuum/axes/log_met"][:]) + _LOG10_ZSUN

        return CloudyGridData(
            line_wavelengths=jnp.array(f["lines/wavelength"][:]),
            line_luminosity=jnp.array(line_lum_log),  # log10 space!
            line_log_met=jnp.array(line_log_met_abs),  # absolute log10(Z)
            line_log_age=jnp.array(f["lines/axes/log_age_yr"][:]),
            line_log_U=jnp.array(f["lines/axes/log_U"][:]),
            cont_wavelength=jnp.array(f["continuum/wavelength"][:]),
            cont_luminosity=jnp.array(cont_lum_log),  # log10 space!
            cont_log_met=jnp.array(cont_log_met_abs),  # absolute log10(Z)
            cont_log_age=jnp.array(f["continuum/axes/log_age_yr"][:]),
            cont_log_U=jnp.array(f["continuum/axes/log_U"][:]),
        )


# ── Q_H computation (ionizing photon rate) ────────────────────────

# Vectorized over metallicity and age dimensions
_compute_qh_grid = jax.vmap(
    jax.vmap(compute_qh, in_axes=(None, 0)),
    in_axes=(None, 0),
)

#: Log-domain sibling. The linear form overflows float32 on healthy input
#: (Q_H ~ 1e46 against a 3.4e38 ceiling), so the table is built from this and
#: stored normalized (#1568).
_compute_log_qh_grid = jax.vmap(
    jax.vmap(compute_qh_log10, in_axes=(None, 0)),
    in_axes=(None, 0),
)


# ── Grid interpolation (trilinear in logZ, logAge, logU) ──────────


def _trilinear_interp(
    data: jnp.ndarray,
    grid_z: jnp.ndarray,
    grid_age: jnp.ndarray,
    grid_u: jnp.ndarray,
    z_val: float,
    age_val: float,
    u_val: float,
) -> jnp.ndarray:
    """Trilinear interpolation on a 3D grid (+ trailing dimensions).

    Parameters
    ----------
    data : array, shape (n_z, n_age, n_u, ...)
        Grid data with 3 leading axes and arbitrary trailing shape.
    grid_z, grid_age, grid_u : array
        Grid axis values.
    z_val, age_val, u_val : float
        Query point.

    Returns
    -------
    array, shape (...)
        Interpolated value.

    """
    iz, wz = _interp_index_weight(z_val, grid_z)
    ia, wa = _interp_index_weight(age_val, grid_age)
    iu, wu = _interp_index_weight(u_val, grid_u)

    # 8 corners of the cube
    c000 = data[iz, ia, iu]
    c001 = data[iz, ia, iu + 1]
    c010 = data[iz, ia + 1, iu]
    c011 = data[iz, ia + 1, iu + 1]
    c100 = data[iz + 1, ia, iu]
    c101 = data[iz + 1, ia, iu + 1]
    c110 = data[iz + 1, ia + 1, iu]
    c111 = data[iz + 1, ia + 1, iu + 1]

    # Interpolate along U
    c00 = c000 * (1 - wu) + c001 * wu
    c01 = c010 * (1 - wu) + c011 * wu
    c10 = c100 * (1 - wu) + c101 * wu
    c11 = c110 * (1 - wu) + c111 * wu

    # Interpolate along age
    c0 = c00 * (1 - wa) + c01 * wa
    c1 = c10 * (1 - wa) + c11 * wa

    # Interpolate along Z
    return c0 * (1 - wz) + c1 * wz


def _trilinear_interp_smooth(
    data: jnp.ndarray,
    grid_z: jnp.ndarray,
    grid_age: jnp.ndarray,
    grid_u: jnp.ndarray,
    z_val: float,
    age_val: float,
    u_val: float,
    scatter: float = 0.2,
    edges_z: jnp.ndarray | None = None,
    edges_age: jnp.ndarray | None = None,
    edges_u: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Smooth triweight-kernel interpolation on a 3-D CLOUDY grid.

    Replaces :func:`_trilinear_interp` for ``grid_interp="triweight"``.
    Returns C²-continuous gradients through grid nodes; no kinks.

    Uses :func:`_shared.compute_grid_weights` on each axis independently,
    then contracts all three weight vectors against the full grid array
    via ``tensordot`` — equivalent to the outer-product weighted sum

        result = Σ_{z,a,u} wz[z] · wa[a] · wu[u] · data[z, a, u, ...]

    The trailing dimensions of ``data`` (e.g. n_lines or n_wave) pass through
    unchanged.

    Parameters
    ----------
    data : array, shape (n_z, n_age, n_u, ...)
        Grid values with 3 leading axes and arbitrary trailing dimensions.
    grid_z, grid_age, grid_u : array
        Sorted axis values.
    z_val, age_val, u_val : float
        Query point.
    scatter : float
        Triweight kernel bandwidth (same units as each axis).  Default 0.2.
    edges_z, edges_age, edges_u : array or None
        Precomputed bin edges from :func:`edges_for_grid`.  When ``None``,
        edges are computed on the fly.

    Returns
    -------
    array, shape (...)
        Interpolated value at the query point.

    Notes
    -----
    **JIT-compatible**: yes — all operations use ``jnp`` primitives.
    **Gradient-safe**: yes — triweight kernel is C²-continuous.

    """
    wz = compute_grid_weights(z_val, grid_z, scatter, edges=edges_z)
    wa = compute_grid_weights(age_val, grid_age, scatter, edges=edges_age)
    wu = compute_grid_weights(u_val, grid_u, scatter, edges=edges_u)
    result = jnp.tensordot(wz, data, axes=([0], [0]))  # (n_age, n_u, ...)
    result = jnp.tensordot(wa, result, axes=([0], [0]))  # (n_u, ...)
    result = jnp.tensordot(wu, result, axes=([0], [0]))  # (...)
    return result


# ── Main backend class ────────────────────────────────────────────


class CloudyGridBackend:
    """CLOUDY grid-based nebular emission backend.

    Loads a precomputed CLOUDY grid and computes nebular emission
    (lines + continuum) at arbitrary (logU, logZ_gas) via grid interpolation.
    Q_H is computed on-the-fly from the SSP spectrum.

    Parameters
    ----------
    grid_path : str
        Path to tengri-format CLOUDY HDF5 grid.
    ssp_data : SSPData
        SSP templates (for Q_H computation).
    grid_interp : {"linear", "triweight"}
        Interpolation mode for the CLOUDY grid axes (logZ_gas, log_age, logU).

        ``"linear"`` (default) — piecewise-linear trilinear interpolation.
        Fast; exact at grid nodes; kinks in the gradient at node boundaries.

        ``"triweight"`` — smooth triweight-kernel interpolation (Hearin et al.
        2023 Eq. 10).  C²-continuous gradients through every node; all three
        axes use the same kernel bandwidth ``grid_scatter``.  Slightly slower
        than linear (~3× tensordot cost vs 8-corner lookup) but fully
        differentiable everywhere.
    grid_scatter : float
        Triweight kernel bandwidth in the natural units of each axis (dex).
        Only used when ``grid_interp="triweight"``.  Default 0.2.

    """

    #: erg/s per [Lsun] for this backend's line catalog (#1559). IAU 2015, the
    #: convention the CLOUDY grid is tabulated in. See CueBackend for the one
    #: backend that deviates.
    lsun_erg: float = _LSUN_ERG

    def __init__(
        self,
        grid_path: str,
        ssp_data=None,
        grid_interp: str = "linear",
        grid_scatter: float = 0.2,
        ionizing_source_warning: str = "warn",
    ) -> None:
        if grid_interp not in ("linear", "triweight"):
            raise ValueError(f"grid_interp must be 'linear' or 'triweight', got {grid_interp!r}")
        if ionizing_source_warning not in ("raise", "warn", "suppress"):
            raise ValueError(
                "ionizing_source_warning must be 'raise', 'warn', or 'suppress', "
                f"got {ionizing_source_warning!r}"
            )
        if ionizing_source_warning != "suppress":
            msg = (
                "CloudyGridBackend: Q_H is computed from your DSPS SSPs (correct for "
                "stellar mass accounting), but the ionizing spectral shape used to "
                "compute the CLOUDY grid was BPASS v2.1 binary stars. If your SSPs "
                "have a significantly harder/softer ionizing SED than BPASS (e.g., "
                "single-star models, stripped-star prescriptions, very young/old "
                "populations), predicted line ratios will be biased. Use CueBackend "
                "if ionizing SED shape variation matters for your science. "
                "To suppress: pass ionizing_source_warning='suppress'."
            )
            if ionizing_source_warning == "raise":
                raise ValueError(msg)
            warnings.warn(msg, CloudyGridIonizingSpectrumWarning, stacklevel=2)
        self.name = "cloudy_grid"
        self.has_free_params = True
        self.has_continuum = True
        self._grid_interp = grid_interp
        self._grid_scatter = grid_scatter
        self.grid = load_cloudy_grid(grid_path)

        # Max age for nebular emission: 100 Myr (conservative).
        # CLOUDY grid stops at ~20 Myr, but Q_H is non-negligible up to
        # ~100 Myr from post-AGB/HB stars. Beyond 100 Myr, Q_H drops
        # >6 orders of magnitude below peak — safe to ignore.
        self._max_neb_log_age = 8.0  # log10(100 Myr in yr)

        # Precompute triweight bin edges (static grid, avoids rebuilding in JIT)
        if grid_interp == "triweight":
            self._edges_z_line = edges_for_grid(self.grid.line_log_met)
            self._edges_age_line = edges_for_grid(self.grid.line_log_age)
            self._edges_u_line = edges_for_grid(self.grid.line_log_U)
            self._edges_z_cont = edges_for_grid(self.grid.cont_log_met)
            self._edges_age_cont = edges_for_grid(self.grid.cont_log_age)
            self._edges_u_cont = edges_for_grid(self.grid.cont_log_U)

        # Precompute Q_H table and young-age index from SSP if provided
        self._qh_table = None
        #: log10 of the scalar ``_qh_table`` is normalized by (#1568); 0.0 means
        #: "already in photons/s", which is the identity for a missing table.
        self._log_qh_scale = 0.0
        self._young_idx = None  # indices of SSP age bins with nebular emission
        if ssp_data is not None:
            self._precompute_qh(ssp_data)

        # Photometry preintegration storage
        self._preint_continuum = None
        self._preint_lines = None
        self._has_preint_photometry = False

    def preintegrate_for_photometry(
        self,
        filter_waves: list,
        filter_trans: list,
        redshift: float,
        dl_cm: float,
        fixed: dict[int, float] | None = None,
    ) -> None:
        """Preintegrate CLOUDY continuum + lines through photometric filters.

        After calling this, the backend can compute nebular photometry
        via fast grid interpolation instead of full-wavelength evaluation.

        The continuum grid is converted from log10(Lsun_Hz/Q_H) to linear
        Lsun_Hz/Q_H, then preintegrated through filters. The line wavelengths
        are point-sampled through filters for exact line contributions.

        Results stored in:

        - self._preint_continuum: PreintegratedGrid (n_met, n_age, n_logU, n_filters)
        - self._preint_lines: PreintegratedLines (n_lines, n_filters)

        If any axes are fixed, they are collapsed via triweight interpolation
        at initialization time, reducing grid dimensionality.

        Parameters
        ----------
        filter_waves : list
            List of filter wavelength arrays (Angstrom).
        filter_trans : list
            List of filter transmission curves.
        redshift : float
            Redshift for redshifting observed-frame wavelengths.
        dl_cm : float
            Luminosity distance (cm).
        fixed : dict[int, float], optional
            Mapping of axis index → fixed value. Axes are numbered from 0:

            - 0: log_met (metallicity)
            - 1: log_age (age in years)
            - 2: log_U (ionization parameter)

            If provided, these axes are collapsed at init time. Default None.

        """
        from tengri.utils.grid_interp import (
            preintegrate_grid,
            preintegrate_lines,
            slice_fixed_axes,
        )

        # Convert continuum from log10 to linear Lsun_Hz/Q_H.
        # The CLOUDY grid uses a floor of log10 = -95 for zero luminosity.
        # Clip to zero below -90 to avoid 1e-95 polluting the filter integral.
        cont_log = np.asarray(self.grid.cont_luminosity)
        cont_linear = np.where(cont_log > -90.0, 10.0**cont_log, 0.0)

        # Preintegrate continuum through filters
        self._preint_continuum = preintegrate_grid(
            cont_linear,
            np.asarray(self.grid.cont_wavelength),
            filter_waves,
            filter_trans,
            redshift,
            dl_cm,
            axes=(
                np.asarray(self.grid.cont_log_met),
                np.asarray(self.grid.cont_log_age),
                np.asarray(self.grid.cont_log_U),
            ),
        )

        # Preintegrate lines through filters
        self._preint_lines = preintegrate_lines(
            np.asarray(self.grid.line_wavelengths),
            filter_waves,
            filter_trans,
            redshift,
            axes=(
                np.asarray(self.grid.line_log_met),
                np.asarray(self.grid.line_log_age),
                np.asarray(self.grid.line_log_U),
            ),
        )

        # Collapse fixed axes if provided
        if fixed:
            self._preint_continuum = slice_fixed_axes(self._preint_continuum, fixed)

            # Also collapse line_luminosity (n_Z, n_age, n_logU, n_lines) along fixed axes.
            # slice_fixed_axes for PreintegratedLines only updates axes/edges metadata; the
            # actual luminosity grid used by interp_nd_triweight in assembly.py must have the
            # same number of leading dimensions as len(axes). We apply the same triweight
            # contraction here at init time so the two are always in sync.
            line_lum = jnp.asarray(self.grid.line_luminosity)  # (n_Z, n_age, n_logU, n_lines)
            line_axes = [
                jnp.asarray(self.grid.line_log_met),
                jnp.asarray(self.grid.line_log_age),
                jnp.asarray(self.grid.line_log_U),
            ]
            for axis_idx in sorted(fixed.keys(), reverse=True):
                value = fixed[axis_idx]
                ax = line_axes[axis_idx]
                scatter = 0.5 * float(ax[1] - ax[0])
                w = compute_grid_weights(value, ax, scatter=scatter, edges=edges_for_grid(ax))
                # tensordot contracts axis `axis_idx` of line_lum with axis 0 of w.
                # The resulting tensor has all axes of line_lum except axis_idx.
                line_lum = jnp.tensordot(w, line_lum, axes=([0], [axis_idx]))
                line_axes.pop(axis_idx)
            self._line_lum_collapsed = line_lum
        else:
            self._line_lum_collapsed = jnp.asarray(self.grid.line_luminosity)

        # Set flag
        self._has_preint_photometry = True

    def _precompute_qh(self, ssp_data) -> None:
        """Precompute Q_H(metallicity, age) table from SSP spectra.

        This avoids recomputing the ionizing integral at every inference step.
        """
        # Metadata check (#1014): a grid flagged nebular-included is refused
        # outright, BEFORE the Q_H heuristic below — the retained-LyC wNE
        # class keeps its ionizing continuum, so no physics heuristic can
        # catch it. The flag comes from the ``nebular_included`` HDF5
        # attribute or the wNE filename convention via ``load_ssp_data``.
        if getattr(ssp_data, "nebular", "unknown") == "included":
            # Deliberately NOT bypassable by TENGRI_ALLOW_WNE_CLOUDY_GRID,
            # for the reason spelled out in CueBackend (#1579): that switch
            # covers the Q_H *heuristic* below, which has routine false
            # positives on synthetic grids. This branch is a *declaration*
            # read from metadata, with no false-positive mode, so a bypass
            # here can only ever hide a real double-count.
            raise CloudyGridWNESSPError(
                "CloudyGridBackend received an SSP flagged nebular-included "
                "(wNE): nebular continuum and lines are already baked into "
                "the templates, so adding a CLOUDY grid on top double-counts "
                "nebular emission. Fix: use a bare-stellar SSP (e.g. "
                "fsps_prsc_miles_chabrier.h5), or keep this SSP and drop the "
                "neb={'type': 'cloudy'} group — the baked-in backend already "
                "models the lines."
            )

        ssp_wave = ssp_data.ssp_wave
        ssp_flux = ssp_data.ssp_flux  # (n_met, n_age, n_wave)

        # Compute Q_H for each (met, age) — vectorized, in the log domain and
        # stored normalized by its own peak (#1568). Q_H reaches ~1e46
        # photons/s/Msun; the linear build overflowed every entry to ``inf`` in
        # float32 and ``sanitize_qh_table`` then rewrote the lot to 0.0, so
        # every CloudyGrid line and the whole nebular continuum came out
        # silently zero. Same defect and same fix as CB19.
        #
        # Bilinear interpolation is linear, so interpolating ``table / scale``
        # is exactly interpolating ``table`` and dividing — float64 unchanged.
        log_qh_raw = _compute_log_qh_grid(ssp_wave, ssp_flux)
        finite = jnp.isfinite(log_qh_raw)
        self._log_qh_scale = float(jnp.max(jnp.where(finite, log_qh_raw, -jnp.inf)))
        if not np.isfinite(self._log_qh_scale):
            self._log_qh_scale = 0.0
        # Sanitize: replace Inf/NaN with 0 (can arise from SSP grids
        # with incomplete UV coverage or numerical overflow in the
        # ionizing photon integral).
        self._qh_table = sanitize_qh_table(
            pow10(log_qh_raw - self._log_qh_scale), backend_name="CloudyGridBackend"
        )
        # Store as JAX arrays so dynamic indexing works inside jax.grad/vmap
        self._qh_log_met = jnp.asarray(ssp_data.ssp_lgmet)
        self._qh_log_age = jnp.asarray(ssp_data.ssp_lg_age_gyr + 9.0)  # log(age/yr)

        # Precompute indices of young SSP age bins (only these produce
        # ionizing photons and contribute to nebular emission)
        ssp_log_ages = np.array(self._qh_log_age)
        young_mask = ssp_log_ages <= self._max_neb_log_age
        self._young_idx = np.where(young_mask)[0]
        self._n_young = len(self._young_idx)

        # wNE SSP detection: bare O/B stars at < 10 Myr give Q_H ~ 10^47–10^50
        # per Msun.  If all young bins have Q_H < 10^44, the SSP likely has
        # baked-in nebular emission (wNE) and predictions will be unreliable.
        very_young_mask = ssp_log_ages <= _YOUNG_LOG_AGE_MAX_WNE
        if very_young_mask.any():
            # Compare in log space against the *absolute* Q_H — ``_qh_table`` is
            # peak-normalized now, so its raw max is ~1 and would trip this
            # threshold for every SSP (#1568).
            log_qh_young = np.array(log_qh_raw)[:, very_young_mask]
            qh_young_max = 10.0 ** float(np.nanmax(log_qh_young))
            qh_young = np.array([qh_young_max])  # for the message below
            if qh_young_max < _WNE_QH_THRESHOLD:
                msg = (
                    "CloudyGridBackend received a wNE (with-Nebular-Emission) "
                    f"SSP. Max Q_H for bins younger than 10 Myr is "
                    f"{float(qh_young.max()):.2e}, well below the ~10^47-10^50 "
                    "floor for bare stellar populations. Nebular luminosities "
                    "will be under-predicted by 4-7 dex. Fix: use a "
                    "bare-stellar SSP (e.g. data/fsps_prsc_miles_chabrier.h5). "
                    "To bypass for testing, set TENGRI_ALLOW_WNE_CLOUDY_GRID=1 "
                    "(downgrades to a warning)."
                )
                if os.environ.get("TENGRI_ALLOW_WNE_CLOUDY_GRID"):
                    warnings.warn(msg, CloudyGridWNESSPWarning, stacklevel=3)
                else:
                    raise CloudyGridWNESSPError(msg)

    def _get_qh_at(
        self,
        log_z: float,
        log_age_yr: float,
    ) -> float:
        """Get Q_H at a specific (logZ, logAge) via interpolation."""
        return _qh_bilinear(
            self._qh_table,
            self._qh_log_met,
            self._qh_log_age,
            log_z,
            log_age_yr,
            missing=0.0,
        )

    def _make_interp_fn(
        self,
        data: jnp.ndarray,
        grid_z: jnp.ndarray,
        grid_age: jnp.ndarray,
        grid_u: jnp.ndarray,
        edges_z: jnp.ndarray | None = None,
        edges_age: jnp.ndarray | None = None,
        edges_u: jnp.ndarray | None = None,
    ):
        """Build an interpolation closure for the configured grid_interp mode."""
        if self._grid_interp == "triweight":
            s = self._grid_scatter

            def _interp(z, a, u):
                """Interpolate grid data with smooth tricubic filtering."""
                return _trilinear_interp_smooth(
                    data,
                    grid_z,
                    grid_age,
                    grid_u,
                    z,
                    a,
                    u,
                    s,
                    edges_z=edges_z,
                    edges_age=edges_age,
                    edges_u=edges_u,
                )
        else:

            def _interp(z, a, u):
                """Interpolate grid data via trilinear interpolation."""
                return _trilinear_interp(data, grid_z, grid_age, grid_u, z, a, u)

        return _interp

    def predict_nebular_line_luminosities(
        self,
        ssp_weights: jnp.ndarray,
        ssp_log_ages_yr: jnp.ndarray,
        log_z: float,
        neb_logU: float = -3.0,
        neb_logZ_gas: float | None = None,
        neb_fesc: float = 0.0,
        neb_fesc_lya: float = 0.0,
        neb_fdust: float = 0.0,
        template_data: Any | None = None,
        **_kwargs,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        r"""Compute emission line luminosities (vectorized over age bins).

        L_line = sum_i [w_i * Q_H(Z, age_i) * grid(Z_gas, age_i, logU) * k(f_esc, f_dust)]

        where the k-factor accounts for ionizing photon escape and dust absorption:

        .. math::

            k = \frac{1 - f_\mathrm{esc} - f_\mathrm{dust}}
                     {1 + \dfrac{\alpha_1}{\alpha_B}\,(f_\mathrm{esc} + f_\mathrm{dust})}

        Ly-alpha (1215.67 A) is treated separately: its luminosity is scaled
        by (1-neb_fesc_lya)/(1-k*fesc) relative to other lines, reflecting
        resonant scattering that suppresses Ly-alpha escape independently.

        Parameters
        ----------
        ssp_weights : array, shape (n_age,)
            CSP mass weights [Msun per age bin].
        ssp_log_ages_yr : array, shape (n_age,)
            log10(age/yr) of SSP age bins [log10(yr)].
        log_z : float
            Stellar metallicity log10(Z) (absolute) [log10(Z)].
        neb_logU : float
            Ionization parameter log10(U) [log10(U)]. Default -3.0.
        neb_logZ_gas : float or None
            Gas metallicity log10(Z) absolute [log10(Z)]. None = tie to stellar Z.
        neb_fesc : float
            Ionizing photon escape fraction [dimensionless, in [0, 1]]. Default 0.0.
        neb_fesc_lya : float
            Ly-alpha-specific escape fraction [dimensionless, in [0, 1]]. Default 0.0.
        neb_fdust : float
            Lyman-continuum dust-absorption fraction in HII regions
            [dimensionless, in [0, 1]]. Default 0.0. Both ``neb_fesc`` and
            ``neb_fdust`` reduce the ionizing photon budget via the CIGALE
            k-factor.

        Returns
        -------
        wavelengths : array, shape (n_lines,)
            Rest-frame vacuum wavelengths [Angstrom].
        luminosities : array, shape (n_lines,)
            Emission line luminosities [Lsun].

        Notes
        -----
        **JIT-compatible**: yes — all operations use ``jnp`` primitives.

        **Gradient-safe**: yes — differentiable through neb_logU, neb_fesc, and
        neb_fdust.

        **k-factor**: follows CIGALE nebular.py (Ferland 1980) with ionizing
        photon loss due to both escape and dust absorption treated symmetrically.

        References
        ----------
        .. [1] Ferland, G. J. 1980, PASP, 92, 596.
        .. [2] Inoue, A. K. 2011, MNRAS, 415, 2920.
        .. [3] CIGALE nebular module: pcigale/sed_modules/nebular.py, lines
            156-162.

        """
        if neb_logZ_gas is None:
            neb_logZ_gas = log_z

        grid = template_data if template_data is not None else self.grid

        # Only young SSP age bins contribute (age < ~20 Myr)
        # Slice to young bins only — 93 → ~10 bins, ~10x less work
        young_idx = self._young_idx
        young_ages = ssp_log_ages_yr[young_idx]
        young_weights = ssp_weights[young_idx]

        _interp_lines = self._make_interp_fn(
            grid.line_luminosity,
            grid.line_log_met,
            grid.line_log_age,
            grid.line_log_U,
            edges_z=getattr(self, "_edges_z_line", None),
            edges_age=getattr(self, "_edges_age_line", None),
            edges_u=getattr(self, "_edges_u_line", None),
        )

        # Compute k-factor once (shared by all age bins)
        k_factor = lyc_dust_escape_factor(neb_fesc, neb_fdust)

        def _line_contrib_one_age(log_age_i, weight_i):
            """Compute weighted line luminosity contribution for one SSP age bin."""
            qh_i = self._get_qh_at(log_z, log_age_i)
            log_lum_per_qh = _interp_lines(neb_logZ_gas, log_age_i, neb_logU)
            # ``qh_i`` is peak-normalized and ``log_lum_per_qh`` is ~-46, so the
            # scale goes back in *inside* the exponent: the sum is O(1) and
            # neither 1e-46 nor 1e46 ever exists as a float32 array (#1568).
            return weight_i * qh_i * pow10(log_lum_per_qh + self._log_qh_scale) * k_factor

        # vmap over young age bins only, then sum
        all_contribs = jax.vmap(_line_contrib_one_age)(
            young_ages, young_weights
        )  # (n_young, n_lines)

        total_line_lum = jnp.sum(all_contribs, axis=0)  # (n_lines,)

        # Apply differential Ly-alpha escape fraction.
        # Ly-alpha at 1215.67 A: scale by (1-fesc_lya)/k_factor to apply the
        # Ly-alpha-specific escape on top of the k-factor already applied.
        lya_idx = jnp.argmin(jnp.abs(grid.line_wavelengths - 1215.67))
        lya_scale = (1.0 - neb_fesc_lya) / jnp.maximum(k_factor, 1e-10)
        total_line_lum = total_line_lum.at[lya_idx].multiply(lya_scale)

        return grid.line_wavelengths, total_line_lum

    def predict_nebular_continuum(
        self,
        ssp_weights: jnp.ndarray,
        ssp_log_ages_yr: jnp.ndarray,
        log_z: float,
        neb_logU: float = -3.0,
        neb_logZ_gas: float | None = None,
        neb_fesc: float = 0.0,
        neb_fdust: float = 0.0,
        template_data: Any | None = None,
        **_kwargs,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        r"""Compute nebular continuum SED (vectorized over age bins).

        Scales the nebular continuum by the CIGALE k-factor to account for
        ionizing photon escape and dust absorption:

        .. math::

            k = \frac{1 - f_\mathrm{esc} - f_\mathrm{dust}}
                     {1 + \dfrac{\alpha_1}{\alpha_B}\,(f_\mathrm{esc} + f_\mathrm{dust})}

        Parameters
        ----------
        ssp_weights : array, shape (n_age,)
            CSP mass weights [Msun per age bin].
        ssp_log_ages_yr : array, shape (n_age,)
            log10(age/yr) of SSP age bins [log10(yr)].
        log_z : float
            Stellar metallicity log10(Z) absolute [log10(Z)].
        neb_logU : float
            Ionization parameter log10(U) [log10(U)]. Default -3.0.
        neb_logZ_gas : float or None
            Gas metallicity log10(Z) absolute [log10(Z)]. None → tied to stellar.
        neb_fesc : float
            Ionizing photon escape fraction [dimensionless, in [0, 1]]. Default 0.0.
        neb_fdust : float
            Lyman-continuum dust-absorption fraction in HII regions
            [dimensionless, in [0, 1]]. Default 0.0. Both ``neb_fesc`` and
            ``neb_fdust`` reduce the ionizing photon budget via the CIGALE
            k-factor.
        **_kwargs
            Additional keyword arguments (unused).

        Returns
        -------
        wavelength : array, shape (n_wave_cont,)
            Continuum wavelengths [Angstrom].
        luminosity : array, shape (n_wave_cont,)
            Nebular continuum L_nu [L_sun/Hz].

        References
        ----------
        .. [1] N. Byler et al., "Nebular Continuum and Line Emission in Stellar
           Population Synthesis Models," ApJ, 840, 44 (2017).
           https://doi.org/10.3847/1538-4357/aa6c66
        .. [2] Ferland, G. J. 1980, PASP, 92, 596.
        .. [3] Inoue, A. K. 2011, MNRAS, 415, 2920.
        .. [4] CIGALE nebular module: pcigale/sed_modules/nebular.py, lines
            156-162.

        Notes
        -----
        **JIT-compatible**: yes — all operations use ``jnp`` primitives.

        **Gradient-safe**: yes — differentiable through neb_logU, neb_fesc, and
        neb_fdust.

        """
        if neb_logZ_gas is None:
            neb_logZ_gas = log_z

        grid = template_data if template_data is not None else self.grid

        # Only young age bins
        young_idx = self._young_idx
        young_ages = ssp_log_ages_yr[young_idx]
        young_weights = ssp_weights[young_idx]

        _interp_cont = self._make_interp_fn(
            grid.cont_luminosity,
            grid.cont_log_met,
            grid.cont_log_age,
            grid.cont_log_U,
            edges_z=getattr(self, "_edges_z_cont", None),
            edges_age=getattr(self, "_edges_age_cont", None),
            edges_u=getattr(self, "_edges_u_cont", None),
        )

        # Compute k-factor once (shared by all age bins)
        k_factor = lyc_dust_escape_factor(neb_fesc, neb_fdust)

        def _cont_contrib_one_age(log_age_i, weight_i):
            """Compute weighted nebular continuum contribution for one SSP age bin."""
            qh_i = self._get_qh_at(log_z, log_age_i)
            log_cont_per_qh = _interp_cont(neb_logZ_gas, log_age_i, neb_logU)
            # Same seam as the line channel: fold the Q_H normalization back
            # in inside the exponent rather than materializing it (#1568).
            return weight_i * qh_i * pow10(log_cont_per_qh + self._log_qh_scale) * k_factor

        all_contribs = jax.vmap(_cont_contrib_one_age)(
            young_ages, young_weights
        )  # (n_young, n_wave_cont)

        total_cont = jnp.sum(all_contribs, axis=0)
        return grid.cont_wavelength, total_cont

    def predict_nebular_sed(
        self,
        ssp_weights: jnp.ndarray,
        ssp_wave: jnp.ndarray,
        ssp_log_ages_yr: jnp.ndarray,
        log_z: float,
        neb_logU: float = -3.0,
        neb_logZ_gas: float | None = None,
        neb_fesc: float = 0.0,
        neb_fesc_lya: float = 0.0,
        neb_fdust: float = 0.0,
        line_sigma_aa: float = 0.0,
        line_sigma_kms: float = 0.0,
        template_data: Any | None = None,
        **_kwargs,
    ) -> jnp.ndarray:
        r"""Compute total nebular emission on the SSP wavelength grid.

        Combines emission lines (as delta functions or Gaussians) with
        nebular continuum, interpolated onto the SSP wavelength grid. Both
        components are scaled by the CIGALE k-factor:

        .. math::

            k = \frac{1 - f_\mathrm{esc} - f_\mathrm{dust}}
                     {1 + \dfrac{\alpha_1}{\alpha_B}\,(f_\mathrm{esc} + f_\mathrm{dust})}

        Parameters
        ----------
        ssp_weights : array, shape (n_age,)
            CSP mass weights [Msun per age bin].
        ssp_wave : array, shape (n_wave,)
            SSP wavelength grid [Angstrom].
        ssp_log_ages_yr : array, shape (n_age,)
            log10(age/yr) of SSP bins [log10(yr)].
        log_z : float
            Stellar metallicity log10(Z) absolute [log10(Z)].
        neb_logU : float
            Ionization parameter log10(U) [log10(U)]. Default -3.0.
        neb_logZ_gas : float or None
            Gas metallicity log10(Z) absolute [log10(Z)]. None = tie to stellar.
        neb_fesc : float
            Ionizing photon escape fraction [dimensionless, in [0, 1]]. Default 0.0.
        neb_fesc_lya : float
            Ly-alpha-specific escape fraction [dimensionless, in [0, 1]]. Default 0.0.
        neb_fdust : float
            Lyman-continuum dust-absorption fraction in HII regions
            [dimensionless, in [0, 1]]. Default 0.0.
        line_sigma_aa : float
            Gaussian line width (σ) [Angstrom]. 0 = delta function
            (add to nearest pixel).

        Returns
        -------
        array, shape (n_wave,)
            Nebular SED [erg/s/Hz] on the SSP wavelength grid.

        References
        ----------
        .. [1] N. Byler et al., "Nebular Continuum and Line Emission in Stellar
           Population Synthesis Models," ApJ, 840, 44 (2017).
           https://doi.org/10.3847/1538-4357/aa6c66
        .. [2] Ferland, G. J. 1980, PASP, 92, 596.
        .. [3] Inoue, A. K. 2011, MNRAS, 415, 2920.
        .. [4] CIGALE nebular module: pcigale/sed_modules/nebular.py, lines
            156-162.

        Notes
        -----
        **JIT-compatible**: yes — all operations use ``jnp`` primitives.

        **Gradient-safe**: yes — differentiable through neb_logU, neb_fesc, and
        neb_fdust.

        """
        # Get line luminosities
        line_wave, line_lum = self.predict_nebular_line_luminosities(
            ssp_weights,
            ssp_log_ages_yr,
            log_z,
            neb_logU=neb_logU,
            neb_logZ_gas=neb_logZ_gas,
            neb_fesc=neb_fesc,
            neb_fesc_lya=neb_fesc_lya,
            neb_fdust=neb_fdust,
            template_data=template_data,
        )

        # Get continuum
        cont_wave, cont_lum = self.predict_nebular_continuum(
            ssp_weights,
            ssp_log_ages_yr,
            log_z,
            neb_logU=neb_logU,
            neb_logZ_gas=neb_logZ_gas,
            neb_fesc=neb_fesc,
            neb_fdust=neb_fdust,
            template_data=template_data,
        )

        # Interpolate continuum onto SSP wavelength grid
        neb_sed = jnp.interp(ssp_wave, cont_wave, cont_lum, left=0.0, right=0.0)

        # Add emission lines
        neb_sed = neb_sed + render_nebular_lines(
            jnp.asarray(line_wave), jnp.asarray(line_lum), ssp_wave, line_sigma_aa, line_sigma_kms
        )

        # Convert from internal Lsun/Hz to erg/s/Hz
        return neb_sed * _LSUN_ERG
