# SPDX-License-Identifier: BSD-3-Clause
"""CB_19 (Charlot & Bruzual 2019) CLOUDY photoionization grid backend.

Source
------
Martinez-Paredes et al. 2023 (arXiv:2308.05604), stored in the 3MdB_17 database
(ref='CB_19'). 2,358,330 models computed with CLOUDY c17.01 using C&B 2019 SSP and
CSF ionizing SEDs. Grid axes: log(O/H), log_age, logU, log_nH, log(C/O), ΔN/O, HbFrac.

Unit convention: Hβ ratios → L_sun/Q_H
---------------------------------------
CB_19 stores all emission line fluxes as dimensionless ratios relative to Hβ::

    stored value = L_line / L_Hβ

The tengri SED pipeline requires L_line/Q_H (Lsun per photon/s) to be compatible
with the ionizing-photon-weighted summation used by CloudyGridBackend::

    L_line / Q_H = (L_line / L_Hβ) × (L_Hβ / Q_H)

where the Case B conversion factor is (Osterbrock & Ferland 2006, Table 4.4;
T_e = 10^4 K, n_e = 10^2 cm⁻³; also eq. 1 of Byler et al. 2017 ApJ 840 44)::

    L_Hβ / Q_H = 4.78 × 10⁻¹³ erg photon⁻¹
               = 4.78 × 10⁻¹³ / 3.828 × 10³³ L_sun s photon⁻¹
               ≈ 1.249 × 10⁻⁴⁶ L_sun s photon⁻¹

This constant is stored in `_HB_PER_QH_LSUN` and in the HDF5 file's root attrs
(key: ``hb_per_qh_lsun``) for reproducibility.

Metallicity convention
----------------------
CB_19 stores metallicity as log10(O/H) on the CLOUDY c17.01 solar scale where
12 + log(O/H)_sun ≈ 8.93 (log(O/H)_sun ≈ −3.07). tengri's internal metallicity
convention is absolute log10(Z) with log10(Zsun) = −1.848 (Asplund+2009).

The conversion used internally (parameter ``_LOG_OH_OFFSET``) is::

    log_OH_total = neb_logZ_gas + _LOG_OH_OFFSET
    _LOG_OH_OFFSET = _LOG_OH_SOLAR − _LOG10_ZSUN ≈ −3.07 − (−1.848) = −1.222

where ``neb_logZ_gas`` is absolute log10(Z). This assumes O/H scales linearly
with total metallicity Z (solar abundance ratios).

HbFrac and matter-bounded nebulae
----------------------------------
HbFrac = L_Hβ(matter-bounded) / L_Hβ(radiation-bounded).
HbFrac = 1.0 → radiation-bounded (default); HbFrac < 1 → matter-bounded,
with ionizing photon escape fraction ≈ 1 − HbFrac. The HbFrac axis is treated
as discrete (nearest-neighbor snap at init time via ``hbfrac`` argument).

Comparison with BEAGLE (Gutkin+2016)
-------------------------------------
BEAGLE (Chevallard & Charlot 2016) uses the Gutkin, Charlot & Bruzual (2016)
CLOUDY c13.03 grids for stellar-photoionized nebular emission.  The CB_19
backend is the closest tengri equivalent, sharing the same ionizing stellar
population model (CB19 SSPs), but differs in several important ways:

+---------------------------+------------------------------+------------------------------+
| Feature                   | BEAGLE / Gutkin+2016         | tengri / CB_19               |
+===========================+==============================+==============================+
| CLOUDY version            | c13.03 (2013)                | c17.01 (2017); updated       |
|                           |                              | atomic/recombination data    |
+---------------------------+------------------------------+------------------------------+
| Ionizing stellar model    | BC03 + CB19 (Chabrier IMF)   | CB19 SSP + CSF               |
+---------------------------+------------------------------+------------------------------+
| log U range               | -4.0 to -1.0 (7 points)     | full 3MdB_17 coverage        |
+---------------------------+------------------------------+------------------------------+
| Metallicity axis          | Z = 0.0001–0.04 (14 Z)      | 12+log(O/H) axis, wider      |
+---------------------------+------------------------------+------------------------------+
| n_H axis                  | 2 points (100, 1000 cm⁻³)   | continuous axis              |
+---------------------------+------------------------------+------------------------------+
| C/O axis                  | 9 points (0.1–1.4 × solar)  | log(C/O) axis, continuous    |
+---------------------------+------------------------------+------------------------------+
| N/O axis                  | fixed solar                  | ΔN/O axis                    |
+---------------------------+------------------------------+------------------------------+
| Stellar age axis          | **absent** (10^8 yr CSF)     | **present** (full age range) |
+---------------------------+------------------------------+------------------------------+
| Ionizing photon escape    | absent                       | HbFrac axis (matter-bounded) |
+---------------------------+------------------------------+------------------------------+
| N model points            | ~148,000 total               | 2,358,330                    |
+---------------------------+------------------------------+------------------------------+
| Output normalization      | L/SFR [L_⊙ / (M_⊙ yr⁻¹)]   | L_line/Q_H [L_⊙ · s]         |
+---------------------------+------------------------------+------------------------------+
| JAX / JIT compatible      | no                           | yes (triweight interp.)      |
+---------------------------+------------------------------+------------------------------+

The BEAGLE normalization (L per unit SFR) bakes in a constant SFR assumption
for 10^8 yr with a Chabrier IMF.  CB_19's Q_H normalization is more flexible:
the ionizing photon rate Q_H is computed at runtime from the DSPS SFH, allowing
the line emission to track the actual current ionizing flux for any SFH shape.

Critically, the BEAGLE/Gutkin grid has **no stellar age axis**: it was computed
for a constant SFR averaged over 10^8 yr.  CB_19 explicitly tracks how the
ionizing SED hardens/softens with stellar age, which matters for galaxies
undergoing rapid bursts or quenching.  This makes CB_19 the preferred backend
for stochastic SFH science.

Raw data: ``data/neogal/nebular_emission_Z*.txt`` (NEOGAL ASCII, 14 Z files,
downloaded from ``http://www.iap.fr/neogal/``).

Relation to Synthesizer stellar grids
--------------------------------------
Synthesizer (Lovell et al. 2025; Roper et al. 2026) ships stellar photoionization grids
(``test_grid_sfzh-*.hdf5``) produced with CLOUDY c23.01, a decade newer
than CB_19 (c17.01).  Their HDF5 schema stores ``lines/luminosity`` in W
per bolometric luminosity, normalized to ``bolometric_luminosities`` weight
variables — a different convention from CB_19's L_line/Q_H normalization.

Key comparison:

+---------------------------+---------------------------+---------------------------+
| Feature                   | CB_19 (3MdB_17)           | Synthesizer stellar grids |
+===========================+===========================+===========================+
| CLOUDY version            | c17.01 (2017)             | c23.01 (2023)             |
+---------------------------+---------------------------+---------------------------+
| N emission lines          | 7 (stored; full grid has  | 215                       |
|                           | many more in 3MdB_17)     |                           |
+---------------------------+---------------------------+---------------------------+
| Stellar age axis          | yes (full age range)      | yes                       |
+---------------------------+---------------------------+---------------------------+
| C/O, N/O axes             | yes (log(C/O), ΔN/O)      | no (fixed solar)          |
+---------------------------+---------------------------+---------------------------+
| N model points            | 2,358,330                 | 2-pt test grid only       |
|                           |                           | (prod. grids via Box)     |
+---------------------------+---------------------------+---------------------------+
| Normalization             | L_line / Q_H              | L_line / L_bol            |
+---------------------------+---------------------------+---------------------------+
| JAX / JIT in tengri       | yes (triweight interp.)   | not yet integrated        |
+---------------------------+---------------------------+---------------------------+

CB_19 remains the preferred tengri backend for age-dependent stellar nebular
emission due to its Q_H normalization (which tracks the actual SFH-weighted
ionizing flux) and its C/O + N/O axes (which CB_19 inherits from the 3MdB_17
grid).

References
----------

- Martinez-Paredes et al. 2023, MNRAS, arXiv:2308.05604
- Osterbrock & Ferland 2006, "Astrophysics of Gaseous Nebulae", Table 4.4
- Byler et al. 2017, ApJ, 840, 44 (Hβ conversion factor)
- Morisset et al. 2015, A&A, 3MdBs database
- Gutkin, Charlot & Bruzual 2016, MNRAS, 462, 1757 (BEAGLE HII grids)
- Chevallard & Charlot 2016, MNRAS, 462, 1415 (BEAGLE)
- Lovell et al. 2025 (doi:10.33232/001c.145766) + Roper et al. 2026 (doi:10.21105/joss.09436)

"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, NamedTuple

import h5py
import jax
import jax.numpy as jnp
import numpy as np

from tengri._data_setup import package_or_env_data_path
from tengri.components.nebular._constants import _LOG_OH_OFFSET, _LSUN_ERG
from tengri.components.nebular._recombination_coeffs import lyc_dust_escape_factor
from tengri.components.nebular._shared import (
    _qh_bilinear,
    compute_qh,
    compute_qh_log10,
    render_nebular_lines,
    sanitize_qh_table,
)
from tengri.config.exceptions import warn_measured
from tengri.utils.grid_interp import (
    PreintegratedGrid,
    PreintegratedLines,
    preintegrate_lines,
    slice_fixed_axes,
)
from tengri.utils.interpolation import compute_grid_weights, edges_for_grid
from tengri.utils.scale import pow10

# ── Physical and grid constants ───────────────────────────────────

# Case B L_Hβ/Q_H (Lsun per photon/s = Lsun·s/photon)
# Source: Osterbrock & Ferland 2006, Table 4.4; T_e=10^4 K, n_e=100 cm⁻³
# = 4.78e-13 erg/photon / 3.828e33 erg/Lsun
_HB_PER_QH_LSUN: float = 4.78e-13 / _LSUN_ERG

# Default data file location. Honors $TENGRI_DATA_DIR (#1431).
_DEFAULT_PATH = package_or_env_data_path("cb19_templates.h5")


# ── Ionizing spectrum warnings ────────────────────────────────────


class CB19IonizingSpectrumWarning(UserWarning):
    """Warning: CB19Backend uses BPASS v2.1 ionizing SED, not the DSPS SSPs."""


class CB19NoContinuumWarning(UserWarning):
    """Warning: CB19Backend provides no nebular continuum (returns zeros)."""


class CB19DegenerateGridWarning(UserWarning):
    """Warning: the loaded CB19 line-ratio grid carries no usable variation.

    Emitted at load time when every line ratio in the grid is identical (the
    flat placeholder ``cb19_templates.h5`` of #924). A degenerate grid gives all
    emission lines the same luminosity — e.g. Halpha/Hbeta = 1.0 instead of the
    Case B 2.87 — producing plausible-looking but silently wrong line physics.
    """


def _warn_if_degenerate_line_ratios(ratios: np.ndarray, filepath: Path) -> None:
    """Warn loudly if a CB19 line-ratio slab has no usable variation (#924).

    A grid is degenerate when every finite ratio is identical: interpolating it
    is a no-op and all lines collapse to the same luminosity. This is the shipped
    flat-placeholder failure mode; a real CLOUDY grid varies with metallicity,
    ionization parameter, and density.

    Parameters
    ----------
    ratios : ndarray
        Loaded line-ratio slab (L_line / L_Hbeta), any shape.
    filepath : Path
        Source file, for the diagnostic message.
    """
    finite = ratios[np.isfinite(ratios)]
    if finite.size == 0 or float(finite.min()) == float(finite.max()):
        value = float(finite.min()) if finite.size else float("nan")
        warnings.warn(
            f"CB19: the line-ratio grid loaded from {filepath} is DEGENERATE — "
            f"every line ratio is identical ({value:g}). This is the flat "
            "placeholder cb19_templates.h5 (#924), not a usable CLOUDY grid: all "
            "10 emission lines would receive the same luminosity (Halpha/Hbeta = "
            "1.0 instead of the Case B 2.87), producing plausible-looking but "
            "silently wrong line physics. Regenerate the real grid with "
            "scripts/download_cb19_templates.py.",
            CB19DegenerateGridWarning,
            stacklevel=3,
        )


def _emit_cb19_warnings(ionizing_source_warning: str, continuum_warning: str) -> None:
    """Emit warnings or raise errors about CB19Backend limitations."""
    if ionizing_source_warning not in ("raise", "warn", "suppress"):
        raise ValueError("ionizing_source_warning must be 'raise', 'warn', or 'suppress'")
    if continuum_warning not in ("raise", "warn", "suppress"):
        raise ValueError("continuum_warning must be 'raise', 'warn', or 'suppress'")

    if ionizing_source_warning != "suppress":
        msg = (
            "CB19Backend: the CLOUDY c17.01 grids were computed with BPASS v2.1 "
            "binary stars as the ionizing source. The 6D parameter space does NOT "
            "include variation in ionizing SED hardness. For AGN-ionized or "
            "shock-excited regions, use MappingsPhotoAGNBackend or ShockEmission. "
            "To suppress when building via SEDModel.build: "
            "warnings.filterwarnings('ignore', message='CB19Backend: the CLOUDY'). "
            "(ionizing_source_warning='suppress' also works, but it is a "
            "CB19Backend(...) constructor argument and the build grammar does "
            "not forward it.)"
        )
        if ionizing_source_warning == "raise":
            raise ValueError(msg)
        warnings.warn(msg, CB19IonizingSpectrumWarning, stacklevel=3)

    if continuum_warning != "suppress":
        msg = (
            "CB19Backend provides no nebular continuum — predict_nebular_continuum() "
            "returns zeros. For rest-frame UV continuum accuracy (e.g. z > 2 galaxies "
            "where nebular continuum contributes 10-40% of UV flux), combine with "
            "CloudyGridBackend or CueBackend for the continuum. "
            "To suppress when building via SEDModel.build: "
            "warnings.filterwarnings('ignore', "
            "message='CB19Backend provides no nebular continuum'). "
            "(continuum_warning='suppress' also works, but it is a "
            "CB19Backend(...) constructor argument and the build grammar does "
            "not forward it.)"
        )
        if continuum_warning == "raise":
            raise ValueError(msg)
        warnings.warn(msg, CB19NoContinuumWarning, stacklevel=3)


# ── Grid data container ───────────────────────────────────────────


class CB19GridData(NamedTuple):
    """Pre-loaded CB_19 grid for a fixed (sed_type, imf, mup, hbfrac) combination.

    All line ratios are stored in log10 space (log10(ratio + floor)) for
    interpolation accuracy. The Hβ axis is already collapsed to a single
    HbFrac slice at load time.

    Axes of ``line_ratios``: (N_OH, N_age, N_U, N_nH, N_CO, N_dNO, N_lines).
    """

    # 6 continuous interpolation axes
    log_OH_grid: jnp.ndarray  # (N_OH,) log10(O/H)_total
    log_age_grid: jnp.ndarray  # (N_age,) log10(age/yr)
    log_U_grid: jnp.ndarray  # (N_U,)  log10(U)
    log_nH_grid: jnp.ndarray  # (N_nH,) log10(n_H / cm⁻³)
    log_CO_grid: jnp.ndarray  # (N_CO,) log10(C/O)
    dNO_grid: jnp.ndarray  # (N_dNO,) ΔN/O

    # Line data (last axis of line_ratios)
    line_wavelengths: jnp.ndarray  # (N_lines,) Å vacuum
    log_line_ratios: jnp.ndarray  # (N_OH, N_age, N_U, N_nH, N_CO, N_dNO, N_lines)
    log_hb_per_qh: float  # log10(_HB_PER_QH_LSUN) for fast scaling


# ── HDF5 loader ───────────────────────────────────────────────────

_LOG_FLOOR = 1e-30  # prevent log(0) for zero-flux lines


def load_cb19_grid(
    filepath: str | Path = _DEFAULT_PATH,
    sed_type: str = "SSP",
    imf: str = "Kroupa01",
    mup: float = 100.0,
    hbfrac: float = 1.0,
) -> CB19GridData:
    """Load a CB_19 grid slice from the HDF5 template file.

    Selects the HbFrac slice nearest to ``hbfrac`` and returns a
    ``CB19GridData`` with 6 continuous interpolation axes.

    Line ratios are converted from linear (L_line/L_Hβ) to log10 space.
    The Hβ→L/Q_H conversion constant ``_HB_PER_QH_LSUN`` is stored as
    ``log_hb_per_qh`` for efficient scaling during interpolation.

    Parameters
    ----------
    filepath : str or Path
        Path to ``cb19_templates.h5`` (built by ``scripts/download_cb19_templates.py``).
    sed_type : {"SSP", "CSF"}
        Ionizing SED type. "SSP" = single stellar population (use for line-weighted
        CSP sums); "CSF" = constant star formation.
    imf : {"Kroupa01", "x030"}
        Initial mass function. "Kroupa01" = standard Kroupa (2001).
    mup : {100.0, 300.0}
        Upper stellar mass limit (M_sun).
    hbfrac : float
        HbFrac value (snapped to nearest grid point). HbFrac=1.0 = radiation-bounded.
        Ionizing photon escape fraction ≈ 1 − HbFrac.

    Returns
    -------
    CB19GridData
        Pre-loaded grid with 6 continuous interpolation axes
        (log_OH, log_age, log_U, log_nH, log_CO, dNO) and line data.

    Raises
    ------
    FileNotFoundError
        If the HDF5 file is not found. Run ``scripts/download_cb19_templates.py``
        to build it.
    KeyError
        If the requested (sed_type, imf, mup) combination is not in the file.

    Notes
    -----
    **JIT-compatible**: no — HDF5 I/O is not JAX-compatible. Call once
    at model initialization and cache the result for repeated use.

    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(
            f"CB_19 template file not found: {filepath}\n"
            "Build it with:  python scripts/download_cb19_templates.py"
        )

    group_key = f"grids/{sed_type}/{imf}/mu{int(mup)}"

    with h5py.File(filepath, "r") as f:
        if group_key not in f:
            available = sorted(k for k in f["grids"]) if "grids" in f else []
            raise KeyError(
                f"CB_19 group '{group_key}' not in HDF5. "
                f"Available top-level SED groups: {available}"
            )

        ax = f["axes"]
        log_OH = jnp.array(ax["log_OH_total"][:], dtype=jnp.float32)
        log_U = jnp.array(ax["log_U"][:], dtype=jnp.float32)
        log_nH = jnp.array(ax["log_nH"][:], dtype=jnp.float32)
        log_CO = jnp.array(ax["log_CO"][:], dtype=jnp.float32)
        dNO = jnp.array(ax["dNO"][:], dtype=jnp.float32)
        hbfrac_grid = np.array(ax["HbFrac"][:])

        age_key = "log_age_yr_ssp" if sed_type == "SSP" else "log_age_yr_csf"
        log_age = jnp.array(ax[age_key][:], dtype=jnp.float32)

        line_wavelengths = jnp.array(f["line_wavelengths_aa"][:], dtype=jnp.float32)

        # Select nearest HbFrac slice
        i_hb = int(np.argmin(np.abs(hbfrac_grid - hbfrac)))
        if abs(hbfrac_grid[i_hb] - hbfrac) > 0.15:
            warn_measured(
                f"Requested hbfrac={hbfrac} snapped to nearest grid value "
                f"{hbfrac_grid[i_hb]:.2f} (gap={abs(hbfrac_grid[i_hb] - hbfrac):.2f}). "
                "Available HbFrac values: " + str(hbfrac_grid.tolist()),
                UserWarning,
                stacklevel=2,
                requested_hbfrac=float(hbfrac),
                snapped_hbfrac=float(hbfrac_grid[i_hb]),
                snap_gap=float(abs(hbfrac_grid[i_hb] - hbfrac)),
            )

        # Load line_ratios: (N_OH, N_age, N_U, N_nH, N_CO, N_dNO, N_HbFrac, N_lines)
        # Collapse HbFrac axis → (N_OH, N_age, N_U, N_nH, N_CO, N_dNO, N_lines)
        grp = f[group_key]
        ratios = np.array(grp["line_ratios"][:, :, :, :, :, :, i_hb, :], dtype=np.float32)

    # Guard against a degenerate/placeholder grid (all ratios identical), which
    # would silently give every line the same luminosity (#924).
    _warn_if_degenerate_line_ratios(ratios, filepath)

    # Convert to log10 space; replace NaN with log10(floor)
    log_ratios = np.log10(np.where(np.isfinite(ratios) & (ratios > 0), ratios, _LOG_FLOOR))

    return CB19GridData(
        log_OH_grid=log_OH,
        log_age_grid=log_age,
        log_U_grid=log_U,
        log_nH_grid=log_nH,
        log_CO_grid=log_CO,
        dNO_grid=dNO,
        line_wavelengths=line_wavelengths,
        log_line_ratios=jnp.array(log_ratios),
        log_hb_per_qh=float(np.log10(_HB_PER_QH_LSUN)),
    )


# ── 6D interpolation via map_coordinates ──────────────────────────


def _frac_idx(val: float, grid: jnp.ndarray) -> jnp.ndarray:
    """Convert a scalar value to a fractional grid index, clipped to grid bounds.

    Returns a float index ``f`` such that ``grid[floor(f)]`` and ``grid[ceil(f)]``
    are the two bracketing grid points and the fractional part is the linear weight.
    """
    val_clipped = jnp.clip(val, grid[0], grid[-1])
    n = grid.shape[0]
    idx = jnp.searchsorted(grid, val_clipped, side="right") - 1
    idx = jnp.clip(idx, 0, n - 2)
    dx = grid[idx + 1] - grid[idx]
    frac = jnp.where(dx > 0, (val_clipped - grid[idx]) / dx, 0.0)
    return (idx + frac).astype(jnp.float32)


def _interp_6d(
    data: jnp.ndarray,
    grids: tuple[jnp.ndarray, ...],
    vals: tuple[float, ...],
) -> jnp.ndarray:
    """6D linear interpolation over (OH, age, U, nH, CO, dNO) returning all lines.

    Parameters
    ----------
    data : array, shape (N_OH, N_age, N_U, N_nH, N_CO, N_dNO, N_lines)
        Log-space grid values.
    grids : 6-tuple of 1-D arrays
        Axis values for each of the 6 continuous dimensions.
    vals : 6-tuple of floats
        Query point in the same order as grids.

    Returns
    -------
    array, shape (N_lines,)
        Interpolated log10(ratio) at the query point.

    """
    coords = [_frac_idx(v, g) for v, g in zip(vals, grids)]

    # Cast to float64 before interpolation so gradients stay float64 throughout.
    # Without this, `log_ratios_i.astype(float64)` in lum_per_qh has a VJP that casts
    # the cotangent back to float32.  At the scale of _LSUN_ERG (~3.8e33) the upstream
    # gradient reaches ~1e44, overflowing float32 max (~3.4e38) → +inf → NaN in backward.
    data_f64 = data.astype(jnp.float64)

    # vmap over the lines axis (last): each call to map_coordinates handles one line
    # over the 6D continuous axes.
    data_lines_first = jnp.moveaxis(data_f64, -1, 0)  # (N_lines, N_OH, ...)

    def _interp_one(d6: jnp.ndarray) -> jnp.ndarray:
        """Interpolate a single line over 6D continuous grid axes."""
        return jax.scipy.ndimage.map_coordinates(d6, coords, order=1, mode="nearest")

    return jax.vmap(_interp_one)(data_lines_first)  # (N_lines,)


# ── Q_H helpers (same interface as CloudyGridBackend) ─────────────

_compute_qh_grid = jax.vmap(
    jax.vmap(compute_qh, in_axes=(None, 0)),
    in_axes=(None, 0),
)  # vmap over (n_met, n_age)

#: Log-domain sibling of ``_compute_qh_grid``. The linear form overflows float32
#: on healthy input (Q_H ~ 1e46 against a 3.4e38 ceiling), so the table is built
#: from this and stored normalized (#1568).
_compute_log_qh_grid = jax.vmap(
    jax.vmap(compute_qh_log10, in_axes=(None, 0)),
    in_axes=(None, 0),
)


# ── Main backend class ────────────────────────────────────────────


class CB19Backend:
    """CB_19 CLOUDY grid-based nebular emission backend.

    Loads a precomputed CB_19 CLOUDY photoionization grid (Martinez-Paredes et al.
    2023, arXiv:2308.05604) and computes nebular emission lines at arbitrary
    (logZ_gas, log_age, logU, log_nH, log_CO, dNO) via 6D linear interpolation.

    **Unit convention**: CB_19 stores line fluxes as ratios relative to Hβ
    (dimensionless). This backend converts to L_sun/Q_H at prediction time::

        L_line / Q_H = ratio × (L_Hβ / Q_H)

    where L_Hβ/Q_H = 4.78×10⁻¹³ erg/photon (Case B, T_e=10⁴ K;
    Osterbrock & Ferland 2006, Table 4.4), equivalent to
    ``_HB_PER_QH_LSUN ≈ 1.249×10⁻⁴⁶ Lsun·s/photon``.

    The total line luminosity for a CSP is then::

        L_line = Σ_i  w_i · Q_H(Z, age_i) · (L_line/Q_H)(Z_gas, age_i, logU, ...)

    where w_i is the CSP mass weight (Msun) for SSP bin i and Q_H(Z, age_i) is
    the ionizing photon rate from the actual DSPS SSP spectrum.

    **Note on nebular continuum**: CB_19 provides only line ratios, not a nebular
    continuum (``has_continuum = False``).  For applications that need nebular
    continuum, wrap this backend with
    :class:`~tengri.components.nebular._shared.NebularContinuumFallback`:

    - For analytic free-free + two-photon continuum (fast, no extra data)::

        from tengri.components.nebular._shared import NebularContinuumFallback

        cb19 = CB19Backend(ssp_data=ssp)
        backend = NebularContinuumFallback(cb19, fallback_mode="warn")

    - For full CLOUDY continuum (highest fidelity)::

        cloudy = CloudyGridBackend(grid_path, ssp_data=ssp)
        backend = NebularContinuumFallback(cb19, fallback=cloudy)

    Parameters
    ----------
    sed_type : {"SSP", "CSF"}
        Ionizing SED type. Use "SSP" for SED fitting with DSPS SSP weights.
    imf : {"Kroupa01", "x030"}
        IMF. "Kroupa01" = Kroupa (2001); "x030" = top-heavy (x=−0.30 high-mass slope).
    mup : {100.0, 300.0}
        Upper stellar mass limit in M_sun.
    hbfrac : float
        HbFrac = L_Hβ(matter-bounded)/L_Hβ(radiation-bounded). 1.0 = fully
        radiation-bounded (default). Snapped to nearest grid point at init.
        Ionizing photon escape fraction ≈ 1 − hbfrac.
    grid_path : str or Path, optional
        Path to ``cb19_templates.h5``. Defaults to ``data/cb19_templates.h5``.
    ssp_data : SSPData, optional
        SSP templates used to precompute Q_H(Z, age) table. If None, Q_H must
        be provided externally via ``_qh_table``.

    """

    name = "cb19_grid"
    has_free_params = True

    #: erg/s per [Lsun] for this backend's line catalog (#1559). IAU 2015 —
    #: the CB_19 tabulation and ``_HB_PER_QH_LSUN`` above are both built on it.
    #: Cue overrides this with its own training convention; see CueBackend.
    lsun_erg: float = _LSUN_ERG

    #: log10 of the scalar ``_qh_table`` is normalized by (#1568). Declared on
    #: the class, not only in ``__init__``, so an instance built without the
    #: normal constructor path — the mocked backends in the test suite do
    #: exactly this — still has the identity value rather than an
    #: ``AttributeError``. 0.0 means "table already in photons/s".
    _log_qh_scale: float = 0.0

    def __init__(
        self,
        sed_type: str = "SSP",
        imf: str = "Kroupa01",
        mup: float = 100.0,
        hbfrac: float = 1.0,
        grid_path: str | Path = _DEFAULT_PATH,
        ssp_data=None,
        ionizing_source_warning: str = "warn",
        continuum_warning: str = "warn",
    ) -> None:
        _emit_cb19_warnings(ionizing_source_warning, continuum_warning)

        self.has_continuum = False
        self.sed_type = sed_type
        self.imf = imf
        self.mup = mup
        self.hbfrac = hbfrac

        self.grid = load_cb19_grid(
            filepath=grid_path,
            sed_type=sed_type,
            imf=imf,
            mup=mup,
            hbfrac=hbfrac,
        )

        # log10(L_Hβ/Q_H in Lsun·s/photon): scalar used in all predictions
        self._log_hb_per_qh = self.grid.log_hb_per_qh

        # Max SSP age with non-negligible ionizing flux: 100 Myr
        self._max_neb_log_age = 8.0  # log10(yr)

        # Q_H precomputation (same pattern as CloudyGridBackend)
        self._qh_table = None
        self._qh_log_met = None
        self._qh_log_age = None
        self._young_idx = None
        #: log10 of the scalar the Q_H table is normalized by (#1568). 0.0 with
        #: no table, which makes ``_get_qh_at``'s ``missing=1.0`` the identity
        #: it documents itself to be.
        self._log_qh_scale = 0.0

        if ssp_data is not None:
            self._precompute_qh(ssp_data)

        # Photometry preintegration storage (mirrors CloudyGridBackend duck-type)
        self._preint_continuum: PreintegratedGrid | None = None
        self._preint_lines: PreintegratedLines | None = None
        self._line_lum_collapsed: jnp.ndarray | None = None
        self._has_preint_photometry: bool = False

    def preintegrate_for_photometry(
        self,
        filter_waves: list,
        filter_trans: list,
        redshift: float,
        dl_cm: float,
        fixed: dict[int, float] | None = None,
        *,
        neb_log_nH: float = 2.0,
        neb_co: float = -0.36,
        neb_dno: float = 0.0,
    ) -> None:
        """Preintegrate CB19 lines through filters; expose CLOUDY-shaped surface.

        CB19 has six continuous interpolation axes (log_OH, log_age, log_U,
        log_nH, log_CO, dNO). The hybrid kernel's nebular preint branch
        (``_kernels/hybrid.py``) only knows how to interpolate the
        CLOUDY-shaped 3-axis surface ``(log_met_abs, log_age_yr, log_U)``.
        We bridge by:

        1. Adding ``log_hb_per_qh`` to ``log_line_ratios`` so the grid is in
           log10(L_line/Q_H) [Lsun·s/photon] — the same units the kernel
           feeds through ``10**log_lum × Q_H``.
        2. Triweight-collapsing axes 3 (log_nH), 4 (log_CO), 5 (dNO) at the
           caller-supplied default values (defaults: HII region, near-solar
           C/O, ΔN/O = 0).
        3. Relabeling axis 0 from log10(O/H) on the CLOUDY c17.01 scale to
           absolute log10(Z) by subtracting ``_LOG_OH_OFFSET`` so the
           kernel's ``_gas_z`` (absolute log10(Z)) lands on the correct
           coordinates.
        4. Filling ``_preint_continuum.phot`` with zeros — CB19 has no
           nebular continuum component, so the kernel's continuum age-sum
           contributes nothing.

        After the call the backend exposes the same surface as
        :class:`~tengri.components.nebular.cloudy_grid.CloudyGridBackend`:
        ``_has_preint_photometry``, ``_preint_continuum``, ``_preint_lines``,
        ``_line_lum_collapsed``, ``_qh_table``, ``_qh_log_met``,
        ``_qh_log_age``, ``_young_idx``, ``grid.line_wavelengths``.

        Parameters
        ----------
        filter_waves : list[ndarray]
            Per-filter observed-frame wavelength grids [Angstrom].
        filter_trans : list[ndarray]
            Per-filter transmission curves (0-1).
        redshift : float
            Source redshift [dimensionless].
        dl_cm : float
            Luminosity distance [cm]. Currently unused (CB19 has no
            continuum and lines are projected via filter weights), kept for
            signature compatibility with CloudyGridBackend.
        fixed : dict[int, float], optional
            CLOUDY-shape axis index → value mapping. ``0`` = absolute
            log10(Z), ``1`` = log10(age/yr), ``2`` = log10(U).
        neb_log_nH : float, keyword-only
            Default density to collapse the log_nH axis on. CB19 grid range
            [1, 4]. Default 2.0 (HII region) [log10(cm^-3)].
        neb_co : float, keyword-only
            Default log10(C/O) for the log_CO axis collapse. Default -0.36
            (≈ solar) [log10].
        neb_dno : float, keyword-only
            Default ΔN/O offset for the dNO axis collapse. Default 0.0.

        Notes
        -----
        **JIT-compatible**: no — build-time NumPy / one-time triweight
        collapses. The resulting attributes are JAX arrays usable inside
        the JIT'd kernel body.
        """
        del dl_cm  # CB19: no continuum, lines via point-sampling — no F_nu scaling needed
        grid = self.grid

        # 1. Lift line ratios into log10(L_line/Q_H) [Lsun·s/photon].
        log_lum_per_qh = jnp.asarray(grid.log_line_ratios) + grid.log_hb_per_qh

        # 2. Collapse axes 3 (log_nH), 4 (log_CO), 5 (dNO) to defaults.
        # Iterate from the highest axis index down so earlier indices stay valid.
        extra_axes = (
            jnp.asarray(grid.log_nH_grid),
            jnp.asarray(grid.log_CO_grid),
            jnp.asarray(grid.dNO_grid),
        )
        extra_vals = (neb_log_nH, neb_co, neb_dno)
        for axis_idx in (5, 4, 3):
            ax = extra_axes[axis_idx - 3]
            val = extra_vals[axis_idx - 3]
            scatter = 0.5 * float(ax[1] - ax[0])
            w = compute_grid_weights(val, ax, scatter=scatter, edges=edges_for_grid(ax))
            log_lum_per_qh = jnp.tensordot(w, log_lum_per_qh, axes=([0], [axis_idx]))
        # log_lum_per_qh now: (N_OH, N_age, N_U, N_lines)

        # 3. Build CLOUDY-shape surface axes (axis 0 relabeled to absolute log10(Z)).
        log_met_abs_axis = jnp.asarray(grid.log_OH_grid) - _LOG_OH_OFFSET
        log_age_axis = jnp.asarray(grid.log_age_grid)
        log_U_axis = jnp.asarray(grid.log_U_grid)
        surface_axes = (log_met_abs_axis, log_age_axis, log_U_axis)
        surface_edges = tuple(edges_for_grid(ax) for ax in surface_axes)

        # 4. Continuum: zeros (CB19 has no nebular continuum).
        n_filters = len(filter_waves)
        cont_zero = jnp.zeros(
            (
                log_met_abs_axis.shape[0],
                log_age_axis.shape[0],
                log_U_axis.shape[0],
                n_filters,
            ),
            dtype=jnp.float64,
        )
        self._preint_continuum = PreintegratedGrid(
            phot=cont_zero,
            moment=None,
            axes=surface_axes,
            edges=surface_edges,
            effective_wavelengths=jnp.zeros(n_filters),
            effective_wavelengths_rest=jnp.zeros(n_filters),
            log10_flux_scale=0.0,  # unit scale; the caller applies the cosmology
            n_filters=n_filters,
        )

        # 5. Line filter weights via point-sampling (delegates to shared helper).
        self._preint_lines = preintegrate_lines(
            np.asarray(grid.line_wavelengths),
            filter_waves,
            filter_trans,
            redshift,
            axes=surface_axes,
        )

        # 6. Apply caller-provided fixed dict (axes are CLOUDY-shape indices: 0,1,2).
        line_lum = log_lum_per_qh
        if fixed:
            self._preint_continuum = slice_fixed_axes(self._preint_continuum, fixed)
            line_axes_remaining = list(surface_axes)
            for axis_idx in sorted(fixed.keys(), reverse=True):
                ax = line_axes_remaining[axis_idx]
                val = fixed[axis_idx]
                scatter = 0.5 * float(ax[1] - ax[0])
                w = compute_grid_weights(val, ax, scatter=scatter, edges=edges_for_grid(ax))
                line_lum = jnp.tensordot(w, line_lum, axes=([0], [axis_idx]))
                line_axes_remaining.pop(axis_idx)
            # Update PreintegratedLines.axes/edges to match the collapsed line_lum
            new_axes = tuple(line_axes_remaining)
            self._preint_lines = PreintegratedLines(
                line_filter_weights=self._preint_lines.line_filter_weights,
                axes=new_axes,
                edges=tuple(edges_for_grid(ax) for ax in new_axes),
            )

        self._line_lum_collapsed = line_lum
        self._has_preint_photometry = True

    def _precompute_qh(self, ssp_data) -> None:
        """Precompute Q_H(metallicity, age) from SSP spectra (called once at init).

        Stores a (n_met, n_age) table of ionizing photon rates. At prediction
        time, bilinear interpolation gives Q_H for each SSP age bin.
        """
        ssp_wave = ssp_data.ssp_wave
        ssp_flux = ssp_data.ssp_flux  # (n_met, n_age, n_wave)

        # Build in the log domain and store the table *normalized* by its own
        # peak (#1568). Q_H reaches ~1e46 photons/s/Msun, which float32 cannot
        # hold: the linear build overflowed every entry to ``inf``, and
        # ``sanitize_qh_table`` then rewrote the lot to 0.0 — every CB19 line
        # silently zero in pure float32.
        #
        # Bilinear interpolation is a linear operator, so interpolating
        # ``table / scale`` is exactly interpolating ``table`` and dividing;
        # float64 values are unchanged. Storing ``log10(Q_H)`` and interpolating
        # *that* would instead be a geometric interpolation — a different
        # physical answer — which is why the scale is a scalar, not a log table.
        log_qh_raw = _compute_log_qh_grid(ssp_wave, ssp_flux)
        finite = jnp.isfinite(log_qh_raw)
        self._log_qh_scale = float(jnp.max(jnp.where(finite, log_qh_raw, -jnp.inf)))
        if not np.isfinite(self._log_qh_scale):
            # No usable ionizing flux anywhere: keep the identity scale so the
            # normalized table stays meaningful and Q_H comes out zero.
            self._log_qh_scale = 0.0
        qh_norm = pow10(log_qh_raw - self._log_qh_scale)  # (0, 1], float32-safe
        # Match the other backends: a non-finite Q_H from the SSP grid is
        # scrubbed at build time rather than propagated into every lookup.
        self._qh_table = sanitize_qh_table(qh_norm, backend_name="CloudyCB19Backend")
        # Store as JAX arrays so they can be indexed with traced integers
        # inside jax.vmap (numpy arrays fail when indexed with traced values).
        self._qh_log_met = jnp.array(ssp_data.ssp_lgmet)  # log10(Z) absolute
        self._qh_log_age = jnp.array(ssp_data.ssp_lg_age_gyr + 9.0)  # log10(age/yr)

        ssp_log_ages = np.array(self._qh_log_age)
        young_mask = ssp_log_ages <= self._max_neb_log_age
        self._young_idx = np.where(young_mask)[0]

    @property
    def _lum_scale(self) -> float:
        r"""The one in-range constant the line chain needs [Lsun per unit ratio].

        ``10**_log_hb_per_qh`` is 1.2e-46 and ``10**_log_qh_scale`` is ~1e46 —
        one below float32's smallest subnormal, the other above its ceiling.
        Their product is order unity, and evaluating it here in Python float64
        means neither ever exists as a float32 array (#1568).
        """
        return 10.0 ** (self._log_hb_per_qh + self._log_qh_scale)

    def _get_qh_at(self, log_z: float, log_age_yr: float) -> float:
        """Bilinear interpolation of the precomputed Q_H table.

        Returns the **peak-normalized** Q_H in (0, 1]; the ``10**_log_qh_scale``
        that reconstructs photons/s is folded into :attr:`_lum_scale` rather
        than materialized, because it does not fit in float32 (#1568).

        ``missing=1.0`` is the identity for a multiplicative Q_H — a CB19
        backend without a table applies no Q_H scaling rather than zeroing
        the lines.
        """
        return _qh_bilinear(
            self._qh_table,
            self._qh_log_met,
            self._qh_log_age,
            log_z,
            log_age_yr,
            missing=1.0,
        )

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
        neb_log_nH: float = 2.0,
        neb_co: float = -0.36,
        neb_dno: float = 0.0,
        template_data: Any | None = None,
        **_kwargs,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        r"""Compute emission line luminosities via 6D interpolation over the CB_19 grid.

        **Hβ conversion and k-factor**: CB_19 stores L_line/L_Hβ (dimensionless).
        This method converts to absolute L_line (Lsun) using::

            L_line = Σ_i  w_i · Q_H(Z, age_i) · ratio(Z_gas, age_i, logU, nH, CO, dNO)
                         · (L_Hβ/Q_H) · k(f_esc, f_dust)

        where L_Hβ/Q_H = 4.78×10⁻¹³ / 3.828×10³³ Lsun s/photon (Case B,
        Osterbrock & Ferland 2006 Table 4.4) and the k-factor is:

        .. math::

            k = \frac{1 - f_\mathrm{esc} - f_\mathrm{dust}}
                     {1 + \dfrac{\alpha_1}{\alpha_B}\,(f_\mathrm{esc} + f_\mathrm{dust})}

        Parameters
        ----------
        ssp_weights : array, shape (n_age,)
            CSP stellar mass weights (Msun per SSP age bin).
        ssp_log_ages_yr : array, shape (n_age,)
            log10(age/yr) of SSP bins [log10(yr)].
        log_z : float
            Stellar metallicity log10(Z) (absolute). Used for Q_H interpolation
            [log10(Z)].
        neb_logU : float
            Log ionization parameter log10(U) [log10(U)]. Grid range: [−4, −1.5].
            Default -3.0.
        neb_logZ_gas : float or None
            Gas metallicity log10(Z) absolute [log10(Z)]. None → tied to stellar
            ``log_z``. Converted internally to log10(O/H) using CLOUDY c17.01
            solar scale (log(O/H)_sun = −3.07, i.e. 12+log(O/H)_sun = 8.93).
        neb_fesc : float
            Ionizing photon escape fraction [dimensionless, in [0, 1]].
            Default 0.0.
        neb_fesc_lya : float
            Ly-alpha-specific escape fraction [dimensionless, in [0, 1]].
            Default 0.0. Applied on top of the k-factor.
        neb_fdust : float
            Lyman-continuum dust-absorption fraction in HII regions
            [dimensionless, in [0, 1]]. Default 0.0. Both ``neb_fesc`` and
            ``neb_fdust`` reduce the ionizing photon budget via the CIGALE
            k-factor.
        neb_log_nH : float
            Log hydrogen density log10(n_H/cm⁻³) [log10(cm^-3)]. Grid range: [1, 4].
            Default 2.0.
        neb_co : float
            Log C/O ratio log10(C/O) [log10]. Grid range: [−1, 0.15]. Default -0.36.
        neb_dno : float
            ΔN/O offset (log10) from default N/O scaling [log10]. Grid range:
            [−0.25, 0.25]. Default 0.0.
        template_data : CB19GridData, optional
            Pre-loaded grid threaded as a JIT argument instead of read from
            ``self.grid`` under the trace, where it bakes 0.665 MB of
            ``log_line_ratios`` into every compile (#1694). ``None`` uses
            ``self.grid``.

        Returns
        -------
        wavelengths : array, shape (n_lines,)
            Rest-frame vacuum wavelengths [Angstrom].
        luminosities : array, shape (n_lines,)
            Emission line luminosities [Lsun].

        Notes
        -----
        **JIT-compatible**: yes — all operations use ``jnp`` primitives.

        **Gradient-safe**: yes — differentiable through neb_logU, neb_fesc,
        neb_fdust, neb_log_nH, neb_co, neb_dno parameters.

        References
        ----------
        .. [1] Ferland, G. J. 1980, PASP, 92, 596.
        .. [2] Inoue, A. K. 2011, MNRAS, 415, 2920.
        .. [3] CIGALE nebular module: pcigale/sed_modules/nebular.py, lines
            156-162.

        """
        if neb_logZ_gas is None:
            neb_logZ_gas = log_z

        # Convert absolute log10(Z) → log10(O/H) on CLOUDY scale
        log_oh = neb_logZ_gas + _LOG_OH_OFFSET

        # Prefer the threaded grid. ``SEDModel._template_data_for_jit`` has
        # published this backend's grid all along and ``NebularSEDComponent``
        # has passed it here all along — but the signature ended in ``**_kwargs``,
        # so ``template_data`` was accepted and discarded, and ``self.grid``
        # (a concrete array) was read under the trace instead. Every layer
        # looked wired; nothing was, and the 0.665 MB ``log_line_ratios`` cube
        # was an XLA ``Constant`` on every compile (#1694). Same idiom as
        # ``CloudyGridBackend`` and ``CueBackend``.
        grid = template_data if template_data is not None else self.grid
        grids_6d = (
            grid.log_OH_grid,
            grid.log_age_grid,
            grid.log_U_grid,
            grid.log_nH_grid,
            grid.log_CO_grid,
            grid.dNO_grid,
        )

        # Only young SSP bins (age < 100 Myr) produce ionizing photons
        young_idx = (
            self._young_idx if self._young_idx is not None else np.arange(len(ssp_log_ages_yr))
        )
        young_ages = ssp_log_ages_yr[young_idx]
        young_weights = ssp_weights[young_idx]

        # Compute k-factor once (shared by all age bins)
        k_factor = lyc_dust_escape_factor(neb_fesc, neb_fdust)

        def _line_contrib_one_age(
            log_age_i: float,
            weight_i: float,
        ) -> jnp.ndarray:
            """Compute weighted line luminosity contribution for one SSP age bin."""
            qh_i = self._get_qh_at(log_z, log_age_i)
            log_ratios_i = _interp_6d(
                grid.log_line_ratios,
                grids_6d,
                (log_oh, log_age_i, neb_logU, neb_log_nH, neb_co, neb_dno),
            )
            # Convert: ratio → L_line/Q_H → L_line.
            #
            # The two constants that used to appear separately are both outside
            # float32 range and cancel each other into one that is not (#1568):
            #
            #   10**_log_hb_per_qh = 1.2487e-46   (below float32's 1.4e-45
            #                                      smallest subnormal -> 0.0)
            #   10**_log_qh_scale  ~ 1.06e+46     (above float32's 3.4e38 -> inf)
            #   product            ~ 1.3          (evaluated once, in float64)
            #
            # ``qh_i`` is the peak-normalized Q_H in (0, 1], so every factor
            # below is order unity and nothing needs a dtype cast. The previous
            # spelling reached for ``.astype(jnp.float64)``, which is a no-op
            # under ``jax.enable_x64(False)`` — inert in the mode it guarded.
            lum_per_qh_ratio = 10.0**log_ratios_i
            return weight_i * qh_i * lum_per_qh_ratio * self._lum_scale * k_factor

        # vmap over young age bins, then sum
        all_contribs = jax.vmap(_line_contrib_one_age)(
            young_ages, young_weights
        )  # (n_young, n_lines)
        total_line_lum = jnp.sum(all_contribs, axis=0)  # (n_lines,)

        # Apply differential Ly-alpha escape (resonant scattering).
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
        **_kwargs,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Return zero nebular continuum (not in CB_19 grid).

        CB_19 stores only line flux ratios. If nebular continuum is needed,
        combine this backend with ``CloudyGridBackend`` (use CB_19 for lines,
        FSPS/Byler grid for continuum).

        Parameters
        ----------
        ssp_weights : array, shape (n_age,)
            SSP mass weights (unused).
        ssp_log_ages_yr : array, shape (n_age,)
            SSP log-space ages in years (unused) [log10(yr)].
        log_z : float
            Stellar metallicity log10(Z) (unused) [log10(Z)].
        **_kwargs
            Additional keyword arguments (all unused).

        Returns
        -------
        wavelength : array, shape (1,)
            Dummy wavelength [Angstrom].
        luminosity : array, shape (1,)
            Zero array [erg/s/Hz] — no continuum from CB_19.

        References
        ----------
        .. [1] M. Martinez-Paredes et al., "The 3MdB stellar and AGN library of
           CLOUDY models," MNRAS, 527, 11698 (2024). arXiv:2308.05604.
           https://doi.org/10.1093/mnras/stad3891

        Notes
        -----
        **JIT-compatible**: yes — returns constant arrays.

        """
        return jnp.array([5000.0]), jnp.array([0.0])

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
        neb_log_nH: float = 2.0,
        neb_co: float = -0.36,
        neb_dno: float = 0.0,
        line_sigma_aa: float = 0.0,
        line_sigma_kms: float = 0.0,
        template_data: Any | None = None,
        **_kwargs,
    ) -> jnp.ndarray:
        r"""Compute CB_19 nebular emission lines on the SSP wavelength grid.

        Emission lines from CB_19 are placed onto the SSP wavelength grid as
        Gaussian profiles (if ``line_sigma_aa > 0``) or delta functions added
        to the nearest pixel. There is no nebular continuum component.

        **Hβ conversion and k-factor**: CB_19 stores L_line/L_Hβ (ratios). This
        method converts to absolute L_line via the Case B factor L_Hβ/Q_H =
        4.78×10⁻¹³ erg/photon (Osterbrock & Ferland 2006, Table 4.4), scales by
        the CIGALE k-factor:

        .. math::

            k = \frac{1 - f_\mathrm{esc} - f_\mathrm{dust}}
                     {1 + \dfrac{\alpha_1}{\alpha_B}\,(f_\mathrm{esc} + f_\mathrm{dust})}

        and sums over ionizing SSP age bins.

        Parameters
        ----------
        ssp_weights : array, shape (n_age,)
            CSP stellar mass weights (Msun per SSP age bin).
        ssp_wave : array, shape (n_wave,)
            SSP wavelength grid [Angstrom].
        ssp_log_ages_yr : array, shape (n_age,)
            log10(age/yr) of SSP bins [log10(yr)].
        log_z : float
            Stellar metallicity log10(Z) absolute [log10(Z)].
        neb_logU : float
            Log ionization parameter log10(U) [log10(U)]. Grid range: [−4, −1.5].
            Default -3.0.
        neb_logZ_gas : float or None
            Gas metallicity log10(Z) absolute [log10(Z)]. None → tied to stellar
            ``log_z``.
        neb_fesc : float
            Ionizing photon escape fraction [dimensionless, in [0, 1]].
            Default 0.0.
        neb_fesc_lya : float
            Ly-alpha-specific escape fraction [dimensionless, in [0, 1]].
            Default 0.0.
        neb_fdust : float
            Lyman-continuum dust-absorption fraction in HII regions
            [dimensionless, in [0, 1]]. Default 0.0. Both ``neb_fesc`` and
            ``neb_fdust`` reduce the ionizing photon budget via the CIGALE
            k-factor.
        neb_log_nH : float
            log10(n_H / cm⁻³) [log10(cm^-3)]. Grid range [1, 4]. Default 2.0.
        neb_co : float
            log10(C/O) [log10]. Grid range [−1, 0.15]. Default −0.36 (near-solar).
        neb_dno : float
            ΔN/O offset from default N/O scaling [log10]. Grid range [−0.25, 0.25].
            Default 0.0.
        line_sigma_aa : float
            Gaussian line width (σ) for line profiles [Å]. 0 = nearest-pixel
            delta function. Default 0.0.
        template_data : CB19GridData, optional
            Pre-loaded grid, forwarded to
            :meth:`predict_nebular_line_luminosities` so it threads as a JIT
            argument rather than baking 0.665 MB into every compile (#1694).

        Returns
        -------
        array, shape (n_wave,)
            Nebular emission SED [erg/s/Hz] on the SSP wavelength grid.

        References
        ----------
        .. [1] M. Martinez-Paredes et al., "The 3MdB stellar and AGN library of
           CLOUDY models," MNRAS, 527, 11698 (2024). arXiv:2308.05604.
           https://doi.org/10.1093/mnras/stad3891
        .. [2] D. E. Osterbrock and G. J. Ferland, "Astrophysics of Gaseous Nebulae
           and Active Galactic Nuclei," 2nd edn. (University Science Books, 2006).
           Table 4.4: Case B recombination coefficients.
        .. [3] Ferland, G. J. 1980, PASP, 92, 596.
        .. [4] Inoue, A. K. 2011, MNRAS, 415, 2920.
        .. [5] CIGALE nebular module: pcigale/sed_modules/nebular.py, lines
            156-162.

        Notes
        -----
        **JIT-compatible**: yes — all operations use ``jnp`` primitives.

        **Gradient-safe**: yes — differentiable through neb_logU, neb_fesc,
        neb_fdust, neb_log_nH, neb_co, neb_dno parameters.

        """
        line_wave, line_lum = self.predict_nebular_line_luminosities(
            ssp_weights,
            ssp_log_ages_yr,
            log_z,
            neb_logU=neb_logU,
            neb_logZ_gas=neb_logZ_gas,
            neb_fesc=neb_fesc,
            neb_fesc_lya=neb_fesc_lya,
            neb_fdust=neb_fdust,
            neb_log_nH=neb_log_nH,
            neb_co=neb_co,
            neb_dno=neb_dno,
            # Forward it. This method is the OTHER door into the line
            # luminosities, and leaving it unforwarded left the grid baked even
            # after the callee learned to thread: of the four calls a single
            # trace makes, two arrived here with ``template_data=None``. XLA
            # de-duplicates identical constants, so a half-threaded component
            # measures exactly the same as an un-threaded one (#1694).
            template_data=template_data,
        )

        neb_sed = render_nebular_lines(
            jnp.asarray(line_wave), jnp.asarray(line_lum), ssp_wave, line_sigma_aa, line_sigma_kms
        )

        return neb_sed * _LSUN_ERG  # convert Lsun/Hz → erg/s/Hz, matching CloudyGridBackend

    def __repr__(self) -> str:
        return (
            f"CB19Backend(sed_type={self.sed_type!r}, imf={self.imf!r}, "
            f"mup={self.mup}, hbfrac={self.hbfrac})"
        )
