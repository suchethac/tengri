# SPDX-License-Identifier: BSD-3-Clause
"""MAPPINGS III + V shock emission line model.

Loads Allen+2008 (ApJS 178 20) MAPPINGS III and Alarie & Morisset 2019
(3MdBs, RMxAA 55 279) MAPPINGS V grids from ``data/mappings_templates.h5``
and interpolates line ratios across the full 4-D grid:

    (v_shock, B/√n or B, log_density, abundance)

**Discrete-family structure**: The MAPPINGS V grid is a set of discrete model
families indexed by (abundance, log_density, B-field). Each family is
continuous only in velocity (100–1000 km/s). The populated families are:

- **Allen2008_Solar**: 75 families spanning all six densities (log_density = −2 to 3),
  with 8–18 B nodes per density.
- **Other four abundances** (Allen2008_SMC, Allen2008_LMC, Allen2008_Dopita2005,
  Allen2008_TwiceSolar): 8 families each, all at log_density = 0.

Interpolation on off-node (log_density, B) points snaps to the nearest populated
family and emits a warning. Exact matches (within 1e-6 relative tolerance) return
silently. Velocity is interpolated continuously via triweight kernel.

If the HDF5 file is missing, falls back to the Allen+2008 Table 5
hardcoded arrays (solar, n=1 cm⁻³, 8 velocity points, 10 lines) with a
``DeprecationWarning``.  Build the grid with::

    python scripts/download_mappings_templates.py

Interpolation strategy
----------------------

- **velocity** : C²-continuous triweight kernel (Hearin et al. 2023 / DSPS),
  interpolated over the family's populated velocity range. A velocity outside
  the range raises ``ValueError`` (message names the family and range).
- **log_density, B-field** : Exact node match only (relative tolerance 1e-6).
  Any off-node pair raises ``ValueError`` listing the populated nodes. Under
  ``jax.jit``, the check is skipped as today.
- **abundance, component** : Python string → integer index (static).

References
----------

- Allen et al. 2008, ApJS, 178, 20         (MAPPINGS III)
- Sutherland & Dopita 2017, ApJS, 229, 34  (MAPPINGS V)
- Alarie & Morisset 2019, RMxAA, 55, 279  (3MdBs / Zenodo 14140949)

"""

from __future__ import annotations

import functools
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from tengri.components.nebular._shared import render_nebular_lines as _place_line_profiles

# Physical constants
from tengri.utils.interpolation import edges_for_grid as _edges_for_grid

# ── Legacy hardcoded fallback: Allen+2008 Table 5 (solar, n=1 cm⁻³)

# 8-point velocity grid [km/s]
_FALLBACK_V = jnp.array([100.0, 150.0, 200.0, 300.0, 400.0, 500.0, 750.0, 1000.0])

# Line ratios relative to Hβ (Table 5)
_FALLBACK_R_OII = jnp.array([3.5, 5.2, 4.8, 3.1, 2.4, 1.9, 1.2, 0.8])
_FALLBACK_R_OIII = jnp.array([0.3, 1.5, 4.2, 5.8, 6.1, 5.5, 3.8, 2.5])
_FALLBACK_R_OI = jnp.array([0.8, 1.2, 0.9, 0.5, 0.3, 0.2, 0.15, 0.1])
_FALLBACK_R_NII = jnp.array([2.5, 3.8, 3.2, 2.1, 1.6, 1.3, 0.9, 0.6])
_FALLBACK_R_SII = jnp.array([2.8, 4.5, 3.5, 2.0, 1.4, 1.0, 0.6, 0.4])
_FALLBACK_R_HA = jnp.array([3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7])

# Doublet splitting (atomic physics, independent of shock model)
_OIII_DOUBLET_RATIO = 2.98  # 5007 / 4959
_NII_DOUBLET_RATIO = 2.94  # 6583 / 6548

# PyNeb-format line names and vacuum wavelengths for the fallback lines
_FALLBACK_LINE_NAMES = [
    "OII_3726A",
    "OII_3729A",
    "Hb_4861A",
    "O3_4959A",
    "O3_5007A",
    "OI_6300A",
    "HA_6563A",
    "NII_6548A",
    "NII_6583A",
    "SII_6716A",
    "SII_6731A",
]
_FALLBACK_LINE_WAVES = jnp.array(
    [
        3726.0,
        3729.0,
        4861.0,
        4959.0,
        5007.0,
        6300.0,
        6563.0,
        6548.0,
        6583.0,
        6716.0,
        6731.0,
    ]
)

# ── Abundance short-name aliases (param_spec uses short names; DB uses full names)

_ABUNDANCE_ALIASES: dict[str, str] = {
    "solar": "Allen2008_Solar",
    "2xsolar": "Allen2008_TwiceSolar",
    "twice_solar": "Allen2008_TwiceSolar",
    "dopita2005": "Allen2008_Dopita2005",
    "lmc": "Allen2008_LMC",
    "smc": "Allen2008_SMC",
}


def _resolve_abundance(name: str, available: list[str]) -> int:
    """Return axis index for *name*, resolving short aliases.

    Raises
    ------
    ValueError
        If *name* (or its alias) is not found in *available*.

    """
    resolved = _ABUNDANCE_ALIASES.get(name, name)
    if resolved in available:
        return available.index(resolved)
    if name in available:
        return available.index(name)
    short_keys = sorted(_ABUNDANCE_ALIASES)
    raise ValueError(
        f"shock_abundance={name!r} is not available. "
        f"Valid short names: {short_keys}. "
        f"Full DB names: {available}."
    )


def _resolve_discrete_family_concrete(
    shock_log_density: float,
    shock_b_over_sqrt_n: float,
    shock_abundance: str,
    populated_families: dict,
    abundance_names: list[str],
    log_density_grid: np.ndarray,
    b_grid: np.ndarray,
) -> tuple[float, float]:
    """Resolve (log_density, B) to a populated family node (concrete values only).

    This function validates discrete parameters and is called ONLY when inputs
    are concrete Python floats (not JAX tracers). Under JAX tracing, validation
    is skipped and a pure-JAX fallback is used.

    The MAPPINGS shock grid is discrete in density and B-field: each family
    is discrete at grid nodes. Requests near an exact node (within relative
    tolerance 1e-6) return that node silently. Requests far from any node snap
    to the nearest populated family and emit a warning. Out-of-bounds requests
    (outside the overall grid extent) raise ValueError.

    Parameters
    ----------
    shock_log_density : float
        Pre-shock log10 density in cm⁻³.
    shock_b_over_sqrt_n : float
        B-field in μG.
    shock_abundance : str
        Abundance name (short alias or full).
    populated_families : dict[str, set[tuple[float, float]]]
        Lookup of (log_density, B) pairs populated for each abundance.
    abundance_names : list[str]
        List of full abundance names.
    log_density_grid : ndarray
        Overall log_density axis extent.
    b_grid : ndarray
        Overall B-field axis extent.

    Returns
    -------
    tuple[float, float]
        (log_density, B) of the matched or snapped family node.

    Raises
    ------
    ValueError
        If the abundance is not available, has no populated families,
        or if the requested values are outside the grid extent.
    """
    from tengri.config.exceptions import warn_measured

    # Resolve abundance name
    try:
        i_abund = _resolve_abundance(shock_abundance, abundance_names)
        abund_key = abundance_names[i_abund]
    except ValueError as e:
        raise e

    families = populated_families.get(abund_key, set())
    if not families:
        raise ValueError(
            f"shock_abundance={shock_abundance!r} has no populated families in the grid."
        )

    # Check that the requested values are within the grid extent
    ld_grid_np = np.asarray(log_density_grid)
    b_grid_np = np.asarray(b_grid)

    if not (ld_grid_np[0] <= shock_log_density <= ld_grid_np[-1]):
        raise ValueError(
            f"shock_log_density={shock_log_density:.2f} is outside the grid "
            f"[{ld_grid_np[0]:.2f}, {ld_grid_np[-1]:.2f}] log10(cm⁻³). "
            "Use a value within this range."
        )

    if not (b_grid_np[0] <= shock_b_over_sqrt_n <= b_grid_np[-1]):
        raise ValueError(
            f"shock_b_over_sqrt_n={shock_b_over_sqrt_n:.4g} μG is outside the "
            f"grid [{b_grid_np[0]:.4g}, {b_grid_np[-1]:.4g}] μG. "
            "Use a value within this range."
        )

    # Try to find an exact match within relative tolerance 1e-6
    tol_ld = max(abs(shock_log_density) * 1e-6, 1e-6)
    tol_b = max(abs(shock_b_over_sqrt_n) * 1e-6, 1e-6)

    for ld, b in families:
        if abs(shock_log_density - ld) < tol_ld and abs(shock_b_over_sqrt_n - b) < tol_b:
            return ld, b

    # No exact match: find the nearest family and snap to it
    families_array = np.array(sorted(families))
    requested = np.array([shock_log_density, shock_b_over_sqrt_n])

    # Compute distances (normalized to handle different scales)
    diffs = families_array - requested
    # Use log-scale differences for log_density (which spans ≈5 orders of magnitude)
    # and linear for B-field (which spans ≈5 orders of magnitude at log scale)
    # Normalize both to be dimensionless
    ld_diffs = diffs[:, 0] / (max(abs(shock_log_density), 1.0))
    b_diffs = diffs[:, 1] / (max(abs(shock_b_over_sqrt_n), 1.0))
    distances = np.sqrt(ld_diffs**2 + b_diffs**2)

    nearest_idx = int(np.argmin(distances))
    ld_nearest, b_nearest = families_array[nearest_idx]

    # Group by log_density for readable warning message
    by_density = {}
    for ld, b in sorted(families):
        if ld not in by_density:
            by_density[ld] = []
        by_density[ld].append(b)

    msg = (
        f"The MAPPINGS shock grid is discrete in (log_density, B). "
        f"Requested (log_density={shock_log_density:.3g}, B={shock_b_over_sqrt_n:.4g}) "
        f"does not match a family node for {shock_abundance!r}. "
        f"Snapping to nearest: (log_density={ld_nearest:.3g}, B={b_nearest:.4g}). "
        f"Populated B values for log_density={ld_nearest:.3g}: "
    )
    if ld_nearest in by_density:
        b_str = ", ".join(f"{b:.4g}" for b in sorted(by_density[ld_nearest]))
        msg += f"[{b_str}]"

    warn_measured(msg, UserWarning, stacklevel=3)

    return ld_nearest, b_nearest


def _find_nearest_family_jax(
    shock_log_density: float,
    shock_b_over_sqrt_n: float,
    shock_abundance: str,
    g: dict,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Find nearest family using pure JAX operations (JIT-safe).

    When called under jax.jit with traced parameters, this function uses
    pure JAX primitives (jnp.argmin, jnp.take, etc.) to find the nearest family
    without any Python boolean conversions. No validation or warnings are
    emitted during JIT tracing — those happen at call time in shock_line_ratios
    when inputs are concrete.

    Parameters
    ----------
    shock_log_density : float
        Log10 pre-shock density in cm⁻³ (may be a tracer).
    shock_b_over_sqrt_n : float
        B-field in μG (may be a tracer).
    shock_abundance : str
        Abundance name (must be concrete/static).
    g : dict
        Grid dict from _load_mappings_grids, containing precomputed family lookup.

    Returns
    -------
    tuple[jnp.ndarray, jnp.ndarray]
        (log_density, B) of the nearest family as JAX arrays.
    """
    # Get the precomputed family lookup for this abundance
    families_lookup = g.get("families_lookup", {})
    if shock_abundance not in families_lookup:
        # Fallback: use first density and first B if lookup is missing
        return (
            jnp.array(g["log_density_cm3"][0]),
            jnp.array(g["b_axis"][0]),
        )

    family_data = families_lookup[shock_abundance]
    # family_data is {"ld": array, "b": array, "indices": array}
    ld_array = family_data["ld"]  # shape (N_families,)
    b_array = family_data["b"]    # shape (N_families,)

    # Compute distance in (log_density, B) space
    # Normalize by the range to handle different scales
    ld_min = jnp.min(ld_array)
    ld_max = jnp.max(ld_array)
    ld_range = jnp.maximum(ld_max - ld_min, 1e-6)

    b_min = jnp.min(b_array)
    b_max = jnp.max(b_array)
    b_range = jnp.maximum(b_max - b_min, 1e-6)

    ld_norm = (shock_log_density - ld_array) / ld_range
    b_norm = (shock_b_over_sqrt_n - b_array) / b_range
    distances = jnp.sqrt(ld_norm**2 + b_norm**2)

    # Find nearest family
    nearest_idx = jnp.argmin(distances)
    ld_nearest = jnp.take(ld_array, nearest_idx)
    b_nearest = jnp.take(b_array, nearest_idx)

    return ld_nearest, b_nearest


_VALID_COMPONENTS = frozenset({"shock", "precursor", "combined"})


def _validate_shock_params(
    shock_velocity: float,
    shock_log_density: float,
    shock_b_over_sqrt_n: float,
    shock_abundance: str,
    shock_component: str,
    g: dict,
) -> None:
    """Raise ``ValueError`` for any shock parameter that is out of range.

    Discrete parameters (density, B-field, abundance, component) are always
    validated because they must be concrete values (they are never JAX-traced
    in normal use: they are Fixed in Parameters).

    Velocity is validated only when it is a concrete Python number; when called
    inside ``jax.jit`` with a traced velocity, the check is skipped so that JIT
    compilation succeeds. Velocity is also checked against the per-family
    velocity range for the selected (abundance, log_density, B) family.
    """
    # Component is always a string (Fixed enum, never traced).
    if shock_component not in _VALID_COMPONENTS:
        raise ValueError(
            f"shock_component={shock_component!r} is invalid. "
            f"Choose from {sorted(_VALID_COMPONENTS)}."
        )

    # Each continuous param is checked independently: when traced under jax.jit
    # the float() cast raises and we defer that one param to build-time spec
    # validation. We must NOT bundle them in a single try block: otherwise a
    # traced first param would also skip the bounds check on concrete later
    # params (e.g. user passes traced velocity but a concrete out-of-range
    # density).
    log_den_grid = np.asarray(g["log_density_cm3"])
    try:
        ld = float(shock_log_density)
    except (TypeError, AttributeError):
        pass
    else:
        if not (log_den_grid[0] <= ld <= log_den_grid[-1]):
            raise ValueError(
                f"shock_log_density={ld:.2f} is outside the grid "
                f"[{log_den_grid[0]:.2f}, {log_den_grid[-1]:.2f}] log10(cm⁻³). "
                "Use a value within this range."
            )

    b_arr = np.asarray(g["b_axis"])
    try:
        b = float(shock_b_over_sqrt_n)
    except (TypeError, AttributeError):
        pass
    else:
        if not (b_arr[0] <= b <= b_arr[-1]):
            raise ValueError(
                f"shock_b_over_sqrt_n={b:.4g} μG is outside the "
                f"grid [{b_arr[0]:.4g}, {b_arr[-1]:.4g}] μG. "
                "Use a value within this range."
            )

    v_arr = np.asarray(g["velocities_kms"])
    try:
        v = float(shock_velocity)
    except (TypeError, AttributeError):
        pass
    else:
        # Check against overall grid range
        if not (v_arr[0] <= v <= v_arr[-1]):
            raise ValueError(
                f"shock_velocity={v:.1f} km/s is outside the grid "
                f"[{v_arr[0]:.1f}, {v_arr[-1]:.1f}] km/s."
            )

        # Check against per-family velocity range if available
        # Try to get the resolved abundance name
        abundance_names = g.get("abundance_names", [])
        abund_str = shock_abundance if isinstance(shock_abundance, str) else str(shock_abundance)
        # Resolve the abundance name to the canonical form used in populated_families
        try:
            abund_index = _resolve_abundance(abund_str, abundance_names)
            canonical_abund = abundance_names[abund_index]
            if isinstance(canonical_abund, bytes):
                canonical_abund = canonical_abund.decode()
        except (ValueError, IndexError):
            # If we cannot resolve the abundance, skip the per-family check
            # (it will be caught later when trying to access the family data)
            return

        # Look up the per-family velocity range
        family_velocity_ranges = g.get("family_velocity_ranges", {})
        family_key = (canonical_abund, float(shock_log_density), float(shock_b_over_sqrt_n))
        if family_key in family_velocity_ranges:
            v_min, v_max = family_velocity_ranges[family_key]
            if not (v_min <= v <= v_max):
                raise ValueError(
                    f"shock_velocity={v:.1f} km/s is outside the velocity range "
                    f"[{v_min:.1f}, {v_max:.1f}] km/s for the selected family "
                    f"({canonical_abund}, log_density={shock_log_density:.1f}, "
                    f"B={shock_b_over_sqrt_n:.4g}). "
                    f"This family has data only over {v_min:.1f}–{v_max:.1f} km/s."
                )


# ── HDF5 grid cache ───────────────────────────────────────────────


def _build_populated_families(
    ratio_array: np.ndarray,
    abundance_names: list[str],
    log_density_grid: jnp.ndarray,
    b_grid: jnp.ndarray,
    velocities: jnp.ndarray,
) -> dict[str, set[tuple[float, float]]]:
    """Build a lookup of (log_density, B) pairs that have data for each abundance.

    A (log_density, B) pair is populated iff all velocity and line entries are
    non-NaN (i.e., the raw data before NaN-to-0 conversion is complete).

    Parameters
    ----------
    ratio_array : ndarray, shape (N_abund, N_n, N_v, N_b, N_lines)
        The shock/precursor/combined ratio grid (raw, pre-NaN-conversion).
    abundance_names : list[str]
        List of abundance names.
    log_density_grid : ndarray, shape (N_n,)
        Log10 density grid.
    b_grid : ndarray, shape (N_b,)
        B-field grid.
    velocities : ndarray, shape (N_v,)
        Velocity grid in km/s.

    Returns
    -------
    dict[str, set[tuple[float, float]]]
        Mapping of abundance name to set of (log_density, B) pairs that are populated.
    """
    families = {}
    log_density_grid_np = np.asarray(log_density_grid)
    b_grid_np = np.asarray(b_grid)
    ratio_array_np = np.asarray(ratio_array)

    for i_a, abund_name in enumerate(abundance_names):
        abund_key = abund_name if isinstance(abund_name, str) else abund_name.decode()
        populated = set()
        for i_d, ld in enumerate(log_density_grid_np):
            for i_b, b in enumerate(b_grid_np):
                # A family is populated if all velocity and line entries are non-NaN
                cell = ratio_array_np[i_a, i_d, :, i_b, :]
                if np.all(np.isfinite(cell)):
                    populated.add((float(ld), float(b)))
        families[abund_key] = populated

    return families


def _build_family_velocity_ranges(
    ratio_array: np.ndarray,
    abundance_names: list[str],
    log_density_grid: jnp.ndarray,
    b_grid: jnp.ndarray,
    velocities: jnp.ndarray,
) -> dict[tuple[str, float, float], tuple[float, float]]:
    """Build velocity ranges for each (abundance, log_density, B) family.

    For each populated family, find the minimum and maximum velocity at which
    the family has data (not NaN).

    Parameters
    ----------
    ratio_array : ndarray, shape (N_abund, N_n, N_v, N_b, N_lines)
        The shock/precursor/combined ratio grid (raw, pre-NaN-conversion).
    abundance_names : list[str]
        List of abundance names.
    log_density_grid : ndarray, shape (N_n,)
        Log10 density grid.
    b_grid : ndarray, shape (N_b,)
        B-field grid.
    velocities : ndarray, shape (N_v,)
        Velocity grid in km/s.

    Returns
    -------
    dict[tuple[str, float, float], tuple[float, float]]
        Mapping of (abundance_name, log_density, B) to (v_min, v_max).
    """
    velocity_ranges = {}
    log_density_grid_np = np.asarray(log_density_grid)
    b_grid_np = np.asarray(b_grid)
    ratio_array_np = np.asarray(ratio_array)
    velocities_np = np.asarray(velocities)

    for i_a, abund_name in enumerate(abundance_names):
        abund_key = abund_name if isinstance(abund_name, str) else abund_name.decode()
        for i_d, ld in enumerate(log_density_grid_np):
            for i_b, b in enumerate(b_grid_np):
                # Find which velocity indices have data for this family
                cell = ratio_array_np[i_a, i_d, :, i_b, :]
                is_finite = np.all(np.isfinite(cell), axis=1)
                if np.any(is_finite):
                    # Find min and max velocity for this family
                    v_indices = np.where(is_finite)[0]
                    v_min = float(velocities_np[v_indices[0]])
                    v_max = float(velocities_np[v_indices[-1]])
                    velocity_ranges[(abund_key, float(ld), float(b))] = (v_min, v_max)

    return velocity_ranges


@functools.cache
def _load_mappings_grids() -> dict | None:
    """Load MAPPINGS III + V grids from ``data/mappings_templates.h5``.

    Returns the cached grid dict on subsequent calls.  Returns ``None`` and
    emits ``DeprecationWarning`` if the file is absent.
    """
    from tengri._data_setup import find_data

    # Honors $TENGRI_DATA_DIR as well as the package root (#1431). The sentinel
    # below wants a path to test, so fall back to a non-existent one.
    h5_path = find_data("mappings_templates.h5") or Path("mappings_templates.h5")

    if not h5_path.exists():
        warnings.warn(
            "MAPPINGS grid file not found; using hardcoded Allen+2008 Table 5 "
            "subset (solar, n=1 cm⁻³, 8 velocity points, 10 lines). "
            "Run scripts/download_mappings_templates.py to enable the full grid.",
            DeprecationWarning,
            stacklevel=3,
        )
        return None

    import h5py  # optional dependency: only needed when HDF5 exists

    def _decode(arr: object) -> list[str]:
        """Decode bytes array to string list, handling both bytes and str types."""
        return [n.decode() if isinstance(n, bytes) else str(n) for n in arr]

    grids: dict = {}
    with h5py.File(h5_path, "r") as f:
        if "mappings5" not in f:
            return None
        g = f["mappings5"]

        def _load_ratios(arr: np.ndarray) -> jnp.ndarray:
            """Load ratio array, replacing NaN with 0.0.

            The MAPPINGS V grid is sparse: not all (abundance, density, B-field)
            combinations have MAPPINGS V model outputs.  Positions without models
            are stored as NaN in the rectangular HDF5 array.  Replacing NaN with
            0.0 here ensures that triweight interpolation smoothly returns zero
            emission in unphysical/unmodeled regions rather than propagating NaN.
            """
            raw = np.asarray(arr[:], dtype=np.float32)
            raw = np.where(np.isnan(raw), 0.0, raw)
            return jnp.array(raw, dtype=jnp.float32)

        abundance_names = _decode(g["abundance_names"][:])
        # Keep arrays as concrete numpy initially for _build_populated_families
        log_density_cm3_np = np.asarray(g["log_density_cm3"][:], dtype=np.float32)
        b_axis_np = np.asarray(g["b_field_uG"][:], dtype=np.float32)
        velocities_kms_np = np.asarray(g["velocities_kms"][:], dtype=np.float32)

        # Load raw shock_ratios BEFORE NaN conversion to build populated families
        shock_ratios_raw = np.asarray(g["shock_ratios"][:])
        shock_ratios = _load_ratios(g["shock_ratios"])

        # Convert to JAX arrays after using them for concrete calculations
        grids["mappings5"] = {
            "velocities_kms": jnp.array(velocities_kms_np, dtype=jnp.float32),
            "b_axis": jnp.array(b_axis_np, dtype=jnp.float32),
            "log_density_cm3": jnp.array(log_density_cm3_np, dtype=jnp.float32),
            "abundance_names": abundance_names,
            "line_names": _decode(g["line_names"][:]),
            "line_wavelengths_aa": jnp.array(g["line_wavelengths_aa"][:], dtype=jnp.float32),
            # Shape (N_abund, N_n, N_v, N_B, N_lines): NaN-filled cells → 0.0
            "shock_ratios": shock_ratios,
            "precursor_ratios": _load_ratios(g["precursor_ratios"]),
            "combined_ratios": _load_ratios(g["combined_ratios"]),
            "hbeta_log_lum_erg_s": _load_ratios(g["hbeta_log_lum_erg_s"]),
        }

    # Precompute bin edges for triweight interpolation (static, avoids rebuilding in JIT)
    g5 = grids["mappings5"]
    g5["v_edges"] = _edges_for_grid(g5["velocities_kms"])
    g5["n_edges"] = _edges_for_grid(g5["log_density_cm3"])

    # Build and cache the populated families lookup for discrete (density, B) validation.
    # Use concrete numpy arrays to avoid tracer issues during grid loading.
    g5["populated_families"] = _build_populated_families(
        shock_ratios_raw,  # Use raw data with NaN (already numpy)
        g5["abundance_names"],
        log_density_cm3_np,  # Use concrete numpy, not JAX array
        b_axis_np,  # Use concrete numpy, not JAX array
        velocities_kms_np,  # Use concrete numpy, not JAX array
    )

    # Build and cache the velocity ranges for each family
    g5["family_velocity_ranges"] = _build_family_velocity_ranges(
        shock_ratios_raw,
        g5["abundance_names"],
        log_density_cm3_np,  # Use concrete numpy, not JAX array
        b_axis_np,  # Use concrete numpy, not JAX array
        velocities_kms_np,  # Use concrete numpy, not JAX array
    )

    # Build and cache a JAX-friendly family lookup for pure JAX family resolution.
    # For each abundance, store arrays of (log_density, B) values that are populated,
    # as float32 JAX arrays suitable for use in _find_nearest_family_jax.
    families_lookup = {}
    for abund_name in g5["abundance_names"]:
        populated = g5["populated_families"].get(abund_name, set())
        if populated:
            ld_list = []
            b_list = []
            for ld, b in sorted(populated):
                ld_list.append(float(ld))
                b_list.append(float(b))
            families_lookup[abund_name] = {
                "ld": jnp.array(ld_list, dtype=jnp.float32),
                "b": jnp.array(b_list, dtype=jnp.float32),
            }
    g5["families_lookup"] = families_lookup

    return grids


class ShockTemplateGrid(NamedTuple):
    """The four MAPPINGS V ratio cubes, in a form that threads through ``jax.jit``.

    Read from the module-level cache inside a trace, the *selected* ratio cube
    freezes into the graph as an XLA ``Constant``: 3.73 MB for the
    ``(5, 6, 37, 35, 24)`` float32 array, on every compile, against a 0.05 MB
    bare-stellar floor (#1694). Passed as an argument instead, it is a
    ``Parameter``.

    Only the cubes are carried here. The grid *axes* (velocity, B-field,
    density), their triweight edges, the line wavelengths, and the two name
    lists stay concrete, read from :func:`_load_mappings_grids` as before:
    together they are ~730 bytes, so baking them is free, and keeping them
    concrete is what lets :func:`_validate_shock_params` compare a user's
    velocity against ``float(v_arr[0])`` and lets the abundance string resolve
    to a static integer index. Threading those too would turn every one of
    those Python-level guards into a ``TracerArrayConversionError``.

    Being a :class:`~typing.NamedTuple` of plain arrays, this is already a JAX
    pytree: no custom registration, and no exposure to the "arrays cannot be
    passed as metadata fields" failure that a hand-rolled ``aux_data`` split
    invites (see :class:`~tengri.components.nebular.cue.CueWeights`, #464).

    Attributes
    ----------
    shock_ratios, precursor_ratios, combined_ratios : ndarray
        Line ratios relative to Hbeta, shape
        ``(n_abund, n_n, n_v, n_b, n_lines)`` [dimensionless]. Sparse grid
        cells are NaN in the file and stored as 0.0 here.
    hbeta_log_lum_erg_s : ndarray
        Hbeta luminosity normalization [log10(erg/s)].

    Notes
    -----
    **JIT-compatible**: yes, that is the point of the type. Pass an instance
    as an argument rather than closing over it.
    """

    shock_ratios: jnp.ndarray
    precursor_ratios: jnp.ndarray
    combined_ratios: jnp.ndarray
    hbeta_log_lum_erg_s: jnp.ndarray


def load_shock_template_grid() -> ShockTemplateGrid | None:
    """Load the MAPPINGS V ratio cubes in threadable form.

    Returns
    -------
    ShockTemplateGrid or None
        ``None`` when ``data/mappings_templates.h5`` is absent or carries no
        ``mappings5`` group: the caller then falls back to the hardcoded
        Allen+2008 subset, exactly as before threading existed.

    Notes
    -----
    **JIT-compatible**: no, call at build time. The returned value is what
    threads.
    """
    grids = _load_mappings_grids()
    if grids is None or "mappings5" not in grids:
        return None
    g = grids["mappings5"]
    return ShockTemplateGrid(
        shock_ratios=g["shock_ratios"],
        precursor_ratios=g["precursor_ratios"],
        combined_ratios=g["combined_ratios"],
        hbeta_log_lum_erg_s=g["hbeta_log_lum_erg_s"],
    )


# ── Public API ────────────────────────────────────────────────────


def shock_line_ratios(
    shock_velocity: float,
    shock_log_density: float = 0.0,
    shock_b_over_sqrt_n: float = 1.0,
    shock_abundance: str = "solar",
    shock_component: str = "combined",
    templates: ShockTemplateGrid | None = None,
) -> dict[str, float]:
    """Return emission line luminosity ratios relative to Hβ.

    Loads the MAPPINGS V (3MdBs) grid from ``data/mappings_templates.h5``.
    If the file is missing, falls back to the Allen+2008 Table 5 hardcoded
    subset with a ``DeprecationWarning``.

    Parameters
    ----------
    shock_velocity : float
        Shock velocity in km/s.  Must be within the grid range
        (100–1000 km/s fallback; 200–1000 km/s HDF5).  Raises ``ValueError``
        if out of range.  Continuously interpolated via triweight kernel:
        safe under ``jax.jit``.
    shock_log_density : float
        Log10 pre-shock density in cm⁻³ (e.g. ``0.0`` = 1 cm⁻³).
        Matched to the nearest populated grid node for the chosen abundance.
        **Discrete parameter**: exact matches (within 1e-6 relative tolerance)
        return silently; off-node values snap to the nearest family and emit a
        ``UserWarning``. Not safe under ``jax.jit``.
    shock_b_over_sqrt_n : float
        Absolute B-field strength in μG (3MdBs MAPPINGS V convention).
        Matched to the nearest populated grid node for the chosen abundance
        and log_density.  **Discrete parameter**: exact matches (within 1e-6
        relative tolerance) return silently; off-node values snap to the nearest
        family and emit a ``UserWarning``. Not safe under ``jax.jit``.
    shock_abundance : str
        Abundance pattern.  Accepted short names:
        ``"solar"``, ``"2xsolar"`` / ``"twice_solar"``, ``"dopita2005"``,
        ``"lmc"``, ``"smc"``.  Full 3MdBs DB names (e.g.
        ``"Allen2008_Solar"``) also accepted.  Raises ``ValueError`` for
        unknown names.
    shock_component : str
        Which emission component to return.  One of ``"shock"``,
        ``"precursor"``, ``"combined"`` (default).  Raises ``ValueError``
        for unknown values.
    templates : ShockTemplateGrid, optional
        Pre-loaded ratio cubes, threaded as a JIT argument so they are not
        baked into the compiled graph as constants (#1694). ``None`` (the
        default) reads them from the module-level cache, which is correct but
        costs 3.73 MB per compile. Supplied automatically by
        :class:`~tengri.components.nebular.shock_model.ShockNebular` on the
        model path; callers using this function directly rarely need it.

    Returns
    -------
    dict[str, float]
        PyNeb-format line name → luminosity ratio relative to Hβ
        [dimensionless].

    Raises
    ------
    ValueError
        If any parameter is invalid. Specifically, if log_density or B is
        outside the grid extent, or if the abundance is not recognized.

    References
    ----------
    .. [1] D. A. Allen et al., "The Distance and Metallicity of the Galaxy M33,"
       ApJS, 178, 20 (2008). https://doi.org/10.1086/589652
    .. [2] R. J. R. Sutherland and M. A. Dopita, "Spectral Synthesis Modeling
       of AGN Heating in Starburst and Post-Starburst Galaxies," ApJS, 229, 34
       (2017). https://doi.org/10.3847/1538-4365/aa6541
    .. [3] D. Alarie and C. Morisset, "Synthetic Narrow-Line Emission from a
       Large Grid of CLOUDY Models," Rev. Mex. Astron. Astrofis., 55, 279
       (2019). https://doi.org/10.22201/ia.01851101p.2019.55.02.14

    Notes
    -----
    **JIT-compatible**: partial. Velocity is interpolated continuously
    via ``interp_nd_triweight`` (safe under ``jax.jit``). Abundance and
    component are resolved at call time (static indices). log_density and
    B-field are discrete and snap to the nearest populated family if off-node
    (not safe under ``jax.jit``).

    **Grid structure**: The MAPPINGS V grid is a set of discrete model
    families indexed by (abundance, log_density, B). Each family is
    continuous only in velocity. Solar abundance has 75 families spanning
    all six densities; other abundances have 8 families each at log_density=0.

    **Fallback**: If the MAPPINGS V HDF5 grid is missing, falls back to
    the Allen+2008 Table 5 hardcoded array (solar abundance, n=1 cm⁻³,
    8 velocity points) with a ``DeprecationWarning``. To avoid this,
    download the grid via ``scripts/download_mappings_templates.py``.

    **Interpolation**: Triweight kernel (C²-continuous) interpolates velocity
    along the selected family's velocity curve. Grid edges are precomputed at
    load time to avoid rebuilding inside JIT traces. log_density and B-field
    are discrete: exact matches (within 1e-6) return silently; off-node values
    snap to the nearest family with a warning.

    Examples
    --------
    >>> ratios = shock_line_ratios(300.0)  # solar, combined, 300 km/s
    >>> ratios = shock_line_ratios(500.0, shock_component="precursor")
    >>> ratios = shock_line_ratios(400.0, shock_abundance="lmc", shock_log_density=0.0)

    """
    grids = _load_mappings_grids()

    # ── HDF5 grid path ─────────────────────────────────────────────
    if grids is not None and "mappings5" in grids:
        g = grids["mappings5"]

        # Resolve the discrete (log_density, B) family.
        # When inputs are concrete (direct call), validate and emit warnings.
        # When inputs are traced (under jax.jit), use pure JAX operations.
        try:
            # Try to resolve with concrete values; fails if inputs are traced.
            ld_resolved, b_resolved = _resolve_discrete_family_concrete(
                shock_log_density,
                shock_b_over_sqrt_n,
                shock_abundance,
                g["populated_families"],
                g["abundance_names"],
                g["log_density_cm3"],
                g["b_axis"],
            )
            # Only validate when concrete (no exceptions raised above)
            try:
                _validate_shock_params(
                    shock_velocity,
                    ld_resolved,
                    b_resolved,
                    shock_abundance,
                    shock_component,
                    g,
                )
            except (TypeError, AttributeError):
                # Skip validation if any parameter becomes a tracer during validation
                pass
        except (TypeError, AttributeError):
            # Inputs are JAX tracers (under jax.jit). Use pure JAX resolution.
            ld_resolved, b_resolved = _find_nearest_family_jax(
                shock_log_density,
                shock_b_over_sqrt_n,
                shock_abundance,
                g,
            )
            # No validation under trace

        # --- static string → integer index (outside JIT tracing) ---
        i_abund = _resolve_abundance(shock_abundance, g["abundance_names"])

        # Find the grid indices for the resolved (density, B) family.
        # Use JAX operations for compatibility with traced parameters.
        log_den_grid_jnp = g["log_density_cm3"]  # Already a JAX array
        b_grid_jnp = g["b_axis"]  # Already a JAX array
        i_ld = jnp.argmin(jnp.abs(log_den_grid_jnp - ld_resolved))
        i_b = jnp.argmin(jnp.abs(b_grid_jnp - b_resolved))

        component_map = {
            "shock": "shock_ratios",
            "precursor": "precursor_ratios",
            "combined": "combined_ratios",
        }
        ratio_field = component_map.get(shock_component, "combined_ratios")
        # Prefer the threaded cube: read from ``g`` it is a closure-captured
        # concrete array and XLA inlines all 3.73 MB of it into every compile
        # (#1694). ``templates`` carries the identical values as a traced
        # argument. The velocity axis below stays concrete either way: it is ~148
        # bytes and the bounds check needs real numbers.
        ratio_array = getattr(templates, ratio_field) if templates is not None else g[ratio_field]
        # shape: (N_abund, N_n, N_v, N_B, N_lines)

        # Extract the 1-D velocity curve for this family and component using JAX operations.
        # Using jnp.take to handle both traced and concrete indices.
        # First, extract along the density axis: ratio_array[i_abund, i_ld, :, :, :]
        family_2d = jnp.take(ratio_array[i_abund], i_ld, axis=0)  # shape (N_v, N_B, N_lines)
        # Then, extract along the B axis: family_2d[:, i_b, :]
        family_1d = jnp.take(family_2d, i_b, axis=1)  # shape (N_v, N_lines)

        # Get full velocity grid and the family's populated velocity range
        v_grid = g["velocities_kms"]
        canonical_abund = (
            shock_abundance if isinstance(shock_abundance, str) else shock_abundance.decode()
        )
        # Only look up in family_velocity_ranges if ld_resolved and b_resolved are concrete floats
        # (not JAX arrays). When using the JAX path, use full grid range.
        family_velocity_ranges = g.get("family_velocity_ranges", {})
        v_min_family = None
        v_max_family = None
        try:
            # Try to convert to floats for dict lookup
            ld_float = float(ld_resolved) if isinstance(ld_resolved, (int, float)) else float(np.asarray(ld_resolved))
            b_float = float(b_resolved) if isinstance(b_resolved, (int, float)) else float(np.asarray(b_resolved))
            family_key = (canonical_abund, ld_float, b_float)
            v_min_family, v_max_family = family_velocity_ranges.get(family_key, (None, None))
        except (TypeError, AttributeError, ValueError):
            # ld_resolved or b_resolved are JAX tracers; use None as fallback
            pass

        # If lookup failed or values are None, use full grid range (as JAX arrays)
        if v_min_family is None:
            v_min_family = v_grid[0]
        if v_max_family is None:
            v_max_family = v_grid[-1]

        # Clip velocity to the family's populated range
        # Use jnp.clip to be JAX-safe during jit tracing
        v_q = jnp.clip(shock_velocity, v_min_family, v_max_family)

        # 1-D linear interpolation across velocity only, for each line
        # Use the full velocity grid (boolean masking is not JAX-safe under tracing)
        line_names = g["line_names"]
        n_lines = len(line_names)
        ratios_vec = jnp.zeros(n_lines, dtype=jnp.float32)

        for j in range(n_lines):
            line_curve = family_1d[:, j]  # shape (N_v,) - use full grid
            # Linear interpolate this line's values along the full velocity grid
            # NaN values in the grid are handled by jnp.interp, replaced with 0 during loading
            interp_val = jnp.interp(v_q, v_grid, line_curve)
            ratios_vec = ratios_vec.at[j].set(interp_val)

        return {name: ratios_vec[j] for j, name in enumerate(line_names)}

    # ── Fallback path: hardcoded Allen+2008 Table 5 ───────────────
    v_clip = jnp.clip(shock_velocity, 100.0, 1000.0)
    r_oiii = jnp.interp(v_clip, _FALLBACK_V, _FALLBACK_R_OIII)
    r_nii = jnp.interp(v_clip, _FALLBACK_V, _FALLBACK_R_NII)
    r_sii = jnp.interp(v_clip, _FALLBACK_V, _FALLBACK_R_SII)

    return {
        "OII_3726A": jnp.interp(v_clip, _FALLBACK_V, _FALLBACK_R_OII) / 2.0,
        "OII_3729A": jnp.interp(v_clip, _FALLBACK_V, _FALLBACK_R_OII) / 2.0,
        "Hb_4861A": jnp.array(1.0),
        "O3_4959A": r_oiii / _OIII_DOUBLET_RATIO,
        "O3_5007A": r_oiii,
        "OI_6300A": jnp.interp(v_clip, _FALLBACK_V, _FALLBACK_R_OI),
        "HA_6563A": jnp.interp(v_clip, _FALLBACK_V, _FALLBACK_R_HA),
        "NII_6548A": r_nii / _NII_DOUBLET_RATIO,
        "NII_6583A": r_nii,
        "SII_6716A": r_sii / 2.0,
        "SII_6731A": r_sii / 2.0,
    }


def _shock_line_arrays(
    shock_velocity: float,
    l_shock_halpha: float,
    shock_log_density: float = 0.0,
    shock_b_over_sqrt_n: float = 1.0,
    shock_abundance: str = "solar",
    shock_component: str = "combined",
    templates: ShockTemplateGrid | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute shock line wavelengths and absolute luminosities.

    Returns a ``(wavelengths_aa, luminosities_lsun)`` tuple, both shape
    ``(N_lines,)``, ordered consistently with the active grid or fallback.

    The absolute scaling anchors on ``l_shock_halpha``:
    ``L_line = (ratio / r_ha) * l_shock_halpha``.

    ``templates`` is forwarded to :func:`shock_line_ratios`; see it for why.
    """
    ratios = shock_line_ratios(
        shock_velocity,
        shock_log_density=shock_log_density,
        shock_b_over_sqrt_n=shock_b_over_sqrt_n,
        shock_abundance=shock_abundance,
        shock_component=shock_component,
        templates=templates,
    )

    grids = _load_mappings_grids()
    if grids is not None and "mappings5" in grids:
        line_waves = grids["mappings5"]["line_wavelengths_aa"]
        line_names = grids["mappings5"]["line_names"]
    else:
        line_waves = _FALLBACK_LINE_WAVES
        line_names = _FALLBACK_LINE_NAMES

    # Halpha key in PyNeb format
    ha_key = "HA_6563A"
    r_ha = ratios.get(ha_key, jnp.array(3.0))
    # Guard against zero Hα (e.g. query in unmodeled sparse-grid region): when r_ha=0,
    # all other ratios are also 0, so dividing by 1.0 still gives zero luminosities.
    r_ha_safe = jnp.where(r_ha > 0.0, r_ha, jnp.ones_like(r_ha))

    # Compute luminosities with proper scaling to avoid overflow when l_shock_halpha is large.
    # Strategy: compute (ratio / r_ha) first (which is typically < 10), then multiply by l_shock_halpha.
    # This avoids the case where ratio * l_shock_halpha overflows before the division happens.
    # When l_shock_halpha is very large (e.g., 1e40), the division by r_ha first keeps values smaller.
    lums = jnp.array(
        [(ratios.get(n, jnp.array(0.0)) / r_ha_safe * l_shock_halpha) for n in line_names]
    )
    return line_waves, lums


def compute_shock_sed(
    wavelength: jnp.ndarray,
    shock_velocity: float,
    l_shock_halpha: float,
    shock_log_density: float = 0.0,
    shock_b_over_sqrt_n: float = 1.0,
    shock_abundance: str = "solar",
    shock_component: str = "combined",
    line_sigma_aa: float = 0.0,
    line_sigma_kms: float = 0.0,
    templates: ShockTemplateGrid | None = None,
) -> jnp.ndarray:
    """Compute shock emission line SED.

    Places MAPPINGS V (3MdBs) shock emission lines on an arbitrary wavelength
    grid as Gaussians (``line_sigma_aa > 0``) or delta functions.

    Parameters
    ----------
    wavelength : array, shape (n_wave,)
        Wavelength grid in Å (rest-frame, increasing).
    shock_velocity : float
        Shock velocity in km/s.
    l_shock_halpha : float
        Total shock Hα luminosity in erg/s (normalization anchor).
    shock_log_density : float
        Log10 pre-shock density in cm⁻³.
    shock_b_over_sqrt_n : float
        Absolute B-field in μG (3MdBs MAPPINGS V).  Snapped to nearest grid point.
    shock_abundance : str
        Abundance set (see ``shock_line_ratios``).
    shock_component : str
        ``"shock"``, ``"precursor"``, or ``"combined"``.
    line_sigma_aa : float
        Gaussian line width in Å.  ``0`` → delta function into nearest pixel.
    templates : ShockTemplateGrid, optional
        Pre-loaded ratio cubes threaded as a JIT argument instead of baked into
        the graph as 3.73 MB of constants (#1694). Forwarded to
        :func:`shock_line_ratios`; ``None`` reads the module-level cache.

    Returns
    -------
    ndarray, shape (n_wave,)
        Shock emission SED in erg/s/Hz [erg/s/Hz].

    Notes
    -----
    **JIT-compatible**: yes, all operations use ``jnp`` primitives and
    calls to ``_shock_line_arrays`` and ``_place_line_profiles``.

    **Line placement**: Emission lines are placed as Gaussian profiles
    (if ``line_sigma_aa > 0``) or delta functions (if ``line_sigma_aa = 0``).
    Delta functions scatter energy into the nearest pixel, normalized
    by pixel width to preserve flux.

    """
    line_waves, line_lums = _shock_line_arrays(
        shock_velocity,
        l_shock_halpha,
        shock_log_density=shock_log_density,
        shock_b_over_sqrt_n=shock_b_over_sqrt_n,
        shock_abundance=shock_abundance,
        shock_component=shock_component,
        templates=templates,
    )

    return _place_line_profiles(line_waves, line_lums, wavelength, line_sigma_aa, line_sigma_kms)


# ── Protocol-conformant backend class ─────────────────────────────


@dataclass
class ShockBackend:
    """MAPPINGS V shock emission backend satisfying the NebularBackend Protocol.

    Wraps :func:`compute_shock_sed` as an object-oriented backend so that shock
    emission can be stored and dispatched using the same interface as other
    nebular backends (CueBackend, CloudyGridBackend, etc.).

    Static (non-traced) configuration is stored as dataclass fields.
    JAX-traced continuous parameters (velocity, density, B-field, Hα luminosity)
    are passed as arguments to :meth:`predict_nebular_sed`.

    Parameters
    ----------
    shock_abundance : str
        Abundance set name: ``"solar"``, ``"2xsolar"``, ``"lmc"``, ``"smc"``, etc.
    shock_component : str
        ``"shock"``, ``"precursor"``, or ``"combined"``.
    has_continuum : bool
        Always ``False``: MAPPINGS V provides line emission only.
    has_free_params : bool
        Always ``True``: velocity, density, B-field are differentiable parameters.
    name : str
        Backend identifier string ("shock").

    Notes
    -----
    **JIT-compatible**: Methods return JAX arrays suitable for JIT compilation.
    All computations use pure functions with no side effects.

    **Attributes**: ``has_continuum`` is always False; MAPPINGS V provides
    shock-associated emission lines only (no underlying continuum).
    ``has_free_params`` is always True: all parameters (velocity, density,
    B-field) are differentiable and suitable for optimization.

    """

    shock_abundance: str = "solar"
    shock_component: str = "combined"
    has_continuum: bool = field(default=False, init=False)
    has_free_params: bool = field(default=True, init=False)
    name: str = field(default="shock", init=False)

    def predict_nebular_sed(
        self,
        wavelength: jnp.ndarray,
        shock_velocity: float,
        l_shock_halpha: float,
        shock_log_density: float = 0.0,
        shock_b_over_sqrt_n: float = 1.0,
        line_sigma_aa: float = 0.0,
        line_sigma_kms: float = 0.0,
        **_kwargs,
    ) -> jnp.ndarray:
        """Compute shock emission line SED on a wavelength grid.

        Parameters
        ----------
        wavelength : array, shape (n_wave,)
            Wavelength grid in Å (rest-frame, increasing).
        shock_velocity : float
            Shock velocity in km/s.
        l_shock_halpha : float
            Total shock Hα luminosity in erg/s (normalization anchor).
        shock_log_density : float
            Log10 pre-shock density in cm⁻³.
        shock_b_over_sqrt_n : float
            Absolute B-field in μG.
        line_sigma_aa : float
            Gaussian line width in Å.  ``0`` → delta function into nearest pixel.
        **_kwargs
            Extra keyword arguments silently ignored for protocol compatibility.

        Returns
        -------
        ndarray, shape (n_wave,)
            Shock emission SED [erg/s/Hz].

        Notes
        -----
        **JIT-compatible**: yes, delegates to ``compute_shock_sed``.

        **Abundance and component**: These are fixed at backend initialization
        via ``shock_abundance`` and ``shock_component`` dataclass fields.
        Continuous parameters (velocity, density, B-field, H-alpha luminosity)
        are traced and differentiable under ``jax.jit``.

        """
        return compute_shock_sed(
            wavelength,
            shock_velocity,
            l_shock_halpha,
            shock_log_density=shock_log_density,
            shock_b_over_sqrt_n=shock_b_over_sqrt_n,
            shock_abundance=self.shock_abundance,
            shock_component=self.shock_component,
            line_sigma_aa=line_sigma_aa,
            line_sigma_kms=line_sigma_kms,
        )
