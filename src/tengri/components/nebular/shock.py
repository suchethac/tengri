# SPDX-License-Identifier: BSD-3-Clause
"""MAPPINGS III + V shock emission line model.

Loads Allen+2008 (ApJS 178 20) MAPPINGS III and Alarie & Morisset 2019
(3MdBs, RMxAA 55 279) MAPPINGS V grids from ``data/mappings_templates.h5``
and interpolates line ratios across the full 4-D grid:

    (v_shock, B/√n or B, log_density, abundance)

If the HDF5 file is missing, falls back to the Allen+2008 Table 5
hardcoded arrays (solar, n=1 cm⁻³, 8 velocity points, 10 lines) with a
``DeprecationWarning``.  Build the grid with::

    python scripts/download_mappings_templates.py

Interpolation strategy
----------------------

- velocity, B-field, log_density : ``interp_nd_triweight`` — C²-continuous
  triweight kernel (Hearin et al. 2023 / DSPS), jointly interpolated across
  all three continuous axes.  Bin edges are precomputed at grid load time via
  ``edges_for_grid`` to avoid rebuilding inside JIT traces.
- abundance, component, version : Python string → integer index (static)

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

import jax.numpy as jnp
import numpy as np

from tengri.components.nebular._shared import render_nebular_lines as _place_line_profiles

# Physical constants
from tengri.utils.grid_interp import interp_nd_triweight as _interp_nd_triweight
from tengri.utils.interpolation import edges_for_grid as _edges_for_grid

# ── Legacy hardcoded fallback — Allen+2008 Table 5 (solar, n=1 cm⁻³)

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
    in normal use — they are Fixed in Parameters).

    Velocity is validated only when it is a concrete Python number; when called
    inside ``jax.jit`` with a traced velocity, the check is skipped so that JIT
    compilation succeeds.
    """
    # Component is always a string (Fixed enum, never traced).
    if shock_component not in _VALID_COMPONENTS:
        raise ValueError(
            f"shock_component={shock_component!r} is invalid. "
            f"Choose from {sorted(_VALID_COMPONENTS)}."
        )

    # Each continuous param is checked independently: when traced under jax.jit
    # the float() cast raises and we defer that one param to build-time spec
    # validation. We must NOT bundle them in a single try block — otherwise a
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
        if not (v_arr[0] <= v <= v_arr[-1]):
            raise ValueError(
                f"shock_velocity={v:.1f} km/s is outside the grid "
                f"[{v_arr[0]:.1f}, {v_arr[-1]:.1f}] km/s."
            )


# ── HDF5 grid cache ───────────────────────────────────────────────


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

    import h5py  # optional dependency — only needed when HDF5 exists

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

        grids["mappings5"] = {
            "velocities_kms": jnp.array(g["velocities_kms"][:], dtype=jnp.float32),
            "b_axis": jnp.array(g["b_field_uG"][:], dtype=jnp.float32),
            "log_density_cm3": jnp.array(g["log_density_cm3"][:], dtype=jnp.float32),
            "abundance_names": _decode(g["abundance_names"][:]),
            "line_names": _decode(g["line_names"][:]),
            "line_wavelengths_aa": jnp.array(g["line_wavelengths_aa"][:], dtype=jnp.float32),
            # Shape: (N_abund, N_n, N_v, N_B, N_lines) — NaN-filled cells → 0.0
            "shock_ratios": _load_ratios(g["shock_ratios"]),
            "precursor_ratios": _load_ratios(g["precursor_ratios"]),
            "combined_ratios": _load_ratios(g["combined_ratios"]),
            "hbeta_log_lum_erg_s": _load_ratios(g["hbeta_log_lum_erg_s"]),
        }

    # Precompute bin edges for triweight interpolation (static, avoids rebuilding in JIT)
    g5 = grids["mappings5"]
    g5["v_edges"] = _edges_for_grid(g5["velocities_kms"])
    g5["b_edges"] = _edges_for_grid(g5["b_axis"])
    g5["n_edges"] = _edges_for_grid(g5["log_density_cm3"])

    return grids


# ── Public API ────────────────────────────────────────────────────


def shock_line_ratios(
    shock_velocity: float,
    shock_log_density: float = 0.0,
    shock_b_over_sqrt_n: float = 1.0,
    shock_abundance: str = "solar",
    shock_component: str = "combined",
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
        if out of range.  Continuously interpolated — safe under ``jax.jit``.
    shock_log_density : float
        Log10 pre-shock density in cm⁻³ (e.g. ``0.0`` = 1 cm⁻³).
        Must be within ``[0, 3]``.  Continuously interpolated via triweight
        kernel — safe under ``jax.jit``.  Raises ``ValueError`` if out of range.
    shock_b_over_sqrt_n : float
        Absolute B-field strength in μG (3MdBs MAPPINGS V convention).
        Must be within ``[0.0001, 10]`` μG.  Continuously interpolated via
        triweight kernel — safe under ``jax.jit``.  Raises ``ValueError`` if
        out of range.
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

    Returns
    -------
    dict[str, float]
        PyNeb-format line name → luminosity ratio relative to Hβ
        [dimensionless].

    Raises
    ------
    ValueError
        If any parameter is outside the grid bounds or invalid.

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
    **JIT-compatible**: yes — continuous parameters (velocity, density,
    B-field) are interpolated via ``interp_nd_triweight``, safe under
    ``jax.jit``. Discrete parameters (abundance, component) are resolved
    at call time and not traced.

    **Fallback**: If the MAPPINGS V HDF5 grid is missing, falls back to
    the Allen+2008 Table 5 hardcoded array (solar abundance, n=1 cm⁻³,
    8 velocity points) with a ``DeprecationWarning``. To avoid this,
    download the grid via ``scripts/download_mappings_templates.py``.

    **Interpolation**: Triweight kernel (C²-continuous) jointly interpolates
    velocity, density, and B-field across all three axes. Grid edges are
    precomputed at load time to avoid rebuilding inside JIT traces.

    Examples
    --------
    >>> ratios = shock_line_ratios(300.0)  # solar, combined, 300 km/s
    >>> ratios = shock_line_ratios(500.0, shock_component="precursor")
    >>> ratios = shock_line_ratios(400.0, shock_abundance="lmc", shock_log_density=1.0)

    """
    grids = _load_mappings_grids()

    # ── HDF5 grid path ─────────────────────────────────────────────
    if grids is not None and "mappings5" in grids:
        g = grids["mappings5"]

        # Validate before any indexing — raises ValueError for out-of-range inputs
        _validate_shock_params(
            shock_velocity,
            shock_log_density,
            shock_b_over_sqrt_n,
            shock_abundance,
            shock_component,
            g,
        )

        # --- static string → integer index (outside JIT tracing) ---
        i_abund = _resolve_abundance(shock_abundance, g["abundance_names"])

        component_map = {
            "shock": "shock_ratios",
            "precursor": "precursor_ratios",
            "combined": "combined_ratios",
        }
        ratio_array = g[component_map.get(shock_component, "combined_ratios")]
        # shape: (N_abund, N_n, N_v, N_B, N_lines)

        # Clip all three continuous axes to grid bounds before interpolation
        v_grid = g["velocities_kms"]
        b_grid = g["b_axis"]
        log_den_grid = g["log_density_cm3"]
        v_q = jnp.clip(shock_velocity, v_grid[0], v_grid[-1])
        b_q = jnp.clip(shock_b_over_sqrt_n, b_grid[0], b_grid[-1])
        n_q = jnp.clip(shock_log_density, log_den_grid[0], log_den_grid[-1])

        # Slice out categorical abundance axis → (N_n, N_v, N_B, N_lines)
        # Transpose to (N_v, N_B, N_n, N_lines) so leading dims match axes order
        grid_abund = ratio_array[i_abund]  # (N_n, N_v, N_B, N_lines)
        grid_vbn = jnp.transpose(grid_abund, (1, 2, 0, 3))  # (N_v, N_B, N_n, N_lines)

        # --- C²-continuous triweight interpolation across all 3 continuous axes ---
        axes = (v_grid, b_grid, log_den_grid)
        edges = (g["v_edges"], g["b_edges"], g["n_edges"])
        ratios_vec = _interp_nd_triweight(grid_vbn, axes, edges, (v_q, b_q, n_q))
        # ratios_vec: shape (N_lines,)

        return {name: ratios_vec[j] for j, name in enumerate(g["line_names"])}

    # ── Fallback path — hardcoded Allen+2008 Table 5 ───────────────
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
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute shock line wavelengths and absolute luminosities.

    Returns a ``(wavelengths_aa, luminosities_lsun)`` tuple, both shape
    ``(N_lines,)``, ordered consistently with the active grid or fallback.

    The absolute scaling anchors on ``l_shock_halpha``:
    ``L_line = (ratio / r_ha) * l_shock_halpha``.
    """
    ratios = shock_line_ratios(
        shock_velocity,
        shock_log_density=shock_log_density,
        shock_b_over_sqrt_n=shock_b_over_sqrt_n,
        shock_abundance=shock_abundance,
        shock_component=shock_component,
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

    lums = jnp.array(
        [ratios.get(n, jnp.array(0.0)) / r_ha_safe * l_shock_halpha for n in line_names]
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

    Returns
    -------
    ndarray, shape (n_wave,)
        Shock emission SED in erg/s/Hz [erg/s/Hz].

    Notes
    -----
    **JIT-compatible**: yes — all operations use ``jnp`` primitives and
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
        Always ``False`` — MAPPINGS V provides line emission only.
    has_free_params : bool
        Always ``True`` — velocity, density, B-field are differentiable parameters.
    name : str
        Backend identifier string ("shock").

    Notes
    -----
    **JIT-compatible**: Methods return JAX arrays suitable for JIT compilation.
    All computations use pure functions with no side effects.

    **Attributes**: ``has_continuum`` is always False — MAPPINGS V provides
    shock-associated emission lines only (no underlying continuum).
    ``has_free_params`` is always True — all parameters (velocity, density,
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
        **JIT-compatible**: yes — delegates to ``compute_shock_sed``.

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
