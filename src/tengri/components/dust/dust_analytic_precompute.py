# SPDX-License-Identifier: BSD-3-Clause
"""Precompute adapters for analytic dust-emission models.

Implements :class:`~tengri.forward.precompute.protocol.PrecomputeModule` for
three analytic dust-emission models:

1. **modified_blackbody**: Optically-thin modified blackbody SED.
   Two free axes: ``dust_T`` (temperature), ``dust_beta_ir`` (emissivity index).

2. **casey2012**: Casey (2012) modified blackbody + mid-IR power law.
   Three free axes: ``dust_T`` (temperature), ``dust_beta_ir`` (emissivity index),
   ``dust_alpha_mir`` (mid-IR power-law slope).

3. **pah_drude**: Sum of 18 PAH Drude profiles (Smith et al. 2007).
   No free axes (pure template; runtime amplitudes scale the combined PAH profile).

Each model is preintegrated through filter curves at model-initialization time.
Auto-collapses axes whose corresponding parameters are ``Fixed`` in the user's
``Parameters``: e.g., a user who pins ``dust_T`` gets a 1D grid.

References
----------
.. [1] Casey, C. M., "Dusty star-forming galaxies at high redshift,"
       MNRAS, 425, 3094 (2012). arXiv:1206.1595.
       https://doi.org/10.1111/j.1365-2966.2012.21455.x
.. [2] Smith, J. D., et al., "The mid-infrared emission of ultraluminous
       infrared galaxies," ApJ, 656, 770 (2007). arXiv:astro-ph/0701042.
       https://doi.org/10.1086/510378
.. [3] Hildebrand, R. H., "The determination of cloud masses from dust continuum
       emission," QJRAS, 24, 267 (1983).

"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from tengri.components.dust.drude_profiles import compute_pah_template as _compute_pah
from tengri.components.dust.emission import (
    casey2012 as _casey2012,
    modified_blackbody as _modified_blackbody,
)
from tengri.forward.precompute.templates import (
    build_template_photometry_lookup,
    collapse_fixed_axes,
    precompute_template_photometry,
)
from tengri.utils.grid_interp import PreintegratedGrid
from tengri.utils.physics_constants import C_CGS as _C_CGS

# ── Axis definitions per model ──────────────────────────────────

# modified_blackbody: parametrized by temperature and emissivity index
AXIS_PARAMS_MBB = ("dust_T", "dust_beta_ir")

# casey2012: parametrized by temperature, emissivity index, mid-IR slope
AXIS_PARAMS_CASEY = ("dust_T", "dust_beta_ir", "dust_alpha_mir")

# pah_drude: pure template; no grid axes (runtime amplitude scaling)
AXIS_PARAMS_PAH = ()

# Protocol-required dict form for multi-model module
AXIS_PARAMS: dict[str, tuple[str, ...]] = {
    "modified_blackbody": AXIS_PARAMS_MBB,
    "casey2012": AXIS_PARAMS_CASEY,
    "pah_drude": AXIS_PARAMS_PAH,
}


def _build_grid_modified_blackbody(
    filter_waves: list,
    filter_trans: list,
    redshift: float,
    T_grid: np.ndarray,
    beta_grid: np.ndarray,
    L_absorbed_ref: float = 1.0,
) -> PreintegratedGrid:
    """Preintegrate modified blackbody over a 2D grid of (T, beta) values.

    Parameters
    ----------
    filter_waves : list[ndarray]
        Per-filter wavelength arrays [Angstrom].
    filter_trans : list[ndarray]
        Per-filter transmission curves.
    redshift : float
        Source redshift.
    T_grid : ndarray, shape (n_T,)
        Temperature grid [K].
    beta_grid : ndarray, shape (n_beta,)
        Emissivity-index grid [dimensionless].
    L_absorbed_ref : float
        Reference absorbed luminosity for normalization [L_sun]. Default 1.0.

    Returns
    -------
    PreintegratedGrid
        Preintegrated photometry with shape (n_T, n_beta, n_filters).
    """
    T_grid = np.asarray(T_grid, dtype=np.float64)
    beta_grid = np.asarray(beta_grid, dtype=np.float64)

    # Standard rest-frame wavelength grid for integration
    wave_rest = np.logspace(2, 5.5, 1000, dtype=np.float64)

    # Precompute L_nu for each (T, beta) grid point
    phot_grid = []
    for T in T_grid:
        phot_beta = []
        for beta in beta_grid:
            l_nu = np.asarray(
                _modified_blackbody(
                    jnp.asarray(wave_rest),
                    L_absorbed=L_absorbed_ref,
                    dust_T=float(T),
                    dust_beta_ir=float(beta),
                    redshift=float(redshift),
                )
            )
            phot_beta.append(l_nu)
        phot_grid.append(phot_beta)

    templates = np.array(phot_grid, dtype=np.float64)  # (n_T, n_beta, n_wave)

    # Preintegrate through filters using template helper
    return precompute_template_photometry(
        templates=templates,
        wave_rest=wave_rest,
        filter_waves=[np.asarray(fw, dtype=np.float64) for fw in filter_waves],
        filter_trans=[np.asarray(ft, dtype=np.float64) for ft in filter_trans],
        axes=(T_grid, beta_grid),
        redshift=0.0,  # redshift already baked into L_nu via CMB correction
        dl_cm=1.0,
        energy_normalize=False,  # already normalized to L_absorbed_ref per model
        units="lnu",
    )


def _build_grid_casey2012(
    filter_waves: list,
    filter_trans: list,
    redshift: float,
    T_grid: np.ndarray,
    beta_grid: np.ndarray,
    alpha_mir_grid: np.ndarray,
    L_absorbed_ref: float = 1.0,
) -> PreintegratedGrid:
    """Preintegrate casey2012 over a 3D grid of (T, beta, alpha_mir) values.

    Parameters
    ----------
    filter_waves : list[ndarray]
        Per-filter wavelength arrays [Angstrom].
    filter_trans : list[ndarray]
        Per-filter transmission curves.
    redshift : float
        Source redshift.
    T_grid : ndarray, shape (n_T,)
        Temperature grid [K].
    beta_grid : ndarray, shape (n_beta,)
        Emissivity-index grid [dimensionless].
    alpha_mir_grid : ndarray, shape (n_alpha,)
        Mid-IR power-law slope grid [dimensionless].
    L_absorbed_ref : float
        Reference absorbed luminosity for normalization [L_sun]. Default 1.0.

    Returns
    -------
    PreintegratedGrid
        Preintegrated photometry with shape (n_T, n_beta, n_alpha, n_filters).
    """
    T_grid = np.asarray(T_grid, dtype=np.float64)
    beta_grid = np.asarray(beta_grid, dtype=np.float64)
    alpha_mir_grid = np.asarray(alpha_mir_grid, dtype=np.float64)

    # Standard rest-frame wavelength grid for integration
    wave_rest = np.logspace(2, 5.5, 1000, dtype=np.float64)

    # Precompute L_nu for each (T, beta, alpha_mir) grid point
    phot_grid = []
    for T in T_grid:
        phot_beta = []
        for beta in beta_grid:
            phot_alpha = []
            for alpha_mir in alpha_mir_grid:
                l_nu = np.asarray(
                    _casey2012(
                        jnp.asarray(wave_rest),
                        L_absorbed=L_absorbed_ref,
                        dust_T=float(T),
                        dust_beta_ir=float(beta),
                        dust_alpha_mir=float(alpha_mir),
                        redshift=float(redshift),
                    )
                )
                phot_alpha.append(l_nu)
            phot_beta.append(phot_alpha)
        phot_grid.append(phot_beta)

    templates = np.array(phot_grid, dtype=np.float64)  # (n_T, n_beta, n_alpha, n_wave)

    # Preintegrate through filters using template helper
    return precompute_template_photometry(
        templates=templates,
        wave_rest=wave_rest,
        filter_waves=[np.asarray(fw, dtype=np.float64) for fw in filter_waves],
        filter_trans=[np.asarray(ft, dtype=np.float64) for ft in filter_trans],
        axes=(T_grid, beta_grid, alpha_mir_grid),
        redshift=0.0,  # redshift already baked into L_nu via CMB correction
        dl_cm=1.0,
        energy_normalize=False,  # already normalized to L_absorbed_ref per model
        units="lnu",
    )


def _build_grid_pah_drude(
    filter_waves: list,
    filter_trans: list,
    redshift: float,
    L_absorbed_ref: float = 1.0,
) -> PreintegratedGrid:
    """Preintegrate PAH Drude template through filters.

    The PAH template is pure shape (no axes); runtime amplitude scales it.
    Precomputes the filter-integrated template so the hybrid kernel can scale
    by the user's PAH amplitude parameter at runtime.

    Parameters
    ----------
    filter_waves : list[ndarray]
        Per-filter wavelength arrays [Angstrom].
    filter_trans : list[ndarray]
        Per-filter transmission curves.
    redshift : float
        Source redshift.
    L_absorbed_ref : float
        Reference absorbed luminosity for normalization [L_sun]. Default 1.0.

    Returns
    -------
    PreintegratedGrid
        Preintegrated photometry with shape (1, n_filters) (scalar template).
    """
    # Standard rest-frame wavelength grid for integration
    wave_rest = np.logspace(2, 5.5, 1000, dtype=np.float64)

    # Compute PAH template using Smith+2007 SINGS median strengths
    pah_llam = np.asarray(_compute_pah(jnp.asarray(wave_rest * 1e-4)))  # Å -> μm

    # L_nu = L_lambda * lambda^2 / c. PAH template is dimensionless (relative);
    # scale to L_absorbed_ref.
    wave_cm = wave_rest * 1e-8
    lnu = L_absorbed_ref * pah_llam * (wave_cm**2) / _C_CGS

    # Wrap in shape (1, n_wave) for template compatibility
    templates = np.array([lnu], dtype=np.float64)

    # No grid axes for PAH: preintegrate as a single template
    return precompute_template_photometry(
        templates=templates,
        wave_rest=wave_rest,
        filter_waves=[np.asarray(fw, dtype=np.float64) for fw in filter_waves],
        filter_trans=[np.asarray(ft, dtype=np.float64) for ft in filter_trans],
        axes=(),  # No axes: scalar template
        redshift=0.0,  # redshift correction deferred to runtime
        dl_cm=1.0,
        energy_normalize=False,  # template already normalized to L_absorbed_ref
        units="lnu",
    )


# ── Protocol-shaped entry points ──────────────────────────────────


def precompute(
    filter_waves: list,
    filter_trans: list,
    redshift: float,
    parameters: Any,
    *,
    model: str = "modified_blackbody",
    T_grid: np.ndarray | None = None,
    beta_grid: np.ndarray | None = None,
    alpha_mir_grid: np.ndarray | None = None,
) -> dict:
    """Build preintegrated analytic dust grid, auto-collapsing Fixed-parameter axes.

    Multi-model entry point. Dispatches to the appropriate builder based
    on ``model`` parameter.

    Parameters
    ----------
    filter_waves : list[ndarray]
        Wavelength grid per filter [Angstrom], observed frame.
    filter_trans : list[ndarray]
        Transmission per filter (0–1).
    redshift : float
        Source redshift. [dimensionless]
    parameters : Parameters | None
        Parameters spec, used to detect Fixed-axis parameters.
    model : str, keyword-only
        One of "modified_blackbody", "casey2012", "pah_drude".
        Default: "modified_blackbody".
    T_grid : ndarray, optional
        Temperature grid for modified_blackbody/casey2012 [K]. If None, uses a
        default range [20, 60] with 9 points.
    beta_grid : ndarray, optional
        Emissivity-index grid [dimensionless]. If None, uses [1.5, 1.8, 2.0].
    alpha_mir_grid : ndarray, optional
        Mid-IR power-law slope grid for casey2012 [dimensionless]. If None,
        uses [1.5, 2.0, 2.5].

    Returns
    -------
    dict
        Keys: "grid_phot" (photometry array), "axes" (free axes), "_preint"
        (PreintegratedGrid), optionally "_collapsed_axes" (if any axes fixed).

    References
    ----------
    .. [1] Casey, C. M., "Dusty star-forming galaxies at high redshift,"
           MNRAS, 425, 3094 (2012).
    .. [2] Smith, J. D., et al., "The mid-infrared emission of ultraluminous
           infrared galaxies," ApJ, 656, 770 (2007).

    Notes
    -----
    **JIT-compatible**: no, this is a build-time function using NumPy.
    """
    if model == "modified_blackbody":
        if T_grid is None:
            T_grid = np.linspace(20.0, 60.0, 9, dtype=np.float64)
        if beta_grid is None:
            beta_grid = np.array([1.5, 1.8, 2.0], dtype=np.float64)
        result = {
            "grid_phot": _build_grid_modified_blackbody(
                filter_waves, filter_trans, redshift, T_grid, beta_grid
            ).phot,
            "axes": (jnp.asarray(T_grid), jnp.asarray(beta_grid)),
            "_preint": _build_grid_modified_blackbody(
                filter_waves, filter_trans, redshift, T_grid, beta_grid
            ),
        }
        axis_params = AXIS_PARAMS_MBB

    elif model == "casey2012":
        if T_grid is None:
            T_grid = np.linspace(25.0, 60.0, 8, dtype=np.float64)
        if beta_grid is None:
            beta_grid = np.array([1.5, 1.8, 2.0], dtype=np.float64)
        if alpha_mir_grid is None:
            alpha_mir_grid = np.array([1.5, 2.0, 2.5], dtype=np.float64)
        result = {
            "grid_phot": _build_grid_casey2012(
                filter_waves, filter_trans, redshift, T_grid, beta_grid, alpha_mir_grid
            ).phot,
            "axes": (
                jnp.asarray(T_grid),
                jnp.asarray(beta_grid),
                jnp.asarray(alpha_mir_grid),
            ),
            "_preint": _build_grid_casey2012(
                filter_waves, filter_trans, redshift, T_grid, beta_grid, alpha_mir_grid
            ),
        }
        axis_params = AXIS_PARAMS_CASEY

    elif model == "pah_drude":
        result = {
            "grid_phot": _build_grid_pah_drude(filter_waves, filter_trans, redshift).phot,
            "axes": (),
            "_preint": _build_grid_pah_drude(filter_waves, filter_trans, redshift),
        }
        axis_params = AXIS_PARAMS_PAH

    else:
        raise ValueError(f"Unknown analytic dust model: {model}")

    # Auto-collapse any Fixed axes
    preint: PreintegratedGrid = result["_preint"]
    collapsed, remaining_axes, fixed = collapse_fixed_axes(
        preint, axis_params, parameters, origin=f"dust_analytic_precompute[{model}]"
    )
    if not fixed:
        return result

    return {
        "grid_phot": collapsed.phot,
        "axes": remaining_axes,
        "_preint": collapsed,
        "_collapsed_axes": fixed,
    }


def build_lookup(
    preint: dict,
    *,
    model: str = "modified_blackbody",
    free_param_names: tuple[str, ...] | None = None,
):
    """Build the runtime analytic dust photometry lookup from a preintegrated dict.

    Delegates to the template helper for triweight interpolation.

    Parameters
    ----------
    preint : dict
        Preintegrated data dict with keys "grid_phot", "axes", optionally
        "_collapsed_axes".
    model : str, keyword-only
        Analytic dust model name (for documentation; not used in lookup logic).
    free_param_names : tuple of str, optional
        Names of remaining free axes in the collapsed case.
        Not used in the default (no-collapse) case.

    Returns
    -------
    callable
        JIT-compiled photometry lookup function with signature::

            fn(L_absorbed, *free_axis_values) -> ndarray, shape (n_filters,)

        Returns dust emission L_ν [erg/s/Hz]. Caller applies flux scaling.

    References
    ----------
    .. [1] Casey, C. M., "Dusty star-forming galaxies at high redshift,"
           MNRAS, 425, 3094 (2012).

    Notes
    -----
    **JIT-compatible**: yes, the returned function uses ``jnp`` and triweight
    interpolation.

    **Gradient-safe**: yes, triweight kernel is fully differentiable.
    """
    if not preint.get("_collapsed_axes"):
        # No axes collapsed: use template helper directly
        return build_template_photometry_lookup(preint["_preint"])

    # Collapsed case: return a wrapped lookup that takes remaining free params
    from tengri.components._collapsed_lookup import interp_collapsed
    from tengri.utils.interpolation import edges_for_grid

    grid_phot = preint["grid_phot"]
    axes = preint["axes"]
    if axes:
        edges = tuple(edges_for_grid(ax) for ax in axes)
    else:
        edges = ()

    @jax.jit
    def dust_phot_collapsed(L_absorbed, *free_axis_values):
        """Compute dust photometry with some axes collapsed (fixed).

        Returns filter-integrated L_nu [erg/s/Hz] at runtime.
        """
        normed = interp_collapsed(
            grid_phot, axes, free_axis_values, kernel="triweight", edges=edges
        )
        return L_absorbed * normed

    return dust_phot_collapsed
