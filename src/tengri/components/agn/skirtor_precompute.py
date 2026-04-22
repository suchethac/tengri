"""Precompute adapter for SKIRTOR AGN torus templates.

Implements :class:`~tengri.forward.precompute.protocol.PrecomputeModule` for
SKIRTOR's 5D torus template grid: (tau, p, q, oa, cos_inc).

Auto-collapses any axis whose corresponding parameter is
:class:`~tengri.parameters.priors.Fixed` in the user's ``Parameters`` — e.g., a
user who pins ``agn_tau_skirtor`` and ``agn_p_skirtor`` gets a 3D runtime grid
instead of the full 5D one, for free.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from tengri.components.agn._phys import C_LIGHT as _C_CGS, LSUN_ERG as _LSUN_ERG
from tengri.utils.grid_interp import (
    PreintegratedGrid,
    interp_nd_triweight,
    preintegrate_grid,
    slice_fixed_axes,
)
from tengri.utils.interpolation import edges_for_grid

# Parameter names corresponding to the 5 SKIRTOR grid axes, in order.
AXIS_PARAMS: tuple[str, ...] = (
    "agn_tau_skirtor",
    "agn_p_skirtor",
    "agn_q_skirtor",
    "agn_oa_skirtor",
    "agn_cos_inc",
)


def precompute_skirtor_photometry(
    grid_path: str,
    filter_waves: list[jnp.ndarray],
    filter_trans: list[jnp.ndarray],
    redshift: float = 0.0,
) -> dict:
    """Pre-integrate SKIRTOR templates through filter curves.

    For each 5D grid point (tau, p, q, oa, cos_inc), compute the
    filter-integrated photometry.  Returns a dict with ``grid_phot``
    and ``axes``.

    Templates are frequency-normalized (matching the runtime normalization
    in ``skirtor.py``) so that ``build_skirtor_photometry_lookup`` returns
    L_ν [erg/s/Hz] per L_sun of bolometric luminosity, consistent with the
    full-wavelength ``agn_emission`` path.

    Parameters
    ----------
    grid_path : str
        Path to ``skirtor_templates.npz``.
    filter_waves, filter_trans : list of array
        Filter curves in Angstrom / relative transmission, **observed frame**.
    redshift : float
        Source redshift.  Used to shift rest-frame templates into the
        observed frame before integrating against observed-frame filters.

    Returns
    -------
    dict
        ``grid_phot`` : array (n_tau, n_p, n_q, n_oa, n_inc, n_filters)
            Filter-integrated L_ν [erg/s/Hz] per L_sun (unit torus fraction).
        ``axes`` : tuple of 5 grid arrays (jnp.ndarray)
        ``_preint`` : :class:`PreintegratedGrid`

    Notes
    -----
    **Build-time operation**: This function performs frequency-domain
    integration via NumPy. The precomputed photometry is grid-independent
    (depends only on filter curves and redshift, not wavelength grid).

    **Normalization**: Templates are frequency-normalized so that the
    integration constant equals L_sun / trapz(template, nu). This matches
    the runtime normalization in ``skirtor.py`` where user luminosity is
    multiplied by the precomputed template per filter.

    **Redshift handling**: Redshift shifts rest-frame templates into
    observed frame before integrating against observed-frame filters.
    For dust-only templates (no redshift effects on the template shape),
    this is the standard flux-scaling convention.
    """
    from tengri.components.agn.skirtor import _load_grid_arrays

    raw = _load_grid_arrays(grid_path)
    grid = np.asarray(raw["total"], dtype=np.float64)
    wave_grid = np.asarray(raw["wave"], dtype=np.float64)
    axes_np = tuple(np.asarray(ax) for ax in raw["axes"])

    # Convert raw dimensionless templates to L_ν [erg/s/Hz per L_sun of L_bol].
    # This matches the normalization in skirtor.py:
    #   l_nu_erg = l_bol_erg * torus_frac * template / trapz(template, nu)
    # Precomputed: lnu_per_lsun = LSUN_ERG * template / trapz(template, nu)
    # Runtime: l_bol_lsun * torus_frac * lnu_per_lsun  →  L_ν [erg/s/Hz]
    nu_grid = _C_CGS / (wave_grid * 1e-8)  # Hz (decreasing order)
    sort_idx = np.argsort(nu_grid)
    nu_sorted = nu_grid[sort_idx]

    *grid_dims, n_wave = grid.shape
    n_pts = int(np.prod(grid_dims)) if grid_dims else 1
    grid_flat = np.array(grid, dtype=np.float64).reshape(n_pts, n_wave)
    lnu_flat = np.empty_like(grid_flat)

    for i in range(n_pts):
        template = grid_flat[i]
        integral = np.trapezoid(template[sort_idx], nu_sorted)
        integral_safe = max(abs(integral), 1e-100)
        lnu_flat[i] = _LSUN_ERG * template / integral_safe

    lnu_grid = lnu_flat.reshape(*grid_dims, n_wave)

    preint = preintegrate_grid(
        templates=lnu_grid,
        wave_rest=wave_grid,
        filter_waves=[np.asarray(fw) for fw in filter_waves],
        filter_trans=[np.asarray(ft) for ft in filter_trans],
        redshift=redshift,
        dl_cm=1.0,
        axes=axes_np,
        energy_normalize=False,  # templates already normalized per L_sun
    )

    axes_jax = tuple(jnp.asarray(ax) for ax in axes_np)
    return {
        "grid_phot": preint.phot,
        "axes": axes_jax,
        "_preint": preint,
    }


def build_skirtor_photometry_lookup(precomp: dict):
    """Build a JIT-compiled SKIRTOR torus photometry function.

    Uses triweight interpolation for C²-continuous gradients.

    Parameters
    ----------
    precomp : dict
        Output of :func:`precompute_skirtor_photometry` or :func:`precompute`
        (the Protocol-shaped entry point).

    Returns
    -------
    callable
        ``(agn_log_lbol, agn_tau_skirtor, agn_p_skirtor, agn_q_skirtor,
           agn_oa_skirtor, agn_cos_inc, agn_torus_frac) -> array (n_filters,)``
        Returns torus L_ν [erg/s/Hz].  Caller applies

    Notes
    -----
    **JIT-compatible**: yes — the returned function uses ``jnp`` and
    triweight interpolation, which are JAX-native.

    **Interpolation kernel**: Triweight kernel provides C²-continuous
    gradients for autodiff, unlike nearest-neighbor or linear interpolation.
    This is important for robust inference when SKIRTOR parameters are
    fitted via gradient descent.
        ``flux_scale = (1+z) / (4π d_L²)`` to get flux density.
    """
    grid_phot = precomp["grid_phot"]
    axes = precomp["axes"]
    edges = tuple(edges_for_grid(ax) for ax in axes)

    @jax.jit
    def skirtor_phot(
        agn_log_lbol,
        agn_tau_skirtor,
        agn_p_skirtor,
        agn_q_skirtor,
        agn_oa_skirtor,
        agn_cos_inc,
        agn_torus_frac,
    ):
        """Compute SKIRTOR torus photometry via triweight interpolation on 5D grid.

        Returns filter-integrated L_nu [erg/s/Hz] at runtime.
        """
        # grid_phot stores L_ν [erg/s/Hz] per L_sun of L_bol (unit torus fraction)
        # Return: L_bol_lsun [L_sun] * torus_frac * phot [erg/s/Hz/L_sun] = L_ν [erg/s/Hz]
        l_bol_lsun = 10.0**agn_log_lbol
        point = (
            agn_tau_skirtor,
            agn_p_skirtor,
            agn_q_skirtor,
            agn_oa_skirtor,
            agn_cos_inc,
        )
        phot_per_lsun = interp_nd_triweight(grid_phot, axes, edges, point)
        return l_bol_lsun * agn_torus_frac * phot_per_lsun

    return skirtor_phot


# ── Protocol-shaped entry points ──────────────────────────────────


def precompute(
    filter_waves: list,
    filter_trans: list,
    redshift: float,
    parameters: Any,
    *,
    grid_path: str,
) -> dict:
    """Build preintegrated SKIRTOR grid, auto-collapsing Fixed-parameter axes.

    Parameters
    ----------
    filter_waves, filter_trans : list
        Filter curves (observed frame).
    redshift : float
        Unused — SKIRTOR templates are rest-frame and the filter grid is
        already observed-frame.  Accepted for Protocol consistency.
    parameters : Parameters | None
        Parameters spec, used to detect Fixed-axis parameters.
    grid_path : str
        Path to ``skirtor_templates.npz``.

    Returns
    -------
    dict
        Same shape as :func:`precompute_skirtor_photometry` but with grid
        axes collapsed for any Fixed :data:`AXIS_PARAMS` entry.
    """
    result = precompute_skirtor_photometry(
        grid_path, filter_waves, filter_trans, redshift=redshift
    )
    if parameters is None:
        return result

    preint: PreintegratedGrid = result["_preint"]
    fixed: dict[int, float] = {}
    for i, pname in enumerate(AXIS_PARAMS):
        if parameters.is_fixed(pname):
            fixed[i] = float(parameters.fixed_value(pname))
    if not fixed:
        return result

    collapsed = slice_fixed_axes(preint, fixed)
    # Rebuild dict view; drop the axes that were collapsed
    remaining_axes = tuple(ax for i, ax in enumerate(result["axes"]) if i not in fixed)
    return {
        "grid_phot": collapsed.phot,
        "axes": remaining_axes,
        "_preint": collapsed,
        "_collapsed_axes": fixed,
    }


def build_lookup(preint: dict, *, free_param_names: tuple[str, ...] | None = None):
    """Build the runtime SKIRTOR photometry lookup from a preintegrated dict.

    When no axes are collapsed, delegates to
    :func:`build_skirtor_photometry_lookup`.  When some axes are collapsed
    (fixed at preintegration time), the returned function expects only the
    remaining free parameter values.

    Parameters
    ----------
    preint : dict
        Preintegrated data dict with keys ``"grid_phot"``, ``"axes"``,
        and optionally ``"_collapsed_axes"`` and ``"_preint"``.
    free_param_names : tuple of str or None
        Names of remaining free axes in the collapsed case.
        Not used in the default (no-collapse) case.

    Returns
    -------
    Callable
        JIT-compiled photometry lookup function with signature
        ``(agn_log_lbol, *free_axis_values, agn_torus_frac) -> ndarray``.
    """
    if not preint.get("_collapsed_axes"):
        return build_skirtor_photometry_lookup(preint)

    # Collapsed case: lookup takes (scale, *remaining_axis_values, torus_frac)
    grid_phot = preint["grid_phot"]
    axes = preint["axes"]
    edges = tuple(edges_for_grid(ax) for ax in axes)

    @jax.jit
    def skirtor_phot_collapsed(agn_log_lbol, *free_axis_values, agn_torus_frac):
        """Compute SKIRTOR torus photometry with collapsed (fixed) axes via triweight interp.

        Returns filter-integrated L_nu [erg/s/Hz] at runtime.
        """
        # Same unit convention as build_skirtor_photometry_lookup: L_ν [erg/s/Hz]
        l_bol_lsun = 10.0**agn_log_lbol
        phot_per_lsun = interp_nd_triweight(grid_phot, axes, edges, tuple(free_axis_values))
        return l_bol_lsun * agn_torus_frac * phot_per_lsun

    return skirtor_phot_collapsed
