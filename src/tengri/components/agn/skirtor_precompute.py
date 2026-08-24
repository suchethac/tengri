# SPDX-License-Identifier: BSD-3-Clause
"""Precompute adapter for SKIRTOR AGN torus templates.

Implements :class:`~tengri.forward.precompute.protocol.PrecomputeModule` for
SKIRTOR's 5D torus template grid: (tau, p, q, oa, cos_inc).

Auto-collapses any axis whose corresponding parameter is
:class:`~tengri.parameters.priors.Fixed` in the user's ``Parameters``: e.g., a
user who pins ``agn_tau_skirtor`` and ``agn_p_skirtor`` gets a 3D runtime grid
instead of the full 5D one, for free.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from tengri.components._collapsed_lookup import interp_collapsed
from tengri.components.agn._phys import C_LIGHT as _C_CGS
from tengri.utils.grid_interp import (
    PreintegratedGrid,
    interp_nd_triweight,
    preintegrate_grid,
    slice_fixed_axes,
)
from tengri.utils.interpolation import edges_for_grid
from tengri.utils.physics_constants import L_SUN as _LSUN_ERG

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

    Templates are wavelength-normalized and converted L_λ → L_ν (matching the
    runtime normalization in ``skirtor.py`` after issue #459) so that
    ``build_skirtor_photometry_lookup`` returns L_ν [erg/s/Hz] per L_sun of
    bolometric luminosity, consistent with the full-wavelength ``agn_emission``
    path.

    Parameters
    ----------
    grid_path: str
        Path to ``skirtor_templates.npz`` or ``.h5``.
    filter_waves: list[ndarray]
        Wavelength grid per filter [Angstrom], observed frame.
    filter_trans: list[ndarray]
        Transmission per filter (0–1).
    redshift: float, optional
        Source redshift. Used to shift rest-frame templates into the
        observed frame before integrating against observed-frame filters.
        Default 0.0.

    Returns
    -------
    dict
        ``grid_phot``: ndarray, shape (n_tau, n_p, n_q, n_oa, n_inc, n_filters)
            Filter-integrated L_ν [erg/s/Hz] per L_sun (unit torus fraction).
        ``axes``: tuple of 5 grid arrays (jnp.ndarray)
            Grid axes (tau, p, q, oa, cos_inc).
        ``_preint``: PreintegratedGrid
            Internal preintegration data structure.

    References
    ----------
    .. [1] M. Stalevski et al., "3D radiative transfer modeling of the dusty
           torus around AGN, the influence of clumping," MNRAS, 420, 2756 (2012).
           arXiv:1109.1286. https://doi.org/10.1111/j.1365-2966.2011.19775.x
    .. [2] M. Stalevski et al., "The dust covering factor in AGN: combining the
           IR torus emission with polar dust component," MNRAS, 458, 2288 (2016).
           arXiv:1602.01954. https://doi.org/10.1093/mnras/stw444

    Notes
    -----
    **JIT-compatible**: no, this is a build-time function using NumPy.

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

    # The photometry precompute pins the SKIRTOR radius ratio R to the CIGALE
    # default (R=20) and stays 5-D (tau, p, q, oa, cos_inc). It is an opt-in
    # speed approximation; the exact spectral path interpolates R fully (#772).
    # ``_load_grid_arrays`` always returns the R axis at position 4 (a degenerate
    # length-1 axis for legacy grids), so this node-exact slice is safe.
    if len(axes_np) == 6:
        r_idx = int(np.argmin(np.abs(axes_np[4] - 20.0)))
        grid = grid[:, :, :, :, r_idx, :]
        axes_np = axes_np[:4] + axes_np[5:]

    # Convert raw L_λ-like templates to L_ν [erg/s/Hz per L_sun of L_bol].
    # SKIRTOR v3 templates are stored as L_λ-like (issue #459), so the
    # bolometric normalization must be taken in the *wavelength* variable and
    # the result converted L_λ → L_ν = L_λ × λ²/c at the end: exactly as the
    # runtime path in ``skirtor.py:_interpolate_and_normalize`` does. The
    # previous frequency-integral here treated the L_λ array as L_ν and left
    # the precomputed LUT with the wrong dimensionality (the WavePrecomp path
    # missed the #459 fix that landed in skirtor.py).
    #   Precomputed: lnu_per_lsun = LSUN_ERG * template / trapz(template, λ) × λ²/c
    #   Runtime:     l_bol_lsun * torus_frac * lnu_per_lsun  →  L_ν [erg/s/Hz]
    c_aa_per_s = _C_CGS * 1e8  # cm/s → Å/s

    *grid_dims, n_wave = grid.shape
    n_pts = int(np.prod(grid_dims)) if grid_dims else 1
    grid_flat = np.array(grid, dtype=np.float64).reshape(n_pts, n_wave)
    lnu_flat = np.empty_like(grid_flat)

    # ``wave_grid`` is monotonically ascending (set in ``_load_grid_arrays``),
    # so trapezoid integrates correctly in λ without an explicit sort.
    for i in range(n_pts):
        template = grid_flat[i]
        integral_lam = np.trapezoid(template, wave_grid)
        integral_safe = max(abs(integral_lam), 1e-100)
        template_lam = _LSUN_ERG * template / integral_safe  # erg/s/Å per L_sun
        lnu_flat[i] = template_lam * wave_grid**2 / c_aa_per_s  # L_λ → L_ν

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


def build_skirtor_photometry_lookup(precomp: dict, grid_arrays_traced: tuple | None = None):
    """Build a JIT-compiled SKIRTOR torus photometry function.

    Uses triweight interpolation for C²-continuous gradients.

    Parameters
    ----------
    precomp: dict
        Output of :func:`precompute_skirtor_photometry` or :func:`precompute`
        (the Protocol-shaped entry point).
    grid_arrays_traced: tuple or None, optional
        Tuple of (grid_phot, axes) to pass as JIT-traced arguments instead of
        capturing them in closure. When None (default), captures from precomp
        for backward compatibility. Threading these as kwargs avoids closure-
        captured XLA constants. Default: None.

    Returns
    -------
    callable
        Function with signature::

            fn(agn_log_lbol, agn_tau_skirtor, agn_p_skirtor,
               agn_q_skirtor, agn_oa_skirtor, agn_cos_inc,
               agn_torus_frac, *grid_arrays_traced) -> ndarray, shape (n_filters,)

        When grid_arrays_traced is provided, the returned function expects
        (grid_phot, axes) as additional positional arguments.

        Returns torus L_ν [erg/s/Hz].  Caller applies
        ``flux_scale = (1+z) / (4π d_L²)`` to get flux density.

    References
    ----------
    .. [1] M. Stalevski et al., "3D radiative transfer modeling of the dusty
           torus around AGN, the influence of clumping," MNRAS, 420, 2756 (2012).
           arXiv:1109.1286. https://doi.org/10.1111/j.1365-2966.2011.19775.x

    Notes
    -----
    **JIT-compatible**: yes, the returned function uses ``jnp`` and
    triweight interpolation, which are JAX-native.

    **Gradient-safe**: yes, triweight kernel is fully differentiable.

    **Interpolation kernel**: Triweight kernel provides C²-continuous
    gradients for autodiff, unlike nearest-neighbor or linear interpolation.
    This is important for robust inference when SKIRTOR parameters are
    fitted via gradient descent.

    **Compile-time performance**: Passing grid_arrays_traced avoids closure-
    captured constants in XLA, reducing HLO size and compile time.
    """
    grid_phot = precomp["grid_phot"]
    axes = precomp["axes"]
    edges = tuple(edges_for_grid(ax) for ax in axes)

    if grid_arrays_traced is not None:
        # Runtime-passed arrays (JIT-traced): no closure capture
        @jax.jit
        def skirtor_phot(
            agn_log_lbol,
            agn_tau_skirtor,
            agn_p_skirtor,
            agn_q_skirtor,
            agn_oa_skirtor,
            agn_cos_inc,
            agn_torus_frac,
            grid_phot_traced,
            axes_traced,
        ):
            """Compute SKIRTOR torus photometry via triweight interpolation on 5D grid.

            Returns filter-integrated L_nu [erg/s/Hz] at runtime.
            """
            edges_traced = tuple(edges_for_grid(ax) for ax in axes_traced)
            l_bol_lsun = 10.0**agn_log_lbol
            point = (
                agn_tau_skirtor,
                agn_p_skirtor,
                agn_q_skirtor,
                agn_oa_skirtor,
                agn_cos_inc,
            )
            phot_per_lsun = interp_nd_triweight(grid_phot_traced, axes_traced, edges_traced, point)
            return l_bol_lsun * agn_torus_frac * phot_per_lsun

        # Return wrapper that inserts traced arrays
        def wrapper(
            agn_log_lbol,
            agn_tau_skirtor,
            agn_p_skirtor,
            agn_q_skirtor,
            agn_oa_skirtor,
            agn_cos_inc,
            agn_torus_frac,
        ):
            return skirtor_phot(
                agn_log_lbol,
                agn_tau_skirtor,
                agn_p_skirtor,
                agn_q_skirtor,
                agn_oa_skirtor,
                agn_cos_inc,
                agn_torus_frac,
                grid_arrays_traced[0],
                grid_arrays_traced[1],
            )

        return wrapper

    # Backward-compatibility: closure-captured (original behavior)
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
    filter_waves: list[ndarray]
        Wavelength grid per filter [Angstrom], observed frame.
    filter_trans: list[ndarray]
        Transmission per filter (0–1).
    redshift: float
        Source redshift. [dimensionless]
    parameters: Parameters | None
        Parameters spec, used to detect Fixed-axis parameters.
    grid_path: str, keyword-only
        Path to ``skirtor_templates.npz`` or ``.h5``.

    Returns
    -------
    dict
        Same shape as :func:`precompute_skirtor_photometry` but with grid
        axes collapsed for any Fixed :data:`AXIS_PARAMS` entry.

    References
    ----------
    .. [1] M. Stalevski et al., "3D radiative transfer modeling of the dusty
           torus around AGN, the influence of clumping," MNRAS, 420, 2756 (2012).
           arXiv:1109.1286. https://doi.org/10.1111/j.1365-2966.2011.19775.x

    Notes
    -----
    **JIT-compatible**: no, this is a build-time function using NumPy.
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
    preint: dict
        Preintegrated data dict with keys ``"grid_phot"``, ``"axes"``,
        and optionally ``"_collapsed_axes"`` and ``"_preint"``.
    free_param_names: tuple of str or None, optional
        Names of remaining free axes in the collapsed case.
        Not used in the default (no-collapse) case.

    Returns
    -------
    callable
        JIT-compiled photometry lookup function with signature::

            fn(agn_log_lbol, *free_axis_values, agn_torus_frac)
                -> ndarray, shape (n_filters,)

        Returns torus L_ν [erg/s/Hz].

    References
    ----------
    .. [1] M. Stalevski et al., "3D radiative transfer modeling of the dusty
           torus around AGN, the influence of clumping," MNRAS, 420, 2756 (2012).
           arXiv:1109.1286. https://doi.org/10.1111/j.1365-2966.2011.19775.x

    Notes
    -----
    **JIT-compatible**: yes, the returned function is fully JAX-native.

    **Gradient-safe**: yes, triweight interpolation is fully differentiable.
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
        phot_per_lsun = interp_collapsed(
            grid_phot, axes, free_axis_values, kernel="triweight", edges=edges
        )
        return l_bol_lsun * agn_torus_frac * phot_per_lsun

    return skirtor_phot_collapsed
