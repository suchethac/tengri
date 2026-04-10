"""Precompute template-based model photometry through filters.

For any template model (DL07, Dale2014, DL14, SKIRTOR, Astrodust, BOSA,
THEMIS, etc.), integrating templates through filter curves at every forward
model call is expensive (~100μs).  By pre-integrating the template grid
through each filter at model init time, we reduce runtime to a cheap
multilinear interpolation + scalar scaling.

The generic ``precompute_template_photometry()`` function works for any
template grid shape.  Model-specific wrappers (``precompute_dl07_photometry``,
``precompute_skirtor_photometry``) handle loading and unit conversion.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from tengri.core.preintegrate import PreintegratedGrid, interp_nd_triweight, preintegrate_grid
from tengri.utils.physics_constants import AA_TO_CM as _AA_TO_CM, C_CGS as _C_CGS

# -------------------------------------------------------------------
# Generic template precomputation
# -------------------------------------------------------------------


def precompute_template_photometry(
    templates: np.ndarray,
    wave_rest: np.ndarray,
    filter_waves: list,
    filter_trans: list,
    axes: tuple[np.ndarray, ...],
    redshift: float = 0.0,
    dl_cm: float = 1.0,
    energy_normalize: bool = True,
    units: str = "lnu",
) -> PreintegratedGrid:
    """Preintegrate any template grid through photometric filters.

    Generic entry point for template-based components. Handles unit
    conversion (L_λ → L_ν if needed) and delegates to
    :func:`~tengri.core.preintegrate.preintegrate_grid`.

    Parameters
    ----------
    templates : ndarray
        Shape ``(*grid_dims, n_wave)``.  Template spectra.
    wave_rest : ndarray
        Shape ``(n_wave,)``.  Rest-frame wavelengths [Ångström].
    filter_waves : list[ndarray]
        Per-filter wavelength arrays (observed frame).
    filter_trans : list[ndarray]
        Per-filter transmission curves.
    axes : tuple[ndarray, ...]
        One array per grid dimension (for triweight interpolation).
    redshift : float
        Source redshift (0 for rest-frame templates).
    dl_cm : float
        Luminosity distance [cm] (1 for normalized templates).
    energy_normalize : bool
        Normalize each template to unit bolometric luminosity before
        integration.  Required for templates scaled by L_absorbed or
        L_bol at runtime (DL07, Dale, SKIRTOR, etc.).  Default True.
    units : str
        ``"lnu"`` if templates are in L_ν [erg/s/Hz], or ``"llam"``
        if templates are in L_λ [erg/s/Å].  When ``"llam"``, converts
        to L_ν via L_ν = L_λ × λ²/c before integration.

    Returns
    -------
    PreintegratedGrid
        Precomputed filter-integrated photometry with triweight axes/edges.
    """
    templates = np.asarray(templates)
    wave_rest = np.asarray(wave_rest)

    if units == "llam":
        wave_cm = wave_rest * _AA_TO_CM
        templates = templates * (wave_cm**2) / _C_CGS

    return preintegrate_grid(
        templates=templates,
        wave_rest=wave_rest,
        filter_waves=[np.asarray(fw) for fw in filter_waves],
        filter_trans=[np.asarray(ft) for ft in filter_trans],
        redshift=redshift,
        dl_cm=dl_cm,
        axes=tuple(np.asarray(ax) for ax in axes),
        energy_normalize=energy_normalize,
    )


def build_template_photometry_lookup(preint: PreintegratedGrid):
    """Build a JIT-compiled lookup from a preintegrated template grid.

    Uses triweight interpolation for C²-continuous gradients.  The
    returned function takes the grid parameters + a scalar scaling
    factor and returns photometry in (n_filters,).

    Parameters
    ----------
    preint : PreintegratedGrid
        Output of ``precompute_template_photometry()``.

    Returns
    -------
    callable
        ``(scale, *grid_params) -> array (n_filters,)``
        where *grid_params* are scalar query points along each axis.
    """
    phot = preint.phot
    axes = preint.axes
    edges = preint.edges

    @jax.jit
    def lookup(scale, *grid_params):
        normed = interp_nd_triweight(phot, axes, edges, grid_params)
        return scale * normed

    return lookup


# -------------------------------------------------------------------
# DL07 / DL14 template photometry precomputation
# -------------------------------------------------------------------


def precompute_dl07_photometry(
    templates: dict,
    filter_waves: list[jnp.ndarray],
    filter_trans: list[jnp.ndarray],
) -> dict:
    """Pre-integrate DL07 templates through filter curves.

    For each (qpah, umin) grid point, compute the filter-integrated
    photometry of the mixed template
    ``j_nu = (1-gamma)*single_U + gamma*powerlaw``.
    Since gamma is a runtime parameter, we store single_U and powerlaw
    photometry separately and mix at runtime.

    Parameters
    ----------
    templates : dict
        DL07 template arrays with keys: ``single_u``, ``powerlaw``,
        ``wavelength``, ``umin_grid``, ``qpah_grid``.
    filter_waves : list of array
        Filter wavelength arrays in Angstrom.
    filter_trans : list of array
        Filter transmission arrays.

    Returns
    -------
    dict
        ``single_u_phot`` : array (n_qpah, n_umin, n_filters)
        ``powerlaw_phot`` : array (n_qpah, n_umin, n_filters)
        ``umin_grid`` : array (n_umin,)
        ``qpah_grid`` : array (n_qpah,)
    """
    import numpy as np

    single_u = templates["single_u"]  # (n_qpah, n_umin, n_wave)
    powerlaw = templates["powerlaw"]  # (n_qpah, n_umin, n_wave)
    tmpl_wave = templates["wavelength"]  # (n_wave,) Angstrom
    umin_grid = templates["umin_grid"]
    qpah_grid = templates["qpah_grid"]

    # Convert templates from L_lambda to L_nu (rest frame)
    wave_cm = np.asarray(tmpl_wave) * _AA_TO_CM
    lnu_single_u = np.asarray(single_u) * (wave_cm**2) / _C_CGS
    lnu_powerlaw = np.asarray(powerlaw) * (wave_cm**2) / _C_CGS

    # Use preintegrate_grid with energy_normalize=True to normalize each
    # template to unit bolometric luminosity before filter integration.
    # This ensures templates can be scaled by L_absorbed at runtime.
    # axes=(qpah_grid, umin_grid) tells preintegrate_grid the grid coordinates.
    single_u_preint = preintegrate_grid(
        templates=lnu_single_u,
        wave_rest=np.asarray(tmpl_wave),
        filter_waves=[np.asarray(fw) for fw in filter_waves],
        filter_trans=[np.asarray(ft) for ft in filter_trans],
        redshift=0.0,
        dl_cm=1.0,
        axes=(np.asarray(qpah_grid), np.asarray(umin_grid)),
        energy_normalize=True,
    )

    powerlaw_preint = preintegrate_grid(
        templates=lnu_powerlaw,
        wave_rest=np.asarray(tmpl_wave),
        filter_waves=[np.asarray(fw) for fw in filter_waves],
        filter_trans=[np.asarray(ft) for ft in filter_trans],
        redshift=0.0,
        dl_cm=1.0,
        axes=(np.asarray(qpah_grid), np.asarray(umin_grid)),
        energy_normalize=True,
    )

    return {
        "single_u_phot": single_u_preint.phot,
        "powerlaw_phot": powerlaw_preint.phot,
        "umin_grid": umin_grid,
        "qpah_grid": qpah_grid,
    }


def build_dl07_photometry_lookup(precomp: dict):
    """Build a JIT-compiled DL07 photometry function from precomputed tables.

    Uses triweight interpolation for C²-continuous gradients (replaces
    the previous piecewise-linear bilinear interpolation).

    Parameters
    ----------
    precomp : dict
        Output of ``precompute_dl07_photometry()``.

    Returns
    -------
    callable
        ``(L_absorbed, dust_umin, dust_gamma_dl, dust_qpah) -> array (n_filters,)``
        Returns L_nu (Lsun/Hz) at each filter.
    """
    from tengri.utils.interpolation import edges_for_grid

    single_u_phot = precomp["single_u_phot"]
    powerlaw_phot = precomp["powerlaw_phot"]
    umin_grid = jnp.asarray(precomp["umin_grid"])
    qpah_grid = jnp.asarray(precomp["qpah_grid"])
    axes = (qpah_grid, umin_grid)
    edges = tuple(edges_for_grid(ax) for ax in axes)

    @jax.jit
    def dl07_phot(L_absorbed, dust_umin, dust_gamma_dl, dust_qpah):
        point = (dust_qpah, dust_umin)

        # 2D triweight interpolation (C²-continuous gradients)
        single = interp_nd_triweight(single_u_phot, axes, edges, point)
        power = interp_nd_triweight(powerlaw_phot, axes, edges, point)

        # Mix single-U and power-law via gamma
        phot = (1.0 - dust_gamma_dl) * single + dust_gamma_dl * power

        # Scale by absorbed luminosity
        return L_absorbed * phot

    return dl07_phot


# -------------------------------------------------------------------
# SKIRTOR template photometry precomputation
# -------------------------------------------------------------------


def precompute_skirtor_photometry(
    grid_path: str,
    filter_waves: list[jnp.ndarray],
    filter_trans: list[jnp.ndarray],
) -> dict:
    """Pre-integrate SKIRTOR templates through filter curves.

    For each 5D grid point (tau, p, q, oa, cos_inc), compute the
    filter-integrated photometry.

    Parameters
    ----------
    grid_path : str
        Path to ``skirtor_templates.npz``.
    filter_waves : list of array
        Filter wavelength arrays in Angstrom.
    filter_trans : list of array
        Filter transmission arrays.

    Returns
    -------
    dict
        ``grid_phot`` : array (n_tau, n_p, n_q, n_oa, n_inc, n_filters)
        ``axes`` : tuple of 5 grid arrays
    """
    import numpy as np

    data = np.load(grid_path)
    grid = data["grid"]  # (n_tau, n_p, n_q, n_oa, n_inc, n_wave)
    wave_grid = data["wavelength"]  # Angstrom
    axes = (
        np.asarray(data["tau"]),
        np.asarray(data["p"]),
        np.asarray(data["q"]),
        np.asarray(data["oa"]),
        np.asarray(data["cos_inc"]),
    )

    # Use preintegrate_grid to handle 5D filter integration with normalization.
    # SKIRTOR templates are already in L_nu units (or treated as such).
    # energy_normalize=True normalizes each template to unit bolometric
    # luminosity so it can be scaled by agn_log_lbol at runtime.
    preint = preintegrate_grid(
        templates=grid,
        wave_rest=np.asarray(wave_grid),
        filter_waves=[np.asarray(fw) for fw in filter_waves],
        filter_trans=[np.asarray(ft) for ft in filter_trans],
        redshift=0.0,
        dl_cm=1.0,
        axes=axes,
        energy_normalize=True,
    )

    # Convert axes back to JAX arrays for backward compatibility
    axes_jax = (
        jnp.array(data["tau"]),
        jnp.array(data["p"]),
        jnp.array(data["q"]),
        jnp.array(data["oa"]),
        jnp.array(data["cos_inc"]),
    )

    return {
        "grid_phot": preint.phot,
        "axes": axes_jax,
    }


def build_skirtor_photometry_lookup(precomp: dict):
    """Build a JIT-compiled SKIRTOR photometry function from precomputed tables.

    Uses triweight interpolation for C²-continuous gradients (replaces
    the previous piecewise-linear ``_multilinear_interp_5d``).

    Parameters
    ----------
    precomp : dict
        Output of ``precompute_skirtor_photometry()``.

    Returns
    -------
    callable
        ``(agn_log_lbol, agn_tau_skirtor, agn_p_skirtor, agn_q_skirtor,
           agn_oa_skirtor, agn_cos_inc, agn_torus_frac) -> array (n_filters,)``
    """
    from tengri.utils.interpolation import edges_for_grid

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
        l_bol_lsun = 10.0**agn_log_lbol

        point = (
            agn_tau_skirtor,
            agn_p_skirtor,
            agn_q_skirtor,
            agn_oa_skirtor,
            agn_cos_inc,
        )
        # 5D triweight interpolation (C²-continuous gradients)
        phot_normed = interp_nd_triweight(grid_phot, axes, edges, point)

        return l_bol_lsun * agn_torus_frac * phot_normed

    return skirtor_phot
