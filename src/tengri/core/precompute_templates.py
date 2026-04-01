"""Precompute template-based model photometry through filters.

For template models like DL07 and SKIRTOR, integrating templates through
filter curves at every forward model call is expensive (~100μs).  By
pre-integrating the template grid through each filter at model init time,
we reduce runtime to a cheap multilinear interpolation + scalar scaling.

Each ``precompute_*`` function returns a lookup table of shape
``(grid_dims..., n_filters)`` and a corresponding interpolation function
that takes physical parameters and returns photometry.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

# Physical constants (CGS)
_C_CGS = 2.99792458e10  # cm/s
_AA_TO_CM = 1e-8
_LSUN_ERG = 3.828e33  # erg/s


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
    single_u = templates["single_u"]  # (n_qpah, n_umin, n_wave)
    powerlaw = templates["powerlaw"]  # (n_qpah, n_umin, n_wave)
    tmpl_wave = templates["wavelength"]  # (n_wave,) Angstrom
    umin_grid = templates["umin_grid"]
    qpah_grid = templates["qpah_grid"]

    n_qpah, n_umin, _n_wave = single_u.shape
    n_filters = len(filter_waves)

    def _integrate_one_template(template_llam):
        """Integrate a single L_lambda template through all filters.

        Returns array of shape (n_filters,) — filter-averaged L_nu.
        """
        # Convert L_lambda → L_nu
        wave_cm = tmpl_wave * _AA_TO_CM
        nu = _C_CGS / wave_cm
        lnu = template_llam * (wave_cm**2) / _C_CGS

        # Normalize (integral over nu = 1, then scale by L_absorbed at runtime)
        integral = -jnp.trapezoid(lnu, nu)
        lnu_normed = jnp.where(integral > 0, lnu / integral, 0.0)

        results = []
        for fw, ft in zip(filter_waves, filter_trans):
            # Resample template onto filter wavelength grid
            lnu_on_filt = jnp.interp(fw, tmpl_wave, lnu_normed, left=0.0, right=0.0)
            nu_filt = _C_CGS / (fw * _AA_TO_CM)
            # Filter-averaged flux: integral(L_nu * T * dnu) / integral(T * dnu)
            numerator = -jnp.trapezoid(lnu_on_filt * ft, nu_filt)
            denominator = -jnp.trapezoid(ft, nu_filt)
            results.append(jnp.where(denominator > 0, numerator / denominator, 0.0))
        return jnp.array(results)

    # Precompute for all grid points
    single_u_phot = jnp.zeros((n_qpah, n_umin, n_filters))
    powerlaw_phot = jnp.zeros((n_qpah, n_umin, n_filters))

    for iq in range(n_qpah):
        for iu in range(n_umin):
            single_u_phot = single_u_phot.at[iq, iu].set(_integrate_one_template(single_u[iq, iu]))
            powerlaw_phot = powerlaw_phot.at[iq, iu].set(_integrate_one_template(powerlaw[iq, iu]))

    return {
        "single_u_phot": single_u_phot,
        "powerlaw_phot": powerlaw_phot,
        "umin_grid": umin_grid,
        "qpah_grid": qpah_grid,
    }


def build_dl07_photometry_lookup(precomp: dict):
    """Build a JIT-compiled DL07 photometry function from precomputed tables.

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
    single_u_phot = precomp["single_u_phot"]
    powerlaw_phot = precomp["powerlaw_phot"]
    umin_grid = precomp["umin_grid"]
    qpah_grid = precomp["qpah_grid"]

    @jax.jit
    def dl07_phot(L_absorbed, dust_umin, dust_gamma_dl, dust_qpah):
        # Bilinear interpolation indices
        umin_c = jnp.clip(dust_umin, umin_grid[0], umin_grid[-1])
        qpah_c = jnp.clip(dust_qpah, qpah_grid[0], qpah_grid[-1])

        i_u = jnp.clip(jnp.searchsorted(umin_grid, umin_c) - 1, 0, len(umin_grid) - 2)
        i_q = jnp.clip(jnp.searchsorted(qpah_grid, qpah_c) - 1, 0, len(qpah_grid) - 2)

        fu = (umin_c - umin_grid[i_u]) / (umin_grid[i_u + 1] - umin_grid[i_u])
        fq = (qpah_c - qpah_grid[i_q]) / (qpah_grid[i_q + 1] - qpah_grid[i_q])

        def _bilinear(grid):
            return (
                (1.0 - fq) * (1.0 - fu) * grid[i_q, i_u]
                + (1.0 - fq) * fu * grid[i_q, i_u + 1]
                + fq * (1.0 - fu) * grid[i_q + 1, i_u]
                + fq * fu * grid[i_q + 1, i_u + 1]
            )

        # Mix single-U and power-law via gamma
        phot = (1.0 - dust_gamma_dl) * _bilinear(single_u_phot) + dust_gamma_dl * _bilinear(
            powerlaw_phot
        )

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
        jnp.array(data["tau"]),
        jnp.array(data["p"]),
        jnp.array(data["q"]),
        jnp.array(data["oa"]),
        jnp.array(data["cos_inc"]),
    )

    shape_5d = grid.shape[:5]
    n_filters = len(filter_waves)
    grid_phot = jnp.zeros((*shape_5d, n_filters))

    # For each grid point, integrate template through filters
    for idx in np.ndindex(*shape_5d):
        template = grid[idx]  # (n_wave,)

        # Filter-averaged L_nu (normalized — L_bol scaling at runtime)
        nu = _C_CGS / (wave_grid * _AA_TO_CM)
        integral = jnp.trapezoid(jnp.array(template[jnp.argsort(nu)]), jnp.sort(nu))
        integral_safe = jnp.maximum(jnp.abs(integral), 1e-100)
        template_normed = jnp.array(template) / integral_safe

        phot_vals = []
        for fw, ft in zip(filter_waves, filter_trans):
            t_resampled = jnp.interp(
                fw, jnp.array(wave_grid), template_normed, left=0.0, right=0.0
            )
            nu_filt = _C_CGS / (fw * _AA_TO_CM)
            num = -jnp.trapezoid(t_resampled * ft, nu_filt)
            den = -jnp.trapezoid(ft, nu_filt)
            phot_vals.append(jnp.where(den > 0, num / den, 0.0))

        grid_phot = grid_phot.at[idx].set(jnp.array(phot_vals))

    return {
        "grid_phot": grid_phot,
        "axes": axes,
    }


def build_skirtor_photometry_lookup(precomp: dict):
    """Build a JIT-compiled SKIRTOR photometry function from precomputed tables.

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
    from tengri.models.agn.skirtor import _multilinear_interp_5d

    grid_phot = precomp["grid_phot"]
    axes = precomp["axes"]

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
        # 5D multilinear interpolation in precomputed photometry table
        phot_normed = _multilinear_interp_5d(grid_phot, axes, point)

        return l_bol_lsun * agn_torus_frac * phot_normed

    return skirtor_phot
