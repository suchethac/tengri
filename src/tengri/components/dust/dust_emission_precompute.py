"""Precompute adapter for template-based dust IR emission models.

Implements :class:`~tengri.forward.precompute.protocol.PrecomputeModule` for
every template-based dust-emission model: DL07, Dale2014, DL14, Astrodust,
BOSA, THEMIS.  Analytic models (``modified_blackbody``, ``casey2012``) have no
precompute adapter and are evaluated at full wavelength resolution at runtime.

The module exposes:

- ``AXIS_PARAMS[model_name]`` — ordered parameter names corresponding to grid
  axes, per dust-emission model.
- ``precompute(filter_waves, filter_trans, redshift, parameters, *, model_name)``
  — builds the preintegrated grid, auto-collapsing axes whose parameter is
  :class:`~tengri.parameters.priors.Fixed`.
- ``build_lookup(preint, *, model_name)`` — JIT-compiled ``(scale, *free) →
  phot`` runtime callable, with free-parameter count matching the collapsed axes.

DL07 has a gamma-weighted mixing of single-U and power-law components; the
adapter preintegrates both and mixes at runtime.  The remaining 5 template
families share a simpler generic code path.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from tengri.forward.precompute.templates import (
    build_template_photometry_lookup,
    precompute_template_photometry,
)
from tengri.utils.grid_interp import (
    PreintegratedGrid,
    interp_nd_triweight,
    preintegrate_grid,
    slice_fixed_axes,
)
from tengri.utils.interpolation import edges_for_grid
from tengri.utils.physics_constants import AA_TO_CM as _AA_TO_CM, C_CGS as _C_CGS

# Parameter names corresponding to grid axes, per dust-emission model. Order
# matters — axis i of the preintegrated grid maps to AXIS_PARAMS[model][i].
AXIS_PARAMS: dict[str, tuple[str, ...]] = {
    "draine_li2007": ("dust_qpah", "dust_umin"),
    "dl07": ("dust_qpah", "dust_umin"),  # alias
    "dale2014": ("dust_alpha",),
    "draine_li2014": ("dust_qpah", "dust_umin", "dust_alpha_dl14"),
    "astrodust": ("dust_qpah", "dust_umin"),
    "themis": ("dust_qhac", "dust_umin"),
    "bosa": ("dust_log_ssfr",),  # log_ltir is derived from L_absorbed at runtime
}

# Per-model flag: pass ``energy_normalize=True`` to ``preintegrate_grid`` only
# for templates that are NOT pre-normalised by ∫L_ν dν=1 at load time. The
# four models marked ``False`` already enforce ∫L_ν dν=1 in their loaders
# (see ``components/dust/emission_templates.py``: load_dale2014_templates ~L612,
# load_astrodust_templates ~L847, load_bosa_templates ~L1202,
# load_themis_templates ~L1442) — re-normalising in precompute is an
# unnecessary round-trip. DL14 has no load-time normalisation and relies on
# the precompute-time divide.
_GENERIC_ENERGY_NORMALIZE: dict[str, bool] = {
    # ``load_dale2014_templates`` returns raw L_λ — the runtime path
    # ``create_dale2014_from_grid`` normalises per-template at factory
    # time. The hybrid path normalises in ``preintegrate_grid`` instead.
    "dale2014": True,
    "draine_li2014": True,
    "astrodust": False,
    "bosa": False,
    "themis": False,
}


# ── DL07 / DL14 template photometry precomputation
# (Protocol-shaped entry points below wrap the original functions)


def precompute_dl07_photometry(
    templates: dict,
    filter_waves: list[jnp.ndarray],
    filter_trans: list[jnp.ndarray],
    redshift: float = 0.0,
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
        ``wavelength``, ``umin_grid``, ``qpah_grid`` [erg/s/Hz or L_sun/Hz].
    filter_waves : list of array_like, shape (n_filt,)
        Filter wavelength arrays [Å].
    filter_trans : list of array_like, shape (n_filt,)
        Filter transmission curves [dimensionless].
    redshift : float
        Source redshift [dimensionless]. Default: 0.0.

    Returns
    -------
    dict with keys:
        ``"single_u_phot"`` — ndarray, shape (n_qpah, n_umin, n_filters),
            photometry of single-U component [erg/s/Hz or L_sun/Hz].
        ``"powerlaw_phot"`` — ndarray, shape (n_qpah, n_umin, n_filters),
            photometry of power-law component [erg/s/Hz or L_sun/Hz].
        ``"umin_grid"`` — ndarray, shape (n_umin,), minimum radiation field
            intensity grid [dimensionless].
        ``"qpah_grid"`` — ndarray, shape (n_qpah,), PAH fraction grid
            [dimensionless].

    Notes
    -----
    **JIT-compatible**: no — precomputation happens at factory time.

    **Gradient-safe**: no — precomputation is a offline preparation step.
    """
    single_u = templates["single_u"]  # (n_qpah, n_umin, n_wave)
    powerlaw = templates["powerlaw"]  # (n_qpah, n_umin, n_wave)
    tmpl_wave = templates["wavelength"]  # (n_wave,) Angstrom
    umin_grid = templates["umin_grid"]
    qpah_grid = templates["qpah_grid"]

    # Convert templates from L_lambda to L_nu (rest frame). The universal
    # ``preintegrate_grid(energy_normalize=True)`` branch then divides each
    # template by ∫ L_ν dν, mirroring the exact-path renormalisation in
    # :func:`tengri.components.dust.emission_templates.create_dl07_from_grid`.
    wave_cm = np.asarray(tmpl_wave) * _AA_TO_CM
    lnu_single_u = np.asarray(single_u) * (wave_cm**2) / _C_CGS
    lnu_powerlaw = np.asarray(powerlaw) * (wave_cm**2) / _C_CGS

    single_u_preint = preintegrate_grid(
        templates=lnu_single_u,
        wave_rest=np.asarray(tmpl_wave),
        filter_waves=[np.asarray(fw) for fw in filter_waves],
        filter_trans=[np.asarray(ft) for ft in filter_trans],
        redshift=redshift,
        dl_cm=1.0,
        axes=(np.asarray(qpah_grid), np.asarray(umin_grid)),
        energy_normalize=True,
    )

    powerlaw_preint = preintegrate_grid(
        templates=lnu_powerlaw,
        wave_rest=np.asarray(tmpl_wave),
        filter_waves=[np.asarray(fw) for fw in filter_waves],
        filter_trans=[np.asarray(ft) for ft in filter_trans],
        redshift=redshift,
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

    Parameters
    ----------
    precomp : dict
        Output of ``precompute_dl07_photometry()``.

    Returns
    -------
    callable
        Signature: ``(L_absorbed, dust_umin, dust_gamma_dl, dust_qpah) ->``
        ``ndarray, shape (n_filters,)``. Returns filter-integrated L_ν
        [erg/s/Hz or L_sun/Hz] at each filter.

    Notes
    -----
    **JIT-compatible**: the returned callable is JIT-compiled via ``@jax.jit``.

    **Gradient-safe**: yes. Use for likelihood evaluation and inference.
    """
    single_u_phot = precomp["single_u_phot"]
    powerlaw_phot = precomp["powerlaw_phot"]
    umin_grid = jnp.asarray(precomp["umin_grid"])
    qpah_grid = jnp.asarray(precomp["qpah_grid"])
    axes = (qpah_grid, umin_grid)
    edges = tuple(edges_for_grid(ax) for ax in axes)

    @jax.jit
    def dl07_phot(L_absorbed, dust_umin, dust_gamma_dl, dust_qpah):
        """Compute DL07 dust emission photometry via triweight interpolation on precomputed grid.

        Parameters
        ----------
        L_absorbed : float
            Absorbed luminosity (scaling factor) [Lsun].
        dust_umin : float
            Minimum radiation field intensity [dimensionless].
        dust_gamma_dl : float
            Mixing fraction for power-law component (gamma parameter)
            [dimensionless, in [0, 1]].
        dust_qpah : float
            PAH mass fraction [dimensionless].

        Returns
        -------
        ndarray, shape (n_filters,)
            Filter-integrated dust emission photometry [erg/s/Hz or L_sun/Hz].

        Notes
        -----
        **JIT-compatible**: yes — returned from jax.jit decorator.

        **Gradient-safe**: yes — triweight interpolation is C²-continuous.

        Performs 2D triweight interpolation in (qpah, umin) space on the
        precomputed grid, then mixes single-U and power-law components
        via the gamma parameter, and finally scales by L_absorbed.
        """
        point = (dust_qpah, dust_umin)
        # 2D triweight interpolation (C²-continuous gradients)
        single = interp_nd_triweight(single_u_phot, axes, edges, point)
        power = interp_nd_triweight(powerlaw_phot, axes, edges, point)
        # Mix single-U and power-law via gamma
        phot = (1.0 - dust_gamma_dl) * single + dust_gamma_dl * power
        # Scale by absorbed luminosity
        return L_absorbed * phot

    return dl07_phot


# ── Astrodust / THEMIS: DL07-shaped (single_u + powerlaw, qX × umin) ───
#
# The Astrodust+PAH (Hensley & Draine 2023) and THEMIS (Jones+2017) loaders
# return ``single_u`` and ``powerlaw`` arrays of shape (n_q, n_umin, n_wave),
# pre-normalised to ``∫L_ν dν = 1`` per template at load time. The runtime
# exact path mixes the two via ``j_ν = (1-γ)·single_u + γ·powerlaw`` then
# scales by ``L_absorbed`` (CMB contrast applied at the wavelength level is
# omitted from the hybrid path — same approximation as DL07).


def _precompute_dl07_like_photometry(
    templates: dict,
    filter_waves: list[jnp.ndarray],
    filter_trans: list[jnp.ndarray],
    *,
    q_key: str,
    wave_key: str,
    redshift: float = 0.0,
) -> dict:
    """Shared precompute for DL07-shape models (Astrodust, THEMIS).

    Parameters
    ----------
    templates : dict
        Loader output with ``single_u``, ``powerlaw``, ``umin_grid``,
        the q-axis grid (``qpah_grid`` or ``qhac_grid``), and the
        wavelength array (``wavelength`` or ``wavelength_aa``).
    q_key : str
        Key for the second grid axis (``"qpah_grid"`` or ``"qhac_grid"``).
    wave_key : str
        Key for the wavelength array (``"wavelength"`` or ``"wavelength_aa"``).
    """
    single_u = np.asarray(templates["single_u"])
    powerlaw = np.asarray(templates["powerlaw"])
    tmpl_wave = np.asarray(templates[wave_key])
    umin_grid = np.asarray(templates["umin_grid"])
    q_grid = np.asarray(templates[q_key])

    # Templates already L_ν and pre-normalised at load time
    # → ``energy_normalize=True`` is an idempotent guard (divides by ≈1).
    single_u_preint = preintegrate_grid(
        templates=single_u,
        wave_rest=tmpl_wave,
        filter_waves=[np.asarray(fw) for fw in filter_waves],
        filter_trans=[np.asarray(ft) for ft in filter_trans],
        redshift=redshift,
        dl_cm=1.0,
        axes=(q_grid, umin_grid),
        energy_normalize=True,
    )
    powerlaw_preint = preintegrate_grid(
        templates=powerlaw,
        wave_rest=tmpl_wave,
        filter_waves=[np.asarray(fw) for fw in filter_waves],
        filter_trans=[np.asarray(ft) for ft in filter_trans],
        redshift=redshift,
        dl_cm=1.0,
        axes=(q_grid, umin_grid),
        energy_normalize=True,
    )

    return {
        "single_u_phot": single_u_preint.phot,
        "powerlaw_phot": powerlaw_preint.phot,
        "umin_grid": umin_grid,
        "q_grid": q_grid,
    }


def _build_dl07_like_lookup(precomp: dict):
    """Shared JIT lookup for DL07-shape models. Signature matches DL07:
    ``(L_absorbed, dust_umin, dust_gamma_dl, dust_q)``.
    """
    single_u_phot = precomp["single_u_phot"]
    powerlaw_phot = precomp["powerlaw_phot"]
    umin_grid = jnp.asarray(precomp["umin_grid"])
    q_grid = jnp.asarray(precomp["q_grid"])
    axes = (q_grid, umin_grid)
    edges = tuple(edges_for_grid(ax) for ax in axes)

    @jax.jit
    def phot_fn(L_absorbed, dust_umin, dust_gamma_dl, dust_q):
        point = (dust_q, dust_umin)
        single = interp_nd_triweight(single_u_phot, axes, edges, point)
        power = interp_nd_triweight(powerlaw_phot, axes, edges, point)
        return L_absorbed * ((1.0 - dust_gamma_dl) * single + dust_gamma_dl * power)

    return phot_fn


def precompute_astrodust_photometry(
    templates: dict,
    filter_waves: list[jnp.ndarray],
    filter_trans: list[jnp.ndarray],
    redshift: float = 0.0,
) -> dict:
    """Pre-integrate Astrodust+PAH templates (Hensley & Draine 2023).

    See :func:`_precompute_dl07_like_photometry`. Free param at runtime is
    ``dust_qpah``.
    """
    return _precompute_dl07_like_photometry(
        templates,
        filter_waves,
        filter_trans,
        q_key="qpah_grid",
        wave_key="wavelength_aa",
        redshift=redshift,
    )


def build_astrodust_photometry_lookup(precomp: dict):
    """Build JIT-compiled Astrodust photometry lookup.

    Signature: ``(L_absorbed, dust_umin, dust_gamma_dl, dust_qpah)``.
    """
    return _build_dl07_like_lookup(precomp)


def precompute_themis_photometry(
    templates: dict,
    filter_waves: list[jnp.ndarray],
    filter_trans: list[jnp.ndarray],
    redshift: float = 0.0,
) -> dict:
    """Pre-integrate THEMIS templates (Jones+2017). Free param: ``dust_qhac``."""
    return _precompute_dl07_like_photometry(
        templates,
        filter_waves,
        filter_trans,
        q_key="qhac_grid",
        wave_key="wavelength_aa",
        redshift=redshift,
    )


def build_themis_photometry_lookup(precomp: dict):
    """Build JIT-compiled THEMIS photometry lookup.

    Signature: ``(L_absorbed, dust_umin, dust_gamma_dl, dust_qhac)``.
    """
    return _build_dl07_like_lookup(precomp)


# ── DL14: DL07 + extra alpha axis on the powerlaw component ───────────


def precompute_dl14_photometry(
    templates: dict,
    filter_waves: list[jnp.ndarray],
    filter_trans: list[jnp.ndarray],
    redshift: float = 0.0,
) -> dict:
    """Pre-integrate DL14 templates (Draine+2014).

    DL14 stores ``single_u`` of shape (n_qpah, n_umin, n_wave) and
    ``powerlaw`` of shape (n_qpah, n_umin, n_alpha, n_wave). The runtime
    exact path renormalises the mixed template by ``∫L_ν dν`` per call;
    here we pre-normalise each grid point at precompute time.

    Free params at runtime: ``dust_umin``, ``dust_gamma_dl``, ``dust_qpah``,
    ``dust_alpha_dl14``.
    """
    single_u = np.asarray(templates["single_u"])  # (n_qpah, n_umin, n_wave)
    powerlaw = np.asarray(templates["powerlaw"])  # (n_qpah, n_umin, n_alpha, n_wave)
    tmpl_wave = np.asarray(templates["wavelength"])
    umin_grid = np.asarray(templates["umin_grid"])
    qpah_grid = np.asarray(templates["qpah_grid"])
    alpha_grid = np.asarray(templates["alpha_grid"])

    # Both grids stored as L_ν but NOT pre-normalised at load (DL14 loader
    # leaves raw j_ν). The universal ``energy_normalize=True`` divides by
    # ∫L_ν dν per template so runtime ``L_absorbed * lookup`` is calibrated.
    single_u_preint = preintegrate_grid(
        templates=single_u,
        wave_rest=tmpl_wave,
        filter_waves=[np.asarray(fw) for fw in filter_waves],
        filter_trans=[np.asarray(ft) for ft in filter_trans],
        redshift=redshift,
        dl_cm=1.0,
        axes=(qpah_grid, umin_grid),
        energy_normalize=True,
    )
    powerlaw_preint = preintegrate_grid(
        templates=powerlaw,
        wave_rest=tmpl_wave,
        filter_waves=[np.asarray(fw) for fw in filter_waves],
        filter_trans=[np.asarray(ft) for ft in filter_trans],
        redshift=redshift,
        dl_cm=1.0,
        axes=(qpah_grid, umin_grid, alpha_grid),
        energy_normalize=True,
    )

    return {
        "single_u_phot": single_u_preint.phot,
        "powerlaw_phot": powerlaw_preint.phot,
        "umin_grid": umin_grid,
        "qpah_grid": qpah_grid,
        "alpha_grid": alpha_grid,
    }


def build_dl14_photometry_lookup(precomp: dict):
    """Build JIT-compiled DL14 photometry lookup.

    Signature: ``(L_absorbed, dust_umin, dust_gamma_dl, dust_qpah,
    dust_alpha_dl14)``.
    """
    single_u_phot = precomp["single_u_phot"]
    powerlaw_phot = precomp["powerlaw_phot"]
    umin_grid = jnp.asarray(precomp["umin_grid"])
    qpah_grid = jnp.asarray(precomp["qpah_grid"])
    alpha_grid = jnp.asarray(precomp["alpha_grid"])

    single_axes = (qpah_grid, umin_grid)
    single_edges = tuple(edges_for_grid(ax) for ax in single_axes)
    pl_axes = (qpah_grid, umin_grid, alpha_grid)
    pl_edges = tuple(edges_for_grid(ax) for ax in pl_axes)

    @jax.jit
    def phot_fn(L_absorbed, dust_umin, dust_gamma_dl, dust_qpah, dust_alpha_dl14):
        single = interp_nd_triweight(
            single_u_phot, single_axes, single_edges, (dust_qpah, dust_umin)
        )
        power = interp_nd_triweight(
            powerlaw_phot, pl_axes, pl_edges, (dust_qpah, dust_umin, dust_alpha_dl14)
        )
        return L_absorbed * ((1.0 - dust_gamma_dl) * single + dust_gamma_dl * power)

    return phot_fn


# ── BOSA: log_ltir derived from L_absorbed at runtime ────────────────


def precompute_bosa_photometry(
    templates: dict,
    filter_waves: list[jnp.ndarray],
    filter_trans: list[jnp.ndarray],
    redshift: float = 0.0,
) -> dict:
    """Pre-integrate BOSA templates (Boquien & Salim 2021).

    BOSA's grid is indexed by (log L_TIR, log sSFR) where both axes affect
    template *shape*. Runtime: ``log_ltir = log10(L_absorbed)`` selects the
    L_TIR slice (so it's a derived axis, not free), ``dust_log_ssfr`` is
    the free parameter, and the resulting normalised template is multiplied
    by ``L_absorbed`` for absolute scaling.
    """
    spectra = np.asarray(templates["spectra"])  # (n_ltir, n_ssfr, n_wave)
    tmpl_wave = np.asarray(templates["wavelength_aa"])
    log_ltir_grid = np.asarray(templates["log_ltir_grid"])
    log_ssfr_grid = np.asarray(templates["log_ssfr_grid"])

    preint = preintegrate_grid(
        templates=spectra,
        wave_rest=tmpl_wave,
        filter_waves=[np.asarray(fw) for fw in filter_waves],
        filter_trans=[np.asarray(ft) for ft in filter_trans],
        redshift=redshift,
        dl_cm=1.0,
        axes=(log_ltir_grid, log_ssfr_grid),
        energy_normalize=True,
    )
    return {
        "phot": preint.phot,
        "log_ltir_grid": log_ltir_grid,
        "log_ssfr_grid": log_ssfr_grid,
    }


def build_bosa_photometry_lookup(precomp: dict):
    """Build JIT-compiled BOSA photometry lookup.

    Signature: ``(L_absorbed, dust_log_ssfr)``. ``log_ltir`` is derived
    internally as ``log10(L_absorbed)``.
    """
    phot = precomp["phot"]
    log_ltir_grid = jnp.asarray(precomp["log_ltir_grid"])
    log_ssfr_grid = jnp.asarray(precomp["log_ssfr_grid"])
    axes = (log_ltir_grid, log_ssfr_grid)
    edges = tuple(edges_for_grid(ax) for ax in axes)

    @jax.jit
    def phot_fn(L_absorbed, dust_log_ssfr):
        log_ltir = jnp.log10(jnp.maximum(L_absorbed, 1.0e-30))
        point = (log_ltir, dust_log_ssfr)
        shape_phot = interp_nd_triweight(phot, axes, edges, point)
        return L_absorbed * shape_phot

    return phot_fn


# ── Protocol-shaped entry points (new in restructure) ─────────────


def _auto_collapse(preint: PreintegratedGrid, axis_params, parameters) -> PreintegratedGrid:
    """Collapse grid axes whose corresponding prior is Fixed.

    Shared helper for every dust-IR adapter. Returns the input grid unchanged
    if no axis params are fixed.

    Parameters
    ----------
    preint : PreintegratedGrid
        Preintegrated photometry grid.
    axis_params : tuple[str, ...]
        Ordered parameter names corresponding to grid axes [dimensionless].
    parameters : Parameters or None
        Parameter specification for prior queries.

    Returns
    -------
    PreintegratedGrid
        Grid with axes corresponding to Fixed priors collapsed and removed.

    Notes
    -----
    **JIT-compatible**: no — helper for factory-time preprocessing.

    **Gradient-safe**: no — operates on factory-time data structures.
    """
    if parameters is None or not axis_params:
        return preint

    # Parameters' fixed-prior introspection API varies across the in-flight
    # parameter-routing refactor — prefer the canonical ``is_fixed/fixed_value``
    # pair but degrade gracefully if a different shape is in scope.
    if not hasattr(parameters, "is_fixed") or not hasattr(parameters, "fixed_value"):
        return preint

    fixed: dict[int, float] = {}
    for i, pname in enumerate(axis_params):
        if parameters.is_fixed(pname):
            fixed[i] = float(parameters.fixed_value(pname))
    if not fixed:
        return preint
    return slice_fixed_axes(preint, fixed)


def precompute(
    filter_waves: list,
    filter_trans: list,
    redshift: float,
    parameters: Any,
    *,
    model_name: str,
    templates: dict | None = None,
    grid: np.ndarray | None = None,
    axes: tuple[np.ndarray, ...] | None = None,
    wavelength: np.ndarray | None = None,
    units: str = "lnu",
) -> Any:
    """Build preintegrated grid for a template-based dust-emission model.

    Auto-collapses any axis whose corresponding ``AXIS_PARAMS[model_name]``
    entry is :class:`~tengri.parameters.priors.Fixed` in ``parameters``.

    Parameters
    ----------
    filter_waves : list of array_like, shape (n_filt,)
        Filter wavelength curves (observed frame) [Å].
    filter_trans : list of array_like, shape (n_filt,)
        Filter transmission curves (observed frame) [dimensionless].
    redshift : float
        Source redshift [dimensionless]; filter integrals bake this in.
    parameters : Parameters or None
        Parameter specification. Used only to detect which AXIS_PARAMS are Fixed.
    model_name : str
        One of the keys in :data:`AXIS_PARAMS`.
    templates : dict, optional
        Required for DL07 (``{"single_u"``, ``"powerlaw"``, ``"wavelength"``,
        ``"umin_grid"``, ``"qpah_grid"}``) [erg/s/Hz].
    grid : ndarray, optional
        Template grid array, shape ``(*grid_dims, n_wave)`` [erg/s/Hz or L_sun/Hz].
        Required for generic path (Dale2014/DL14/Astrodust/BOSA/THEMIS).
    axes : tuple[ndarray, ...], optional
        Grid axis coordinate arrays [various dimensionless units].
        Required for generic path.
    wavelength : ndarray, optional
        Template wavelength array [Å]. Required for generic path.
    units : {"lnu", "llam"}
        Template flux units [dimensionless]; ``"llam"`` triggers L_λ → L_ν conversion.

    Returns
    -------
    dict or PreintegratedGrid
        Precomputed photometry data. For DL07, returns a plain dict (not a
        :class:`PreintegratedGrid`); the ``build_lookup`` function accepts both.

    Notes
    -----
    **JIT-compatible**: no — precomputation happens at factory time.

    **Gradient-safe**: no — precomputation is an offline preparation step.
    """
    if model_name in ("draine_li2007", "dl07"):
        if templates is None:
            raise ValueError("DL07 precompute requires 'templates' dict")
        return precompute_dl07_photometry(templates, filter_waves, filter_trans, redshift=redshift)

    # Generic template path (Dale2014, DL14, Astrodust, BOSA, THEMIS)
    if grid is None or wavelength is None or axes is None:
        raise ValueError(
            f"{model_name} precompute requires 'grid', 'wavelength', and 'axes' kwargs"
        )
    preint = precompute_template_photometry(
        templates=grid,
        wave_rest=wavelength,
        filter_waves=filter_waves,
        filter_trans=filter_trans,
        axes=axes,
        redshift=redshift,
        dl_cm=1.0,
        energy_normalize=_GENERIC_ENERGY_NORMALIZE.get(model_name, True),
        units=units,
    )
    return _auto_collapse(preint, AXIS_PARAMS.get(model_name, ()), parameters)


def build_lookup(preint, *, model_name: str):
    """Build JIT lookup for a dust-emission precompute result.

    Dispatches on ``model_name`` because DL07's mixing signature differs from
    the generic single-template path.

    Parameters
    ----------
    preint : dict or PreintegratedGrid
        Precomputed photometry grid from ``precompute()``.
    model_name : str
        Dust model identifier (same as used in ``precompute()``).

    Returns
    -------
    callable
        JIT-compiled photometry lookup function. Signature depends on ``model_name``.

    Notes
    -----
    **JIT-compatible**: the returned callable is JIT-compiled.

    **Gradient-safe**: yes. Use for likelihood evaluation and inference.
    """
    if model_name in ("draine_li2007", "dl07"):
        return build_dl07_photometry_lookup(preint)
    if model_name == "astrodust":
        return build_astrodust_photometry_lookup(preint)
    if model_name == "themis":
        return build_themis_photometry_lookup(preint)
    if model_name == "draine_li2014":
        return build_dl14_photometry_lookup(preint)
    if model_name == "bosa":
        return build_bosa_photometry_lookup(preint)
    return build_template_photometry_lookup(preint)


# ── Turnkey loader + preintegrator — used by SEDModel.__init__ ────


def precompute_for_model(
    model_name: str,
    filter_waves: list,
    filter_trans: list,
    redshift: float,
    parameters: Any = None,
) -> Any:
    """Load templates and preintegrate a dust-IR component from its model name.

    Encapsulates template loading + file discovery + preintegration (+
    auto-collapse on Fixed) into a single call so :class:`SEDModel` does not
    need a per-model switch. Returns ``None`` when required template data
    is not on disk — callers fall back to full-wavelength evaluation.

    Parameters
    ----------
    model_name : str
        Dust emission model identifier: ``"draine_li2007"``, ``"dale2014"``,
        ``"draine_li2014"``, ``"astrodust"``, ``"bosa"``, ``"themis"``.
        Other names (``"dl07"`` alias, ``"modified_blackbody"``,
        ``"casey2012"``) return ``None``.
    filter_waves : list of array_like, shape (n_filt,)
        Filter wavelength curves (observed frame) [Å].
    filter_trans : list of array_like, shape (n_filt,)
        Filter transmission curves (observed frame) [dimensionless].
    redshift : float
        Source redshift [dimensionless] (fixed at init time).
    parameters : Parameters or None
        Parameter specification. Used for auto-collapse on Fixed parameters (Protocol surface).

    Returns
    -------
    dict or PreintegratedGrid or None
        Precomputed photometry (feed to :func:`build_lookup` with the same
        ``model_name``), or ``None`` if template data is unavailable.

    Notes
    -----
    **JIT-compatible**: no — template loading and precomputation happen at factory time.

    **Gradient-safe**: no — returns precomputed factory-time data structures.
    """
    # Analytic models have no preintegration
    if model_name in (None, "modified_blackbody", "casey2012"):
        return None

    from tengri.components.dust.emission import (
        _find_data_file,
        load_astrodust_templates,
        load_bosa_templates,
        load_dl14_templates,
        load_draine_li_templates,
        load_themis_templates,
    )

    # DL07 has a bespoke template structure (single_U + power-law + gamma mixing)
    if model_name in ("draine_li2007", "dl07"):
        from tengri.components.dust.emission import _find_dl07_templates

        path = _find_dl07_templates()
        if path is None:
            return None
        templates = load_draine_li_templates(path)
        return precompute_dl07_photometry(templates, filter_waves, filter_trans, redshift=redshift)

    # Bespoke (DL07-shaped) models — Astrodust/THEMIS/DL14/BOSA — each has a
    # dedicated precompute path. They use their own loader (not the generic
    # ``precompute_template_photometry`` route, which assumes a single-grid
    # ``{grid, axes, wavelength}`` schema that none of these loaders return).
    _BESPOKE_LOADERS: dict[str, tuple[tuple[str, ...], Any, Any]] = {
        "draine_li2014": (
            ("dl14_templates_v2.h5", "dl14_templates.h5"),
            load_dl14_templates,
            precompute_dl14_photometry,
        ),
        "astrodust": (
            ("astrodust_templates.h5", "astrodust_templates.npz"),
            load_astrodust_templates,
            precompute_astrodust_photometry,
        ),
        "themis": (
            ("themis_templates.h5", "themis_templates.npz"),
            load_themis_templates,
            precompute_themis_photometry,
        ),
        "bosa": (
            ("bosa_templates.h5", "bosa_templates.npz"),
            load_bosa_templates,
            precompute_bosa_photometry,
        ),
    }
    if model_name in _BESPOKE_LOADERS:
        candidate_files, loader, precompute_fn = _BESPOKE_LOADERS[model_name]
        path = None
        for fname in candidate_files:
            candidate = _find_data_file(fname)
            if candidate is not None:
                path = candidate
                break
        if path is None:
            return None
        templates = loader(path)
        return precompute_fn(templates, filter_waves, filter_trans, redshift=redshift)

    # Dale2014 — single 1D-axis (alpha) grid. Supported via the generic
    # ``precompute_template_photometry`` route (single-grid schema).
    if model_name != "dale2014":
        return None

    candidate_files = (
        "dale2014_templates.h5",
        "dale2014_templates_v2.h5",
        "dale2014_templates.npz",
    )
    path = None
    for fname in candidate_files:
        candidate = _find_data_file(fname)
        if candidate is not None:
            path = candidate
            break
    if path is None:
        return None

    if str(path).endswith(".h5"):
        import h5py as _h5py

        from tengri.components.dust.emission_templates import load_dale2014_templates

        data = load_dale2014_templates(path)
        wavelength = np.asarray(data["wavelength_aa"])
        alpha_grid = np.asarray(data["alpha_grid"])
        grid = np.asarray(data["spectra"])
        # h5 files may store either raw L_λ or pre-normalised L_ν —
        # respect the ``spectra_unit`` attribute when present.
        with _h5py.File(path, "r") as _f:
            _unit = _f.attrs.get("spectra_unit", "")
        units = "lnu" if "L_nu" in str(_unit) else "llam"
    else:
        npz = np.load(path)
        wavelength = np.array(npz["wavelength_aa"])
        alpha_grid = np.array(npz["alpha_grid"])
        grid = np.array(npz["templates_sf"])
        units = "llam"

    # Normalize shape to (n_alpha, n_wave)
    if grid.shape[0] == len(wavelength) and grid.shape[1] == len(alpha_grid):
        grid = grid.T
    axes = (alpha_grid,)

    preint = precompute_template_photometry(
        templates=grid,
        wave_rest=wavelength,
        filter_waves=filter_waves,
        filter_trans=filter_trans,
        axes=axes,
        redshift=0.0,  # both use rest-frame templates
        dl_cm=1.0,
        energy_normalize=_GENERIC_ENERGY_NORMALIZE.get(model_name, True),
        units=units,
    )
    return _auto_collapse(preint, AXIS_PARAMS.get(model_name, ()), parameters)
