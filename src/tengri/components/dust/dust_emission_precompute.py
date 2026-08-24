# SPDX-License-Identifier: BSD-3-Clause
"""Precompute adapter for template-based dust IR emission models.

Implements :class:`~tengri.forward.precompute.protocol.PrecomputeModule` for
every template-based dust-emission model: DL07, Dale2014, DL14, Astrodust,
BOSA, THEMIS.  Analytic models (``modified_blackbody``, ``casey2012``) have no
precompute adapter and are evaluated at full wavelength resolution at runtime.

The module exposes:

- ``AXIS_PARAMS[model_name]``: ordered parameter names corresponding to grid
  axes, per dust-emission model.
- ``precompute(filter_waves, filter_trans, redshift, parameters, *, model_name)``:
  builds the preintegrated grid, auto-collapsing axes whose parameter is
  :class:`~tengri.parameters.priors.Fixed`.
- ``build_lookup(preint, *, model_name)``: JIT-compiled ``(scale, *free) →
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
# matters: axis i of the preintegrated grid maps to AXIS_PARAMS[model][i].
AXIS_PARAMS: dict[str, tuple[str, ...]] = {
    "draine_li2007": ("dust_qpah", "dust_umin"),
    "dl07": ("dust_qpah", "dust_umin"),  # alias
    "dale2014": ("dust_alpha",),
    "draine_li2014": ("dust_qpah", "dust_umin", "dust_alpha_dl14"),
    "astrodust": ("dust_qpah", "dust_umin"),
    "themis": ("dust_qhac", "dust_umin"),
    "bosa": ("dust_log_ssfr",),  # log_ltir is derived from L_absorbed at runtime
    "draine2021_pah": ("dust_lgU",),
}

# Per-model flag: pass ``energy_normalize=True`` to ``preintegrate_grid`` only
# for templates that are NOT pre-normalized by ∫L_ν dν=1 at load time. The
# four models marked ``False`` already enforce ∫L_ν dν=1 in their loaders
# (see ``components/dust/emission_templates.py``: load_dale2014_templates ~L612,
# load_astrodust_templates ~L847, load_bosa_templates ~L1202,
# load_themis_templates ~L1442): re-normalizing in precompute is an
# unnecessary round-trip. DL14 has no load-time normalization and relies on
# the precompute-time divide.
_GENERIC_ENERGY_NORMALIZE: dict[str, bool] = {
    # ``load_dale2014_templates`` returns raw L_λ: the runtime path
    # ``create_dale2014_from_grid`` normalizes per-template at factory
    # time. The hybrid path normalizes in ``preintegrate_grid`` instead.
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
    templates: dict
        DL07 template arrays with keys: ``single_u``, ``powerlaw``,
        ``wavelength``, ``umin_grid``, ``qpah_grid`` [erg/s/Hz or L_sun/Hz].
    filter_waves: list of array_like, shape (n_filt,)
        Filter wavelength arrays [Å].
    filter_trans: list of array_like, shape (n_filt,)
        Filter transmission curves [dimensionless].
    redshift: float
        Source redshift [dimensionless]. Default: 0.0.

    Returns
    -------
    dict with keys:
        ``"single_u_phot"``: ndarray, shape (n_qpah, n_umin, n_filters),
            photometry of single-U component [erg/s/Hz or L_sun/Hz].
        ``"powerlaw_phot"``: ndarray, shape (n_qpah, n_umin, n_filters),
            photometry of power-law component [erg/s/Hz or L_sun/Hz].
        ``"umin_grid"``: ndarray, shape (n_umin,), minimum radiation field
            intensity grid [dimensionless].
        ``"qpah_grid"``: ndarray, shape (n_qpah,), PAH fraction grid
            [dimensionless].

    Notes
    -----
    **JIT-compatible**: no, precomputation happens at factory time.

    **Gradient-safe**: no, precomputation is a offline preparation step.
    """
    single_u = templates["single_u"]  # (n_qpah, n_umin, n_wave)
    powerlaw = templates["powerlaw"]  # (n_qpah, n_umin, n_wave)
    tmpl_wave = templates["wavelength"]  # (n_wave,) Angstrom
    umin_grid = templates["umin_grid"]
    qpah_grid = templates["qpah_grid"]

    # Convert templates from L_lambda to L_nu (rest frame). The universal
    # ``preintegrate_grid(energy_normalize=True)`` branch then divides each
    # template by ∫ L_ν dν, mirroring the exact-path renormalization in
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


def build_dl07_photometry_lookup(precomp: dict, grid_arrays: tuple | None = None):
    """Build a JIT-compiled DL07 photometry function from precomputed tables.

    Parameters
    ----------
    precomp: dict
        Output of ``precompute_dl07_photometry()``.
    grid_arrays: tuple or None
        Optional tuple of (single_u_phot, powerlaw_phot, umin_grid, qpah_grid)
        passed as JIT-traced inputs. When None, grids are closure-captured
        (backwards compatible).

    Returns
    -------
    callable
        Signature: ``(L_absorbed, dust_umin, dust_gamma_dl, dust_qpah,
        grid_arrays_traced=None) -> ndarray, shape (n_filters,)``.
        Returns filter-integrated L_ν [erg/s/Hz or L_sun/Hz] at each filter.

    Notes
    -----
    **JIT-compatible**: the returned callable is JIT-compiled via ``@jax.jit``.

    **Gradient-safe**: yes. Use for likelihood evaluation and inference.
    """
    single_u_phot_closure = jnp.asarray(precomp["single_u_phot"])
    powerlaw_phot_closure = jnp.asarray(precomp["powerlaw_phot"])
    umin_grid_closure = jnp.asarray(precomp["umin_grid"])
    qpah_grid_closure = jnp.asarray(precomp["qpah_grid"])
    axes_closure = (qpah_grid_closure, umin_grid_closure)
    edges_closure = tuple(edges_for_grid(ax) for ax in axes_closure)

    @jax.jit
    def dl07_phot(
        L_absorbed,
        dust_umin,
        dust_gamma_dl,
        dust_qpah,
        grid_arrays_traced=None,
    ):
        """Compute DL07 dust emission photometry via triweight interpolation on precomputed grid.

        Parameters
        ----------
        L_absorbed: float
            Absorbed luminosity (scaling factor) [Lsun].
        dust_umin: float
            Minimum radiation field intensity [dimensionless].
        dust_gamma_dl: float
            Mixing fraction for power-law component (gamma parameter)
            [dimensionless, in [0, 1]].
        dust_qpah: float
            PAH mass fraction [dimensionless].
        grid_arrays_traced: tuple or None
            Optional JIT-traced (single_u_phot, powerlaw_phot, umin_grid,
            qpah_grid). When provided, these arrays are used instead of
            closure-captured versions.

        Returns
        -------
        ndarray, shape (n_filters,)
            Filter-integrated dust emission photometry [erg/s/Hz or L_sun/Hz].

        Notes
        -----
        **JIT-compatible**: yes, returned from jax.jit decorator.

        **Gradient-safe**: yes, triweight interpolation is C²-continuous.

        Performs 2D triweight interpolation in (qpah, umin) space on the
        precomputed grid, then mixes single-U and power-law components
        via the gamma parameter, and finally scales by L_absorbed.
        """
        # Use traced arrays if provided, else fall back to closure
        if grid_arrays_traced is not None:
            single_u_phot, powerlaw_phot, umin_grid, qpah_grid = grid_arrays_traced
            axes = (qpah_grid, umin_grid)
            edges = tuple(edges_for_grid(ax) for ax in axes)
        else:
            single_u_phot = single_u_phot_closure
            powerlaw_phot = powerlaw_phot_closure
            umin_grid = umin_grid_closure
            qpah_grid = qpah_grid_closure
            axes = axes_closure
            edges = edges_closure

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
# pre-normalized to ``∫L_ν dν = 1`` per template at load time. The runtime
# exact path mixes the two via ``j_ν = (1-γ)·single_u + γ·powerlaw`` then
# scales by ``L_absorbed`` (CMB contrast applied at the wavelength level is
# omitted from the hybrid path: same approximation as DL07).


def _node_bolometric(template_lnu: np.ndarray, tmpl_wave_aa: np.ndarray) -> np.ndarray:
    """Per-node frequency integral ``∫ L_ν dν`` for an L_ν template grid.

    ``template_lnu`` has shape ``(..., n_wave)``; returns the bolometric over the
    leading axes. Integrates over increasing frequency (the ``-`` flips the sign
    introduced by the ascending-wavelength / descending-frequency ordering),
    matching the exact-path energy balance in ``emission_templates.py``.
    """
    nu = _C_CGS / (np.asarray(tmpl_wave_aa) * _AA_TO_CM)  # Hz, descending
    return -np.trapezoid(np.asarray(template_lnu), nu, axis=-1)


def _precompute_dl07_like_photometry(
    templates: dict,
    filter_waves: list[jnp.ndarray],
    filter_trans: list[jnp.ndarray],
    *,
    q_key: str,
    wave_key: str,
    redshift: float = 0.0,
    powerlaw_weight: np.ndarray | None = None,
) -> dict:
    """Shared precompute for DL07-shape models (Astrodust, THEMIS).

    The exact runtime path forms ``mix = (1-γ)·single_u + γ·W·powerlaw`` and then
    renormalizes it to ``L_absorbed`` by its own frequency integral. By linearity
    the filter photometry is therefore
    ``L_abs·[(1-γ)·single_phot + γ·power_phot] / [(1-γ)·single_bol + γ·power_bol]``,
    so we store *both* the (un-normalized) filter photometry and the per-node
    bolometric integral of each component, and bake the PDR luminosity weight
    ``W(U_min)`` into the power-law template here. THEMIS carries its real
    ∫powerlaw/∫single_u ratio in the template itself (``W = 1``); Astrodust /
    DL14 use the analytic DL07 Eq. 33 ``R`` (passed via ``powerlaw_weight``).

    ``energy_normalize`` is intentionally **off**: normalizing each template to
    unit integral would discard the relative single_u↔powerlaw luminosity that
    the energy balance depends on (the #571/#572/#574 PDR-weight family).

    Parameters
    ----------
    templates: dict
        Loader output with ``single_u``, ``powerlaw``, ``umin_grid``,
        the q-axis grid (``qpah_grid`` or ``qhac_grid``), and the
        wavelength array (``wavelength`` or ``wavelength_aa``).
    q_key: str
        Key for the second grid axis (``"qpah_grid"`` or ``"qhac_grid"``).
    wave_key: str
        Key for the wavelength array (``"wavelength"`` or ``"wavelength_aa"``).
    powerlaw_weight: ndarray or None
        Per-``U_min`` PDR luminosity weight ``W(U_min)`` (shape ``(n_umin,)``)
        baked into the power-law template. ``None`` ⇒ ``W = 1`` (THEMIS).
    """
    single_u = np.asarray(templates["single_u"])
    powerlaw = np.asarray(templates["powerlaw"])
    tmpl_wave = np.asarray(templates[wave_key])
    umin_grid = np.asarray(templates["umin_grid"])
    q_grid = np.asarray(templates[q_key])
    # Mirror the exact THEMIS closure (create_themis_from_grid): the shipped
    # THEMIS grid stores qhac in FSPS scaling (CIGALE x 100/2.2). Relabel an
    # FSPS-scaled qhac axis to the user-facing CIGALE convention so this hybrid
    # path interpolates in the same units as the exact path: otherwise the two
    # disagree and the qhac default clips to the grid minimum here. Guarded on
    # the axis max so CIGALE-unit and Astrodust qpah grids are untouched.
    if q_key == "qhac_grid" and float(np.max(q_grid)) > 0.5:
        q_grid = q_grid * (2.2 / 100.0)

    if powerlaw_weight is not None:
        # Bake W(U_min) per node: powerlaw[q, umin, :] *= W[umin].
        powerlaw = powerlaw * np.asarray(powerlaw_weight)[None, :, None]

    single_bol = _node_bolometric(single_u, tmpl_wave)  # (n_q, n_umin)
    power_bol = _node_bolometric(powerlaw, tmpl_wave)  # (n_q, n_umin)

    single_u_preint = preintegrate_grid(
        templates=single_u,
        wave_rest=tmpl_wave,
        filter_waves=[np.asarray(fw) for fw in filter_waves],
        filter_trans=[np.asarray(ft) for ft in filter_trans],
        redshift=redshift,
        dl_cm=1.0,
        axes=(q_grid, umin_grid),
        energy_normalize=False,
    )
    powerlaw_preint = preintegrate_grid(
        templates=powerlaw,
        wave_rest=tmpl_wave,
        filter_waves=[np.asarray(fw) for fw in filter_waves],
        filter_trans=[np.asarray(ft) for ft in filter_trans],
        redshift=redshift,
        dl_cm=1.0,
        axes=(q_grid, umin_grid),
        energy_normalize=False,
    )

    return {
        "single_u_phot": single_u_preint.phot,
        "powerlaw_phot": powerlaw_preint.phot,
        "single_u_bol": jnp.asarray(single_bol),
        "powerlaw_bol": jnp.asarray(power_bol),
        "umin_grid": umin_grid,
        "q_grid": q_grid,
    }


def _build_dl07_like_lookup(precomp: dict, grid_arrays: tuple | None = None):
    """Shared JIT lookup for DL07-shape models. Signature matches DL07:
    ``(L_absorbed, dust_umin, dust_gamma_dl, dust_q, grid_arrays_traced=None)``.

    Uses C²-continuous triweight interpolation via the shared
    :func:`interp_nd_triweight` helper. Smooth gradients are required for
    NIFTy VI / HMC / Hessian-based inference; the resulting ~3-5%
    hybrid-vs-exact bias is below typical dust-template systematic
    uncertainty (~10-30%).

    Parameters
    ----------
    precomp: dict
        Precomputed photometry data from precompute function.
    grid_arrays: tuple or None
        Optional tuple of (single_u_phot, powerlaw_phot, q_grid, umin_grid)
        to be passed as JIT-traced inputs. When None, grids are closure-captured.
    """
    single_u_phot_closure = jnp.asarray(precomp["single_u_phot"])
    powerlaw_phot_closure = jnp.asarray(precomp["powerlaw_phot"])
    single_u_bol_closure = jnp.asarray(precomp["single_u_bol"])
    powerlaw_bol_closure = jnp.asarray(precomp["powerlaw_bol"])
    umin_grid_closure = jnp.asarray(precomp["umin_grid"])
    q_grid_closure = jnp.asarray(precomp["q_grid"])
    axes_closure = (q_grid_closure, umin_grid_closure)
    edges_closure = tuple(edges_for_grid(ax) for ax in axes_closure)

    @jax.jit
    def phot_fn(L_absorbed, dust_umin, dust_gamma_dl, dust_q, grid_arrays_traced=None):
        # Use traced arrays if provided, else fall back to closure. The
        # per-node bolometric integrals are fixed template constants (not part
        # of the threaded grid tuple), so they are always closure-captured.
        single_u_bol = single_u_bol_closure
        powerlaw_bol = powerlaw_bol_closure
        if grid_arrays_traced is not None:
            single_u_phot, powerlaw_phot, q_grid, umin_grid = grid_arrays_traced
            axes = (q_grid, umin_grid)
            edges = tuple(edges_for_grid(ax) for ax in axes)
        else:
            single_u_phot = single_u_phot_closure
            powerlaw_phot = powerlaw_phot_closure
            q_grid = q_grid_closure
            umin_grid = umin_grid_closure
            axes = axes_closure
            edges = edges_closure

        point = (dust_q, dust_umin)
        single = interp_nd_triweight(single_u_phot, axes, edges, point)
        power = interp_nd_triweight(powerlaw_phot, axes, edges, point)
        s_bol = interp_nd_triweight(single_u_bol, axes, edges, point)
        p_bol = interp_nd_triweight(powerlaw_bol, axes, edges, point)
        g = dust_gamma_dl
        # Energy balance: L_abs · mix_phot / ∫mix dν (the powerlaw already
        # carries its PDR luminosity weight W from precompute time).
        num = (1.0 - g) * single + g * power
        den = (1.0 - g) * s_bol + g * p_bol
        return L_absorbed * num / den

    return phot_fn


def precompute_astrodust_photometry(
    templates: dict,
    filter_waves: list[jnp.ndarray],
    filter_trans: list[jnp.ndarray],
    redshift: float = 0.0,
) -> dict:
    """Pre-integrate Astrodust+PAH templates (Hensley & Draine 2023).

    See :func:`_precompute_dl07_like_photometry`. Free param at runtime is
    ``dust_qpah``. The power-law (PDR) template is unit-normalized at load time,
    so its DL07 Eq. 33 luminosity weight ``R(U_min, U_max, α=2)`` is baked in
    here per ``U_min`` (matching the exact runtime path).
    """
    from tengri.components.dust.emission_templates import _pdr_luminosity_weight

    umin_grid = np.asarray(templates["umin_grid"])
    r_power = np.asarray(_pdr_luminosity_weight(umin_grid, umin_grid[-1], 2.0))
    return _precompute_dl07_like_photometry(
        templates,
        filter_waves,
        filter_trans,
        q_key="qpah_grid",
        wave_key="wavelength_aa",
        redshift=redshift,
        powerlaw_weight=r_power,
    )


def build_astrodust_photometry_lookup(precomp: dict):
    """Build JIT-compiled Astrodust photometry lookup.

    Signature: ``(L_absorbed, dust_umin, dust_gamma_dl, dust_qpah)``.
    """
    return _build_dl07_like_lookup(precomp)


def precompute_draine2021_pah_photometry(
    templates: Any,
    filter_waves: list[jnp.ndarray],
    filter_trans: list[jnp.ndarray],
    redshift: float = 0.0,
    *,
    starlight: str = "mMMP",
    ionization: str = "st",
    size_distribution: str = "std",
    slab: bool = False,
) -> dict:
    r"""Pre-integrate Draine+2021 PAHspec templates through filter curves.

    Slices the 6-D PAHspec grid to a 1-D :math:`\lg U` axis at the
    chosen ``(starlight, ionization, size_distribution, slab)``
    configuration, converts :math:`\nu P_\nu` to :math:`L_\nu`, and
    pre-integrates through filters.  The result has the same
    photometry-grid shape as DL07 / Astrodust precomputes, with a
    single runtime free parameter :math:`\lg U`.

    Parameters
    ----------
    templates: Draine2021PAHTemplates
        Loader output from :func:`load_draine2021_pahspec_templates`.
    filter_waves, filter_trans: list of array_like
        Filter curves (observed-frame Å, dimensionless).
    redshift: float
        Source redshift; defaults to rest-frame photometry.
    starlight, ionization, size_distribution, slab :
        Categorical PAHspec axes to slice.  See
        :class:`tengri.components.dust.Draine2021PAHIRConfig`
        for valid values.

    Returns
    -------
    dict
        Keys: ``single_u_phot`` (shape ``(n_filt, n_lgU)``),
        ``lgU_grid`` (shape ``(n_lgU,)``),
        ``wavelength_aa`` (template wave grid in Å).

    Notes
    -----
    **JIT-compatible**: no, file I/O and template slicing happen at
    factory time.
    """
    from tengri.components.dust.emission_templates import Draine2021PAHTemplates

    if not isinstance(templates, Draine2021PAHTemplates):
        raise TypeError(
            "precompute_draine2021_pah_photometry expects a "
            "Draine2021PAHTemplates instance from "
            "load_draine2021_pahspec_templates"
        )

    if starlight not in templates.starlight_names:
        raise ValueError(
            f"starlight={starlight!r} not in template grid {templates.starlight_names}"
        )
    if ionization not in templates.ion_names:
        raise ValueError(f"ionization={ionization!r} not in {templates.ion_names}")
    if size_distribution not in templates.size_names:
        raise ValueError(f"size_distribution={size_distribution!r} not in {templates.size_names}")

    i_sl = templates.starlight_names.index(starlight)
    i_ion = templates.ion_names.index(ionization)
    i_size = templates.size_names.index(size_distribution)
    slab_arr = np.asarray(templates.slab)
    matches = np.where(slab_arr == bool(slab))[0]
    if matches.size == 0:
        raise ValueError(f"slab={slab} not present (have slab={slab_arr})")
    i_slab = int(matches[0])

    nu_pnu = np.asarray(
        templates.nu_pnu_total[i_sl, i_slab, :, i_ion, i_size, :]
    )  # (n_lgU, n_wave_um)
    wave_um = np.asarray(templates.wavelength_um)
    wave_aa = wave_um * 1.0e4  # to Angstrom
    # nu = c / lam_cm.  L_nu = (nu*P_nu) / nu = (nu*P_nu) * lam_cm / c.
    lam_cm = wave_um * 1.0e-4
    L_nu_template = nu_pnu * lam_cm[None, :] / _C_CGS  # (n_lgU, n_wave) erg/s/Hz/H

    lgU_grid = np.asarray(templates.lgU)

    preint = preintegrate_grid(
        templates=L_nu_template,
        wave_rest=wave_aa,
        filter_waves=[np.asarray(fw) for fw in filter_waves],
        filter_trans=[np.asarray(ft) for ft in filter_trans],
        redshift=redshift,
        dl_cm=1.0,
        axes=(lgU_grid,),
        energy_normalize=True,
    )

    return {
        "single_u_phot": preint.phot,  # (n_filt, n_lgU)
        "lgU_grid": lgU_grid,
        "wavelength_aa": wave_aa,
        "starlight": starlight,
        "ionization": ionization,
        "size_distribution": size_distribution,
        "slab": bool(slab),
    }


def build_draine2021_pah_photometry_lookup(precomp: dict):
    r"""Build JIT-compiled photometry lookup for Draine+2021 PAHspec.

    The returned callable has signature
    ``(L_absorbed, dust_lgU) -> photometry``: linearly interpolates the
    pre-integrated photometry grid in :math:`\lg U` (clipped to the
    template support [0, 7]) and rescales by ``L_absorbed`` so the
    integrated bolometric output equals the absorbed luminosity passed
    in.

    Parameters
    ----------
    precomp: dict
        Output of :func:`precompute_draine2021_pah_photometry`.

    Returns
    -------
    callable
        JIT-compiled function ``(L_absorbed, dust_lgU) -> ndarray``.
    """
    phot = jnp.asarray(precomp["single_u_phot"])  # (n_lgU, n_filt)
    lgU_grid = jnp.asarray(precomp["lgU_grid"])

    @jax.jit
    def _lookup(L_absorbed, dust_lgU):
        lgU = jnp.clip(dust_lgU, lgU_grid[0], lgU_grid[-1])
        # Per-filter linear interpolation along the lgU axis.
        # phot has shape (n_lgU, n_filt); vmap over the filter axis (=1).
        per_filt = jax.vmap(
            lambda col: jnp.interp(lgU, lgU_grid, col),
            in_axes=1,
        )(phot)
        return L_absorbed * per_filt

    return _lookup


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
    exact path renormalizes the mixed template by ``∫L_ν dν`` per call;
    here we pre-normalize each grid point at precompute time.

    Free params at runtime: ``dust_umin``, ``dust_gamma_dl``, ``dust_qpah``,
    ``dust_alpha_dl14``.
    """
    single_u = np.asarray(templates["single_u"])  # (n_qpah, n_umin, n_wave)
    powerlaw = np.asarray(templates["powerlaw"])  # (n_qpah, n_umin, n_alpha, n_wave)
    tmpl_wave = np.asarray(templates["wavelength"])
    umin_grid = np.asarray(templates["umin_grid"])
    qpah_grid = np.asarray(templates["qpah_grid"])
    alpha_grid = np.asarray(templates["alpha_grid"])

    # Keep the *raw* single↔power templates (``energy_normalize=False``) and
    # store per-node bolometric integrals so the lookup can energy-balance
    # ``L_abs · mix_phot / ∫mix dν`` exactly as the runtime path does (#572).
    # The DL14 Eq. 33 PDR luminosity weight ``R(U_min, U_max, α)`` is applied
    # *analytically at runtime* (not baked here): baking it would make the
    # power-law vary steeply with α and the lookup's triweight interpolation
    # would smooth it, whereas the exact path applies R at the query point.
    single_bol = _node_bolometric(single_u, tmpl_wave)  # (n_qpah, n_umin)
    power_bol = _node_bolometric(powerlaw, tmpl_wave)  # (n_qpah, n_umin, n_alpha)

    single_u_preint = preintegrate_grid(
        templates=single_u,
        wave_rest=tmpl_wave,
        filter_waves=[np.asarray(fw) for fw in filter_waves],
        filter_trans=[np.asarray(ft) for ft in filter_trans],
        redshift=redshift,
        dl_cm=1.0,
        axes=(qpah_grid, umin_grid),
        energy_normalize=False,
    )
    powerlaw_preint = preintegrate_grid(
        templates=powerlaw,
        wave_rest=tmpl_wave,
        filter_waves=[np.asarray(fw) for fw in filter_waves],
        filter_trans=[np.asarray(ft) for ft in filter_trans],
        redshift=redshift,
        dl_cm=1.0,
        axes=(qpah_grid, umin_grid, alpha_grid),
        energy_normalize=False,
    )

    return {
        "single_u_phot": single_u_preint.phot,
        "powerlaw_phot": powerlaw_preint.phot,
        "single_u_bol": jnp.asarray(single_bol),
        "powerlaw_bol": jnp.asarray(power_bol),
        "umin_grid": umin_grid,
        "qpah_grid": qpah_grid,
        "alpha_grid": alpha_grid,
    }


def build_dl14_photometry_lookup(precomp: dict, grid_arrays: tuple | None = None):
    """Build JIT-compiled DL14 photometry lookup.

    Signature: ``(L_absorbed, dust_umin, dust_gamma_dl, dust_qpah,
    dust_alpha_dl14, grid_arrays_traced=None)``.

    C²-continuous triweight interpolation in (qpah, umin) for single_u
    and (qpah, umin, alpha) for the powerlaw component, via the shared
    :func:`interp_nd_triweight` helper.

    Parameters
    ----------
    precomp: dict
        Precomputed photometry data.
    grid_arrays: tuple or None
        Optional tuple of (single_u_phot, powerlaw_phot, qpah_grid, umin_grid,
        alpha_grid) passed as JIT-traced inputs. When None, grids are
        closure-captured.
    """
    from tengri.components.dust.emission_templates import (
        _DL14_UMAX_POWERLAW,
        _pdr_luminosity_weight,
    )

    single_u_phot_closure = jnp.asarray(precomp["single_u_phot"])
    powerlaw_phot_closure = jnp.asarray(precomp["powerlaw_phot"])
    single_u_bol_closure = jnp.asarray(precomp["single_u_bol"])
    powerlaw_bol_closure = jnp.asarray(precomp["powerlaw_bol"])
    umin_grid_closure = jnp.asarray(precomp["umin_grid"])
    qpah_grid_closure = jnp.asarray(precomp["qpah_grid"])
    alpha_grid_closure = jnp.asarray(precomp["alpha_grid"])

    @jax.jit
    def phot_fn(
        L_absorbed,
        dust_umin,
        dust_gamma_dl,
        dust_qpah,
        dust_alpha_dl14,
        grid_arrays_traced=None,
    ):
        # Use traced arrays if provided, else fall back to closure. Only the
        # phot grids and the parameter axes are needed: the linear interp below
        # works directly off the grids, so no triweight bin-edges are built.
        if grid_arrays_traced is not None:
            single_u_phot, powerlaw_phot, qpah_grid, umin_grid, alpha_grid = grid_arrays_traced
        else:
            single_u_phot = single_u_phot_closure
            powerlaw_phot = powerlaw_phot_closure
            qpah_grid = qpah_grid_closure
            umin_grid = umin_grid_closure
            alpha_grid = alpha_grid_closure

        # Linear (bi/trilinear) interpolation over the parameter grid: the same
        # scheme the exact runtime path uses, so the precomputed filter
        # photometry matches it exactly (linear interpolation commutes with the
        # filter integral). interp_nd_triweight (the smooth-gradient kernel used
        # by the 2D models) is not node-exact and over the steep DL14 α axis
        # leaves a multi-percent bias.
        def _idx_frac(grid, x):
            x = jnp.clip(x, grid[0], grid[-1])
            n = grid.shape[0]
            i = jnp.clip(jnp.searchsorted(grid, x) - 1, 0, n - 2)
            f = (x - grid[i]) / (grid[i + 1] - grid[i])
            return i, f

        iq, fq = _idx_frac(qpah_grid, dust_qpah)
        iu, fu = _idx_frac(umin_grid, dust_umin)
        ia, fa = _idx_frac(alpha_grid, dust_alpha_dl14)

        def _bilinear(grid_data):  # leading axes (q, umin)
            return (
                (1.0 - fq) * (1.0 - fu) * grid_data[iq, iu]
                + (1.0 - fq) * fu * grid_data[iq, iu + 1]
                + fq * (1.0 - fu) * grid_data[iq + 1, iu]
                + fq * fu * grid_data[iq + 1, iu + 1]
            )

        def _trilinear(grid_data):  # leading axes (q, umin, alpha)
            def _bilin_at(ia_):
                return (
                    (1.0 - fq) * (1.0 - fu) * grid_data[iq, iu, ia_]
                    + (1.0 - fq) * fu * grid_data[iq, iu + 1, ia_]
                    + fq * (1.0 - fu) * grid_data[iq + 1, iu, ia_]
                    + fq * fu * grid_data[iq + 1, iu + 1, ia_]
                )

            return (1.0 - fa) * _bilin_at(ia) + fa * _bilin_at(ia + 1)

        single = _bilinear(single_u_phot)
        power = _trilinear(powerlaw_phot)
        s_bol = _bilinear(single_u_bol_closure)
        p_bol = _trilinear(powerlaw_bol_closure)
        # DL14 Eq. 33 PDR luminosity weight at the (analytic-exact) query point.
        r_power = _pdr_luminosity_weight(dust_umin, _DL14_UMAX_POWERLAW, dust_alpha_dl14)
        g = dust_gamma_dl
        # Energy balance: L_abs · mix_phot / ∫mix dν.
        num = (1.0 - g) * single + g * r_power * power
        den = (1.0 - g) * s_bol + g * r_power * p_bol
        return L_absorbed * num / den

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
    the free parameter, and the resulting normalized template is multiplied
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


def build_bosa_photometry_lookup(precomp: dict, grid_arrays: tuple | None = None):
    """Build JIT-compiled BOSA photometry lookup.

    Signature: ``(L_absorbed, dust_log_ssfr, grid_arrays_traced=None)``.
    ``log_ltir`` is derived internally as ``log10(L_absorbed)``.

    Parameters
    ----------
    precomp: dict
        Precomputed photometry data.
    grid_arrays: tuple or None
        Optional tuple of (phot, log_ltir_grid, log_ssfr_grid) passed as
        JIT-traced inputs. When None, grids are closure-captured.
    """
    phot_closure = jnp.asarray(precomp["phot"])
    log_ltir_grid_closure = jnp.asarray(precomp["log_ltir_grid"])
    log_ssfr_grid_closure = jnp.asarray(precomp["log_ssfr_grid"])
    axes_closure = (log_ltir_grid_closure, log_ssfr_grid_closure)
    edges_closure = tuple(edges_for_grid(ax) for ax in axes_closure)

    @jax.jit
    def phot_fn(L_absorbed, dust_log_ssfr, grid_arrays_traced=None):
        # Use traced arrays if provided, else fall back to closure
        if grid_arrays_traced is not None:
            phot, log_ltir_grid, log_ssfr_grid = grid_arrays_traced
            axes = (log_ltir_grid, log_ssfr_grid)
            edges = tuple(edges_for_grid(ax) for ax in axes)
        else:
            phot = phot_closure
            log_ltir_grid = log_ltir_grid_closure
            log_ssfr_grid = log_ssfr_grid_closure
            axes = axes_closure
            edges = edges_closure

        log_ltir = jnp.log10(jnp.maximum(L_absorbed, 1.0e-30))
        shape_phot = interp_nd_triweight(phot, axes, edges, (log_ltir, dust_log_ssfr))
        return L_absorbed * shape_phot

    return phot_fn


# ── Dale 2014: single-grid (alpha): linear interpolation matching exact ─


def precompute_dale2014_photometry(
    templates: dict,
    filter_waves: list[jnp.ndarray],
    filter_trans: list[jnp.ndarray],
    redshift: float = 0.0,
) -> dict:
    """Pre-integrate Dale+2014 templates through filter curves.

    Dale templates are L_ν, pre-normalized at load (∫L_ν dν=1) per the h5
    ``spectra_unit`` attribute. Free param at runtime is ``dust_alpha_dale``.
    """
    spectra = np.asarray(templates["spectra"])  # (n_alpha, n_wave)
    tmpl_wave = np.asarray(templates["wavelength_aa"])
    alpha_grid = np.asarray(templates["alpha_grid"])

    preint = preintegrate_grid(
        templates=spectra,
        wave_rest=tmpl_wave,
        filter_waves=[np.asarray(fw) for fw in filter_waves],
        filter_trans=[np.asarray(ft) for ft in filter_trans],
        redshift=redshift,
        dl_cm=1.0,
        axes=(alpha_grid,),
        energy_normalize=True,
    )
    return {"phot": preint.phot, "alpha_grid": alpha_grid}


def build_dale2014_photometry_lookup(precomp: dict, grid_arrays: tuple | None = None):
    """Build JIT-compiled Dale 2014 photometry lookup.

    Signature: ``(L_absorbed, dust_alpha_dale, grid_arrays_traced=None)``.
    Uses C²-continuous triweight interpolation in alpha via the shared
    :func:`interp_nd_triweight` helper.

    Parameters
    ----------
    precomp: dict
        Precomputed photometry data.
    grid_arrays: tuple or None
        Optional tuple of (phot, alpha_grid) passed as JIT-traced inputs.
        When None, grids are closure-captured.
    """
    phot_closure = jnp.asarray(precomp["phot"])
    alpha_grid_closure = jnp.asarray(precomp["alpha_grid"])
    axes_closure = (alpha_grid_closure,)
    edges_closure = tuple(edges_for_grid(ax) for ax in axes_closure)

    @jax.jit
    def phot_fn(L_absorbed, dust_alpha_dale, grid_arrays_traced=None):
        # Use traced arrays if provided, else fall back to closure
        if grid_arrays_traced is not None:
            phot, alpha_grid = grid_arrays_traced
            axes = (alpha_grid,)
            edges = tuple(edges_for_grid(ax) for ax in axes)
        else:
            phot = phot_closure
            alpha_grid = alpha_grid_closure
            axes = axes_closure
            edges = edges_closure

        shape_phot = interp_nd_triweight(phot, axes, edges, (dust_alpha_dale,))
        return L_absorbed * shape_phot

    return phot_fn


# ── Protocol-shaped entry points (new in restructure) ─────────────


def _auto_collapse(preint: PreintegratedGrid, axis_params, parameters) -> PreintegratedGrid:
    """Collapse grid axes whose corresponding prior is Fixed.

    Shared helper for every dust-IR adapter. Returns the input grid unchanged
    if no axis params are fixed.

    Parameters
    ----------
    preint: PreintegratedGrid
        Preintegrated photometry grid.
    axis_params: tuple[str, ...]
        Ordered parameter names corresponding to grid axes [dimensionless].
    parameters: Parameters or None
        Parameter specification for prior queries.

    Returns
    -------
    PreintegratedGrid
        Grid with axes corresponding to Fixed priors collapsed and removed.

    Notes
    -----
    **JIT-compatible**: no, helper for factory-time preprocessing.

    **Gradient-safe**: no, operates on factory-time data structures.
    """
    if parameters is None or not axis_params:
        return preint

    # Parameters' fixed-prior introspection API varies across the in-flight
    # parameter-routing refactor: prefer the canonical ``is_fixed/fixed_value``
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
    filter_waves: list of array_like, shape (n_filt,)
        Filter wavelength curves (observed frame) [Å].
    filter_trans: list of array_like, shape (n_filt,)
        Filter transmission curves (observed frame) [dimensionless].
    redshift: float
        Source redshift [dimensionless]; filter integrals bake this in.
    parameters: Parameters or None
        Parameter specification. Used only to detect which AXIS_PARAMS are Fixed.
    model_name: str
        One of the keys in :data:`AXIS_PARAMS`.
    templates: dict, optional
        Required for DL07 (``{"single_u"``, ``"powerlaw"``, ``"wavelength"``,
        ``"umin_grid"``, ``"qpah_grid"}``) [erg/s/Hz].
    grid: ndarray, optional
        Template grid array, shape ``(*grid_dims, n_wave)`` [erg/s/Hz or L_sun/Hz].
        Required for generic path (Dale2014/DL14/Astrodust/BOSA/THEMIS).
    axes: tuple[ndarray, ...], optional
        Grid axis coordinate arrays [various dimensionless units].
        Required for generic path.
    wavelength: ndarray, optional
        Template wavelength array [Å]. Required for generic path.
    units: {"lnu", "llam"}
        Template flux units [dimensionless]; ``"llam"`` triggers L_λ → L_ν conversion.

    Returns
    -------
    dict or PreintegratedGrid
        Precomputed photometry data. For DL07, returns a plain dict (not a
        :class:`PreintegratedGrid`); the ``build_lookup`` function accepts both.

    Notes
    -----
    **JIT-compatible**: no, precomputation happens at factory time.

    **Gradient-safe**: no, precomputation is an offline preparation step.
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
    preint: dict or PreintegratedGrid
        Precomputed photometry grid from ``precompute()``.
    model_name: str
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
    if model_name == "dale2014":
        return build_dale2014_photometry_lookup(preint)
    if model_name == "draine2021_pah":
        return build_draine2021_pah_photometry_lookup(preint)
    return build_template_photometry_lookup(preint)


# ── Turnkey loader + preintegrator: used by SEDModel.__init__ ────


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
    is not on disk: callers fall back to full-wavelength evaluation.

    Parameters
    ----------
    model_name: str
        Dust emission model identifier: ``"draine_li2007"``, ``"dale2014"``,
        ``"draine_li2014"``, ``"astrodust"``, ``"bosa"``, ``"themis"``.
        Other names (``"dl07"`` alias, ``"modified_blackbody"``,
        ``"casey2012"``) return ``None``.
    filter_waves: list of array_like, shape (n_filt,)
        Filter wavelength curves (observed frame) [Å].
    filter_trans: list of array_like, shape (n_filt,)
        Filter transmission curves (observed frame) [dimensionless].
    redshift: float
        Source redshift [dimensionless] (fixed at init time).
    parameters: Parameters or None
        Parameter specification. Used for auto-collapse on Fixed parameters (Protocol surface).

    Returns
    -------
    dict or PreintegratedGrid or None
        Precomputed photometry (feed to :func:`build_lookup` with the same
        ``model_name``), or ``None`` if template data is unavailable.

    Notes
    -----
    **JIT-compatible**: no, template loading and precomputation happen at factory time.

    **Gradient-safe**: no, returns precomputed factory-time data structures.
    """
    # Analytic models have no preintegration
    if model_name in (None, "modified_blackbody", "casey2012"):
        return None

    from tengri._data_setup import find_data_str
    from tengri.components.dust.emission import (
        load_astrodust_templates,
        load_bosa_templates,
        load_dl14_templates,
        load_draine_li_templates,
        load_themis_templates,
    )

    # DL07 has a bespoke template structure (single_U + power-law + gamma mixing)
    if model_name in ("draine_li2007", "dl07"):
        path = find_data_str("dl07_templates_v2.h5", "dl07_templates.h5")
        if path is None:
            return None
        templates = load_draine_li_templates(path)
        return precompute_dl07_photometry(templates, filter_waves, filter_trans, redshift=redshift)

    # Bespoke (DL07-shaped) models: Astrodust/THEMIS/DL14/BOSA: each has a
    # dedicated precompute path. They use their own loader (not the generic
    # ``precompute_template_photometry`` route, which assumes a single-grid
    # ``{grid, axes, wavelength}`` schema that none of these loaders return).
    from tengri.components.dust.emission_templates import load_dale2014_templates

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
        "dale2014": (
            ("dale2014_templates.h5", "dale2014_templates_v2.h5"),
            load_dale2014_templates,
            precompute_dale2014_photometry,
        ),
    }
    if model_name in _BESPOKE_LOADERS:
        candidate_files, loader, precompute_fn = _BESPOKE_LOADERS[model_name]
        path = None
        for fname in candidate_files:
            candidate = find_data_str(fname)
            if candidate is not None:
                path = candidate
                break
        if path is None:
            return None
        templates = loader(path)
        return precompute_fn(templates, filter_waves, filter_trans, redshift=redshift)

    if model_name == "draine2021_pah":
        # Categorical config from environment (no-op fallback to standard model).
        # Path discovery: env override > data/ default > None (returns None below).
        import os as _os

        from tengri.components.dust.emission_templates import (
            load_draine2021_pahspec_templates,
        )

        path_str = _os.environ.get("TENGRI_PAHSPEC_PATH")
        if path_str is None:
            for fname in ("pahspec_draine2021.h5",):
                candidate = find_data_str(fname)
                if candidate is not None:
                    path_str = candidate
                    break
        if path_str is None or not _os.path.isfile(path_str):
            return None

        templates_obj = load_draine2021_pahspec_templates(path_str)
        cfg_starlight = _os.environ.get("TENGRI_PAHSPEC_STARLIGHT", "mMMP")
        cfg_ion = _os.environ.get("TENGRI_PAHSPEC_ION", "st")
        cfg_size = _os.environ.get("TENGRI_PAHSPEC_SIZE", "std")
        cfg_slab = _os.environ.get("TENGRI_PAHSPEC_SLAB", "0").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        # Optional: parameters object can override via attributes.
        if parameters is not None:
            cfg_starlight = getattr(parameters, "_pahspec_starlight", cfg_starlight)
            cfg_ion = getattr(parameters, "_pahspec_ionization", cfg_ion)
            cfg_size = getattr(parameters, "_pahspec_size_distribution", cfg_size)
            cfg_slab = getattr(parameters, "_pahspec_slab", cfg_slab)

        return precompute_draine2021_pah_photometry(
            templates_obj,
            filter_waves,
            filter_trans,
            redshift=redshift,
            starlight=cfg_starlight,
            ionization=cfg_ion,
            size_distribution=cfg_size,
            slab=cfg_slab,
        )

    return None
