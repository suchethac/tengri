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

from tengri.forward.precompute.grid import (
    PreintegratedGrid,
    interp_nd_triweight,
    preintegrate_grid,
    slice_fixed_axes,
)
from tengri.forward.precompute.templates import (
    build_template_photometry_lookup,
    precompute_template_photometry,
)
from tengri.utils.interpolation import edges_for_grid
from tengri.utils.physics_constants import AA_TO_CM as _AA_TO_CM, C_CGS as _C_CGS

# Parameter names corresponding to grid axes, per dust-emission model. Order
# matters — axis i of the preintegrated grid maps to AXIS_PARAMS[model][i].
AXIS_PARAMS: dict[str, tuple[str, ...]] = {
    "draine_li2007": ("dust_qpah", "dust_umin"),
    "dl07": ("dust_qpah", "dust_umin"),  # alias
    "dale2014": ("dust_alpha",),
    "draine_li2014": (),  # grid structure varies; populated at runtime from template data
    "astrodust": (),  # grid structure varies
    "bosa": (),
    "themis": (),
}


# -------------------------------------------------------------------
# DL07 / DL14 template photometry precomputation (original functions kept
# for backward compatibility; the Protocol-shaped entry points below wrap them)
# -------------------------------------------------------------------


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

    # Convert templates from L_lambda to L_nu (rest frame)
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
        ``(L_absorbed, dust_umin, dust_gamma_dl, dust_qpah) -> array (n_filters,)``
        Returns L_nu (Lsun/Hz) at each filter.
    """
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
# Protocol-shaped entry points (new in restructure)
# -------------------------------------------------------------------


def _auto_collapse(preint: PreintegratedGrid, axis_params, parameters) -> PreintegratedGrid:
    """Collapse grid axes whose corresponding prior is Fixed.

    Shared helper for every dust-IR adapter.  Returns the input grid unchanged
    if no axis params are fixed.
    """
    if parameters is None or not axis_params:
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
    filter_waves, filter_trans : list
        Filter curves (observed frame).
    redshift : float
        Source redshift; filter integrals bake this in.
    parameters : Parameters | None
        Parameter spec.  Used only to detect which AXIS_PARAMS are Fixed.
    model_name : str
        One of the keys in :data:`AXIS_PARAMS`.
    templates : dict, optional
        Required for DL07 (``{"single_u", "powerlaw", "wavelength",
        "umin_grid", "qpah_grid"}``).
    grid, axes, wavelength : optional
        Required for the generic path (Dale2014 / DL14 / Astrodust / BOSA /
        THEMIS). ``grid`` has shape ``(*grid_dims, n_wave)``.
    units : {"lnu", "llam"}
        Template flux units; ``"llam"`` triggers L_λ → L_ν conversion.

    Returns
    -------
    dict (for DL07) or PreintegratedGrid (for generic path)
        The precomputed data. For DL07 the dict is kept for backward
        compatibility and is not a :class:`PreintegratedGrid`; the
        ``build_lookup`` function accepts both shapes.
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
        energy_normalize=True,
        units=units,
    )
    return _auto_collapse(preint, AXIS_PARAMS.get(model_name, ()), parameters)


def build_lookup(preint, *, model_name: str):
    """Build JIT lookup for a dust-emission precompute result.

    Dispatches on ``model_name`` because DL07's mixing signature differs from
    the generic single-template path.
    """
    if model_name in ("draine_li2007", "dl07"):
        # preint is a dict from precompute_dl07_photometry
        return build_dl07_photometry_lookup(preint)
    return build_template_photometry_lookup(preint)


# -------------------------------------------------------------------
# Turnkey loader + preintegrator — used by SEDModel.__init__
# -------------------------------------------------------------------


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
    need a per-model switch.  Returns ``None`` when required template data
    is not on disk — callers fall back to full-wavelength evaluation.

    Parameters
    ----------
    model_name : str
        One of ``"draine_li2007"``, ``"dale2014"``, ``"draine_li2014"``,
        ``"astrodust"``, ``"bosa"``, ``"themis"``.  Other names (``"dl07"``
        alias, ``"modified_blackbody"``, ``"casey2012"``) return ``None``.
    filter_waves, filter_trans : list
        Filter curves (observed frame).
    redshift : float
        Source redshift (fixed at init time).
    parameters : Parameters | None
        Used for auto-collapse on Fixed parameters (Protocol surface).

    Returns
    -------
    dict (DL07) or PreintegratedGrid or None
        Feed to :func:`build_lookup` with the same ``model_name``.
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

    # Generic template-based models share a load → extract → preintegrate shape.
    # The per-model file paths, loader functions, and flux units are the only
    # differences.
    _GENERIC_LOADERS: dict[str, tuple[tuple[str, ...], Any, str]] = {
        "dale2014": (("dale2014_templates.npz",), None, "llam"),
        "draine_li2014": (
            ("dl14_templates_v2.h5", "dl14_templates.h5"),
            load_dl14_templates,
            "lnu",
        ),
        "astrodust": (("astrodust_templates.npz",), load_astrodust_templates, "lnu"),
        "bosa": (("bosa_templates.npz",), load_bosa_templates, "lnu"),
        "themis": (("themis_templates.npz",), load_themis_templates, "lnu"),
    }

    if model_name not in _GENERIC_LOADERS:
        return None

    candidate_files, loader, units = _GENERIC_LOADERS[model_name]

    # Find first available data file
    path = None
    for fname in candidate_files:
        candidate = _find_data_file(fname)
        if candidate is not None:
            path = candidate
            break
    if path is None:
        return None

    # Dale2014 has a distinct npz layout; extract inline.
    if model_name == "dale2014":
        data = np.load(path)
        tmpl_wave = np.array(data["wavelength_aa"])
        alpha_grid = np.array(data["alpha_grid"])
        templates = np.array(data["templates_sf"])
        # Normalize shape to (n_alpha, n_wave)
        if templates.shape[0] == len(tmpl_wave) and templates.shape[1] == len(alpha_grid):
            templates = templates.T
        grid, axes, wavelength = templates, (alpha_grid,), tmpl_wave
    else:
        # DL14, Astrodust, BOSA, THEMIS share a dict layout from their loaders.
        data = loader(path)
        grid = data.get("grid", None)
        axes = data.get("axes", None)
        wavelength = data.get("wavelength", None)
        if grid is None or axes is None or wavelength is None:
            return None

    preint = precompute_template_photometry(
        templates=grid,
        wave_rest=wavelength,
        filter_waves=filter_waves,
        filter_trans=filter_trans,
        axes=axes,
        redshift=0.0 if model_name != "dale2014" else 0.0,  # both use rest-frame templates
        dl_cm=1.0,
        energy_normalize=True,
        units=units,
    )
    return _auto_collapse(preint, AXIS_PARAMS.get(model_name, ()), parameters)
