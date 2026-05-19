"""Parameter definition dictionaries and registry builder.

Pure-data module: every physics domain (dust, nebular, AGN, radio, X-ray, …)
exports a dict mapping ``param_name → (description, bound_check, bound_error,
default_distribution)``.  :func:`_build_param_registry` assembles them into the
two dicts consumed by :class:`~tengri.parameters.parameters.Parameters`.

Separated from ``parameters.py`` to keep the class file focused on behaviour
rather than data tables.
"""

from __future__ import annotations

from collections.abc import Iterable

from tengri.protocols.component import ParamDeclaration
from tengri.parameters.priors import Fixed


def _bucket_from_declarations(
    decls: Iterable[ParamDeclaration],
) -> dict[str, tuple[str, object, str, object]]:
    """Adapter: component-owned :class:`ParamDeclaration` tuple → legacy
    4-tuple bucket dict consumed by :func:`_build_param_registry` and by
    :mod:`tengri.parameters.translate`.

    A ``bound_check`` of ``None`` is normalised to ``lambda lo, hi: True``
    so the downstream code that always calls the check stays branch-free.
    """
    return {
        d.name: (
            d.description,
            d.bound_check if d.bound_check is not None else (lambda lo, hi: True),
            d.bound_error,
            d.prior,
        )
        for d in decls
    }


# ── Non-SFH parameter registry ─────────────────────────────────────────

# ``_NON_SFH_PARAMS`` was historically a junk drawer covering metallicity,
# dust attenuation, redshift, noise, and spectroscopy params. PR4 split
# it: dust attenuation entries moved to
# :mod:`tengri.components.dust._params` (``ATTENUATION_PARAMS``), and the
# remainder (``met_logzsol``, ``redshift``, ``noise_*``, ``sigma_v_kms``)
# stays here as the genuinely shared / non-component-owned set.
#
# As of ADR-0005 follow-up #1, these parameters are declared cleanly in
# :mod:`tengri.parameters._shared.PARAMS`. This dict is derived from that
# tuple to keep _param_defs.py's legacy 4-tuple contract intact for
# downstream consumers. The canonical source is now _shared.PARAMS.

from tengri.parameters import _shared as _shared_module

_NON_SFH_PARAMS = _bucket_from_declarations(_shared_module.PARAMS)

# All conditional / component-owned buckets are resolved lazily via the
# module-level ``__getattr__`` defined near the bottom of this file. The
# canonical sources live under ``tengri.components.<name>._params``:
#
#   ``_NEBULAR_PARAMS``, ``_CB19_PARAMS``, ``_ELINE_PARAMS``,
#   ``_ELINE_BROAD_PARAMS``, ``_CUE_IONSPEC_PARAMS``,
#   ``_CUE_GAS_EXTRA_PARAMS``, ``_SHOCK_PARAMS`` → nebular._params
#   ``_DUST_EMISSION_PARAMS``, ``_DUST_EXTRA_PARAMS``,
#   ``_SINGLE_COMPONENT_DUST_PARAMS`` → dust._params
#   ``_AGN_PARAMS`` → agn._params (+ ``neb_xid`` extras here)
#   ``_RADIO_PARAMS`` → radio._params
#   ``_XRAY_PARAMS`` → xray._params
#   ``_IGM_PATCHY_PARAMS``, ``_DLA_PARAMS`` → igm._params
#   ``_ALPHA_FE_PARAMS``, ``_EVOLVING_ALPHA_PARAMS`` → stellar._params
# (PR3c). Resolved lazily via module ``__getattr__`` below.

# Radio priors now live in :mod:`tengri.components.radio._params`
# (PR2 of the parameter-registry consolidation). The bucket below is a
# derived view kept for backwards compatibility with consumers that
# still iterate the legacy 4-tuple shape. Resolution is deferred via
# module ``__getattr__`` (see bottom of file) because eager import of
# ``tengri.components.radio._params`` triggers the components package
# init, which transitively re-enters this module.

# X-ray priors now live in :mod:`tengri.components.xray._params` (PR3).
# Resolved lazily via module ``__getattr__`` below.


# AGN priors now live in :mod:`tengri.components.agn._params` (PR3).
# The legacy ``_AGN_PARAMS`` bucket is resolved lazily via module
# ``__getattr__`` below, which merges the canonical agn_* tuple with
# the ``neb_xid`` orphan kept here. (``neb_xid`` is nebular-prefixed
# but consumed by the Feltre NLR backend alongside ``agn_alpha_ion``,
# so the legacy bucket has always carried it. Moving it into
# ``components/agn/_params.py`` would break the agn_* prefix invariant
# checked by ``tools/check_param_prefixes.py``.)
_AGN_EXTRAS: dict = {
    "neb_xid": (
        "Dust-to-metal ratio (Feltre NLR backend) [0.1, 0.3, 0.5]",
        lambda lo, hi: lo >= 0.05 and hi <= 0.6,
        "must be in [0.05, 0.6] (grid values: 0.1, 0.3, 0.5)",
        Fixed(0.3),
    ),
}


# (Legacy alias tables now managed in _aliases.py — imported at top)

# Settings keys that are not model parameters
SETTINGS_KEYS = frozenset(
    {
        "stochastic",
        "n_grid",
        "mean_sfh_type",
        # IGM absorption
        "apply_igm",
        # Nebular emission
        "nebular",
        "nebular_ssp",
        "nebular_cue",
        "neb_ionization",
        "cloudy_grid_path",
        "cue_weights_path",
        # Dust model & law
        "dust_model",
        "dust_approx",
        "dust_law",
        "dust_law_bc",
        "dust_law_diff",
        # Dust emission
        "dust_emission",
        "dl07_grid_path",
        # AGN
        "agn_model",
        # AGN block-recipe selectors (consumed by agn_model="composable")
        "agn_disc_block",
        "agn_lines_block",
        "agn_feii_block",
        "agn_torus_block",
        "agn_attenuation_block",
        # AGN composable-precompute axes: dict[param_name → np.ndarray grid].
        # When set with agn_model="composable", SEDModel builds the
        # composable_precompute lookup at construction time and threads
        # it through the hybrid kernel via PrecomputedData.
        "agn_axis_grids",
        # Radio & X-ray
        "radio",
        "xray",
        # Shock emission
        "shock",
        # Metallicity mode (registry-based, replaces evolving_metallicity/chem_evol)
        "met_mode",
        # Older boolean flags (resolved to met_mode internally)
        "evolving_metallicity",
        "alpha_fe_evolving",
        "chem_evol",
        # Metallicity interpolation
        "met_interp",
        "lgmet_scatter",
        # Emission line fitting mode
        "eline_mode",  # "off", "fixed", "marginalized", "fitted"
        "eline_broad",  # bool — enable broad AGN emission line component
    }
)


# ── Build parameter registry ───────────────────────────────────────────


def _build_param_registry(
    mean_sfh_type,
    nebular=False,
    dust_model="two_component",
    dust_law_bc="power_law",
    dust_law_diff=None,
    dust_emission=None,
    agn_model=None,
    radio=False,
    xray=False,
    shock=False,
    igm_patchy=False,
    dla=False,
    evolving_metallicity=False,
    alpha_fe_evolving=False,
    chem_evol=False,
    met_mode="delta",
    eline_mode="off",
    eline_broad=False,
):
    """Build the parameter registry for a given model configuration.

    Parameters
    ----------
    mean_sfh_type : list[str]
        SFH model components.
    nebular : bool or str
        Enable nebular parameters. True or "cloudy" adds neb_logU, neb_logZ_gas, neb_fesc.
    dust_model : str
        Dust geometry model: ``"two_component"`` (Charlot & Fall) or
        ``"single_component"`` (uniform screen).
    dust_law_bc : str
        Birth cloud dust law name. Non-power-law laws may add extra parameters.
    dust_law_diff : str or None
        Diffuse ISM dust law. None = same as bc.
    evolving_metallicity : bool
        If True, replace met_logzsol with met_logzsol_0 and met_logzsol_final.
    chem_evol : bool
        If True, derive Z(t) from SFH via gas-regulator model. Replaces
        met_logzsol with chem_yield, chem_eta_outflow, etc.

    Returns
    -------
    registry : dict
        param_name -> (description, bound_check, bound_error)
    defaults : dict
        param_name -> default Distribution
    """
    from tengri.components.stellar.sfh.met_registry import resolve_met
    from tengri.components.stellar.sfh.registry import resolve_sfh

    _, sfh_params, _, _ = resolve_sfh(mean_sfh_type)

    registry = {}
    defaults = {}

    # SFH params from registry
    for pname, pdef in sfh_params.items():
        registry[pname] = (pdef.description, pdef.bound_check, pdef.bound_error)
        defaults[pname] = pdef.default

    # Non-SFH global params (redshift, noise, sigma_v_kms; met_logzsol
    # is injected separately by the metallicity registry below).
    for pname, (desc, check, err, default) in _NON_SFH_PARAMS.items():
        if pname == "met_logzsol":
            continue
        registry[pname] = (desc, check, err)
        defaults[pname] = default

    # Dust attenuation params (PR4: moved to components/dust/_params.py).
    # The two Charlot-Fall optical depths are skipped under
    # ``dust_model="single_component"``; ``dust_tau_v`` from
    # ``_SINGLE_COMPONENT_DUST_PARAMS`` takes their place there.
    _is_single = dust_model == "single_component"
    from tengri.components.dust._params import (
        ATTENUATION_PARAMS,
        ATTENUATION_TWO_COMPONENT_ONLY,
    )

    for decl in ATTENUATION_PARAMS:
        if _is_single and decl.name in ATTENUATION_TWO_COMPONENT_ONLY:
            continue
        check = decl.bound_check if decl.bound_check is not None else (lambda lo, hi: True)
        registry[decl.name] = (decl.description, check, decl.bound_error)
        defaults[decl.name] = decl.prior

    if _is_single:
        from tengri.parameters._param_defs import _SINGLE_COMPONENT_DUST_PARAMS

        for pname, (desc, check, err, default) in _SINGLE_COMPONENT_DUST_PARAMS.items():
            registry[pname] = (desc, check, err)
            defaults[pname] = default

    # Metallicity params from registry (replaces ad-hoc evolving_metallicity/chem_evol)
    _, met_params, _, _ = resolve_met(met_mode)
    for pname, pdef in met_params.items():
        registry[pname] = (pdef.description, pdef.bound_check, pdef.bound_error)
        defaults[pname] = pdef.default

    # Nebular params (CLOUDY, Cue, or CB_19 — not BakedIn/ssp/off).
    # Bucket resolved lazily via module ``__getattr__`` (avoids circular
    # import through ``tengri.components``).
    if nebular in ("cloudy", "cue", "cb19"):
        from tengri.parameters._param_defs import _NEBULAR_PARAMS

        for pname, (desc, check, err, default) in _NEBULAR_PARAMS.items():
            registry[pname] = (desc, check, err)
            defaults[pname] = default

    # CB_19, alpha-Fe, shock, eline buckets are all resolved lazily via
    # the module-level ``__getattr__``.
    if nebular == "cb19":
        from tengri.parameters._param_defs import _CB19_PARAMS

        for pname, (desc, check, err, default) in _CB19_PARAMS.items():
            registry[pname] = (desc, check, err)
            defaults[pname] = default

    if alpha_fe_evolving:
        from tengri.parameters._param_defs import _EVOLVING_ALPHA_PARAMS

        for pname, (desc, check, err, default) in _EVOLVING_ALPHA_PARAMS.items():
            registry[pname] = (desc, check, err)
            defaults[pname] = default
    else:
        from tengri.parameters._param_defs import _ALPHA_FE_PARAMS

        for pname, (desc, check, err, default) in _ALPHA_FE_PARAMS.items():
            registry[pname] = (desc, check, err)
            defaults[pname] = default

    # (Dust attenuation extras now registered above with the rest of the
    # ATTENUATION_PARAMS tuple from components/dust/_params.py.)

    # Dust emission params (only when dust emission is enabled). Bucket
    # resolved lazily via module ``__getattr__`` to avoid the circular
    # load through ``tengri.components``.
    if dust_emission:
        from tengri.parameters._param_defs import _DUST_EMISSION_PARAMS

        for pname, (desc, check, err, default) in _DUST_EMISSION_PARAMS.items():
            registry[pname] = (desc, check, err)
            defaults[pname] = default

    # AGN, radio, X-ray buckets are resolved lazily via the module-level
    # ``__getattr__`` defined below — eager import would trigger a
    # circular load through ``tengri.components``.
    if agn_model:
        from tengri.parameters._param_defs import _AGN_PARAMS

        for pname, (desc, check, err, default) in _AGN_PARAMS.items():
            registry[pname] = (desc, check, err)
            defaults[pname] = default

    if radio:
        from tengri.parameters._param_defs import _RADIO_PARAMS

        for pname, (desc, check, err, default) in _RADIO_PARAMS.items():
            registry[pname] = (desc, check, err)
            defaults[pname] = default

    if xray:
        from tengri.parameters._param_defs import _XRAY_PARAMS

        for pname, (desc, check, err, default) in _XRAY_PARAMS.items():
            registry[pname] = (desc, check, err)
            defaults[pname] = default

    if shock:
        from tengri.parameters._param_defs import _SHOCK_PARAMS

        for pname, (desc, check, err, default) in _SHOCK_PARAMS.items():
            registry[pname] = (desc, check, err)
            defaults[pname] = default

    # Patchy IGM + DLA buckets (PR4: now derived from components/igm).
    if igm_patchy:
        from tengri.parameters._param_defs import _IGM_PATCHY_PARAMS

        for pname, (desc, check, err, default) in _IGM_PATCHY_PARAMS.items():
            registry[pname] = (desc, check, err)
            defaults[pname] = default

    if dla:
        from tengri.parameters._param_defs import _DLA_PARAMS

        for pname, (desc, check, err, default) in _DLA_PARAMS.items():
            registry[pname] = (desc, check, err)
            defaults[pname] = default

    if eline_mode in ("marginalized", "fitted"):
        from tengri.parameters._param_defs import _ELINE_PARAMS

        for pname, (desc, check, err, default) in _ELINE_PARAMS.items():
            registry[pname] = (desc, check, err)
            defaults[pname] = default

    if eline_broad:
        from tengri.parameters._param_defs import _ELINE_BROAD_PARAMS

        for pname, (desc, check, err, default) in _ELINE_BROAD_PARAMS.items():
            registry[pname] = (desc, check, err)
            defaults[pname] = default

    return registry, defaults


# ── Lazy bucket resolution (PR2+) ─────────────────────────────────────
#
# Component-owned ``PARAMS`` tuples are imported lazily here to avoid a
# circular import: ``tengri.components/__init__.py`` eagerly loads every
# component subpackage, and some of those transitively re-enter
# :mod:`tengri.parameters._param_defs`. Resolving on first attribute
# access defers the components import until this module has finished
# initialising.
#: Maps legacy bucket name → (component module, attribute name on that
#: module). Default attribute is ``PARAMS``; entries that read a
#: different tuple name use the 2-tuple form.
_LAZY_DECL_SOURCES: dict[str, tuple[str, str]] = {
    "_RADIO_PARAMS": ("tengri.components.radio._params", "PARAMS"),
    "_XRAY_PARAMS": ("tengri.components.xray._params", "PARAMS"),
    "_AGN_PARAMS": ("tengri.components.agn._params", "PARAMS"),
    "_NEBULAR_PARAMS": ("tengri.components.nebular._params", "PARAMS"),
    "_DUST_EMISSION_PARAMS": ("tengri.components.dust._params", "PARAMS"),
    # PR4: dust attenuation + single-component dust + IGM patchy + DLA
    "_DUST_EXTRA_PARAMS": ("tengri.components.dust._params", "ATTENUATION_PARAMS"),
    "_SINGLE_COMPONENT_DUST_PARAMS": (
        "tengri.components.dust._params",
        "SINGLE_COMPONENT_PARAMS",
    ),
    "_IGM_PATCHY_PARAMS": ("tengri.components.igm._params", "PATCHY_PARAMS"),
    "_DLA_PARAMS": ("tengri.components.igm._params", "DLA_PARAMS"),
    # PR5: nebular sub-buckets + shock + stellar α/Fe
    "_CB19_PARAMS": ("tengri.components.nebular._params", "CB19_PARAMS"),
    "_ELINE_PARAMS": ("tengri.components.nebular._params", "ELINE_PARAMS"),
    "_ELINE_BROAD_PARAMS": (
        "tengri.components.nebular._params",
        "ELINE_BROAD_PARAMS",
    ),
    "_CUE_IONSPEC_PARAMS": (
        "tengri.components.nebular._params",
        "CUE_IONSPEC_PARAMS",
    ),
    "_CUE_GAS_EXTRA_PARAMS": (
        "tengri.components.nebular._params",
        "CUE_GAS_EXTRA_PARAMS",
    ),
    "_SHOCK_PARAMS": ("tengri.components.nebular._params", "SHOCK_PARAMS"),
    "_ALPHA_FE_PARAMS": ("tengri.components.stellar._params", "ALPHA_FE_PARAMS"),
    "_EVOLVING_ALPHA_PARAMS": (
        "tengri.components.stellar._params",
        "EVOLVING_ALPHA_PARAMS",
    ),
}

#: Extra entries merged into a lazily-resolved bucket after the
#: component-owned declarations are converted. Keyed by bucket name.
_LAZY_DECL_EXTRAS: dict[str, dict] = {
    "_AGN_PARAMS": _AGN_EXTRAS,
}


def __getattr__(name: str) -> dict:
    src = _LAZY_DECL_SOURCES.get(name)
    if src is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module_path, attr = src
    mod = importlib.import_module(module_path)
    bucket = _bucket_from_declarations(getattr(mod, attr))
    extras = _LAZY_DECL_EXTRAS.get(name)
    if extras:
        bucket = {**bucket, **extras}
    globals()[name] = bucket  # cache so __getattr__ runs only once per name
    return bucket
