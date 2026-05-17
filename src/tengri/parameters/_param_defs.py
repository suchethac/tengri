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

from tengri.core.component import ParamDeclaration
from tengri.parameters.priors import Fixed, Uniform


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
_NON_SFH_PARAMS = {
    "met_logzsol": (
        "log10(Z/Zsun)",
        lambda lo, hi: True,
        "",
        Uniform(-2.0, 0.2),
    ),
    "redshift": (
        "Source redshift",
        lambda lo, hi: lo >= 0,
        "must have lo >= 0",
        Fixed(0.1),
    ),
    "noise_frac_cal": (
        "Fractional calibration noise floor (added in quadrature with obs noise)",
        lambda lo, hi: lo >= 0,
        "noise_frac_cal bounds must have lo >= 0",
        Fixed(0.0),
    ),
    "noise_dof": (
        "Student-t degrees of freedom for outlier robustness (0=Gaussian)",
        lambda lo, hi: lo >= 0,
        "noise_dof bounds must have lo >= 0",
        Fixed(0.0),
    ),
    "sigma_v_kms": (
        "Stellar velocity dispersion sigma_v [km/s] — added in quadrature "
        "to the instrumental LSF when computing spectra",
        lambda lo, hi: lo >= 0 and hi <= 2000,
        "sigma_v_kms must be in [0, 2000]",
        Fixed(0.0),
    ),
}

# Parameters that are only added when specific modules are enabled
# Nebular priors now live in :mod:`tengri.components.nebular._params`
# (PR3b). Resolved lazily via module ``__getattr__`` below — see notes
# at the bottom of this file.

# ── CB_19 extra parameters (nebular == "cb19") ────────────────────────
# CB_19 extends the base CLOUDY grid with three additional continuous axes:
# density, C/O ratio, and ΔN/O. These have no counterpart in the FSPS/Byler grid.
#
# Unit convention reminder: CB_19 stores L_line/L_Hβ (dimensionless ratios).
# The CB19Backend converts to L_sun/Q_H using L_Hβ/Q_H = 4.78e-13 erg/photon
# (Case B, T_e=10^4 K; Osterbrock & Ferland 2006, Table 4.4).
_CB19_PARAMS = {
    "neb_log_nH": (
        "Log hydrogen density log10(n_H / cm⁻³) for CB_19 grid [grid range: 1–4]",
        lambda lo, hi: lo >= 0 and hi <= 6,
        "must be in [0, 6] (CB_19 grid: 1–4; extrapolated outside)",
        Fixed(2.0),  # n_H = 100 cm⁻³, typical HII region
    ),
    "neb_co": (
        "Log C/O abundance ratio log10(C/O) for CB_19 grid [grid range: −1 to 0.15]",
        lambda lo, hi: lo >= -3 and hi <= 2,
        "must be in [−3, 2]",
        Fixed(-0.36),  # near-solar C/O (CLOUDY c17 default)
    ),
    "neb_dno": (
        "ΔN/O offset (log10) from default N/O–O/H scaling [grid range: −0.25 to 0.25]",
        lambda lo, hi: lo >= -1 and hi <= 1,
        "must be in [−1, 1]",
        Fixed(0.0),  # solar N/O scaling (Nicholls+2017)
    ),
    "neb_hbfrac": (
        "HbFrac: L_Hβ(matter-bounded)/L_Hβ(radiation-bounded) for CB_19 [0–1]. "
        "HbFrac=1 = fully radiation-bounded; escape fraction ≈ 1 − HbFrac",
        lambda lo, hi: lo >= 0 and hi <= 1,
        "must be in [0, 1]",
        Fixed(1.0),  # radiation-bounded (default)
    ),
}

# ── Emission line velocity parameters ──────────────────────────────────
_ELINE_PARAMS = {
    "eline_sigma_kms": (
        "Emission line velocity dispersion in km/s (added in quadrature to instrument resolution)",
        lambda lo, hi: lo >= 0,
        "must have lo >= 0",
        Fixed(0.0),  # Default: instrument resolution only
    ),
    "eline_delta_v_kms": (
        "Emission line velocity offset from systemic redshift in km/s",
        lambda lo, hi: True,
        "",
        Fixed(0.0),  # Default: no velocity offset
    ),
}

# ── Broad emission line parameters (AGN) ───────────────────────────────
_ELINE_BROAD_PARAMS = {
    "eline_broad_sigma_kms": (
        "Broad emission line velocity dispersion in km/s",
        lambda lo, hi: lo >= 200,
        "must have lo >= 200 km/s (broad component)",
        Uniform(500.0, 5000.0),
    ),
}

# Cue-specific optional params — only registered if user provides them
_CUE_IONSPEC_PARAMS = {
    "ionspec_index1": (
        "Cue ionizing spectrum slope segment 1 (HeII, 1-228A)",
        lambda lo, hi: lo >= 0 and hi <= 50,
        "must be in [0, 50]",
        None,
    ),
    "ionspec_index2": (
        "Cue ionizing spectrum slope segment 2 (OII, 228-353A)",
        lambda lo, hi: lo >= -1 and hi <= 35,
        "must be in [-1, 35]",
        None,
    ),
    "ionspec_index3": (
        "Cue ionizing spectrum slope segment 3 (HeI, 353-504A)",
        lambda lo, hi: lo >= -2 and hi <= 20,
        "must be in [-2, 20]",
        None,
    ),
    "ionspec_index4": (
        "Cue ionizing spectrum slope segment 4 (HI, 504-912A)",
        lambda lo, hi: lo >= -2 and hi <= 10,
        "must be in [-2, 10]",
        None,
    ),
    "ionspec_logLratio1": (
        "Cue log luminosity ratio seg2/seg1",
        lambda lo, hi: lo >= -1 and hi <= 12,
        "must be in [-1, 12]",
        None,
    ),
    "ionspec_logLratio2": (
        "Cue log luminosity ratio seg3/seg2",
        lambda lo, hi: lo >= -1 and hi <= 3,
        "must be in [-1, 3]",
        None,
    ),
    "ionspec_logLratio3": (
        "Cue log luminosity ratio seg4/seg3",
        lambda lo, hi: lo >= -1 and hi <= 3,
        "must be in [-1, 3]",
        None,
    ),
}

_CUE_GAS_EXTRA_PARAMS = {
    "gas_logn": (
        "Cue gas density log10(n_H/cm^-3)",
        lambda lo, hi: lo >= 0 and hi <= 5,
        "must be in [0, 5]",
        None,
    ),
    "gas_logno": (
        "Cue [N/O] abundance ratio (dex)",
        lambda lo, hi: lo >= -2 and hi <= 2,
        "must be in [-2, 2]",
        None,
    ),
    "gas_logco": (
        "Cue [C/O] abundance ratio (dex)",
        lambda lo, hi: lo >= -2 and hi <= 2,
        "must be in [-2, 2]",
        None,
    ),
}

_ALPHA_FE_PARAMS = {
    "met_alpha_fe": (
        "Alpha-element enhancement [alpha/Fe] (dex). "
        "Applied uniformly to all ages unless alpha_fe_evolving=True.",
        lambda lo, hi: lo >= -0.5 and hi <= 1.0,
        "must be in [-0.5, 1.0]",
        Fixed(0.0),
    ),
}

_EVOLVING_ALPHA_PARAMS = {
    "met_alpha_fe_old": (
        "[alpha/Fe] of oldest stars (at t_lookback = t_universe). "
        "Typically +0.3 to +0.5 for massive ellipticals.",
        lambda lo, hi: lo >= -0.5 and hi <= 1.0,
        "must be in [-0.5, 1.0]",
        Uniform(0.0, 0.6),
    ),
    "met_alpha_fe_young": (
        "[alpha/Fe] at present day (t_lookback ~ 0). Typically ~0.0 (solar) for disk galaxies.",
        lambda lo, hi: lo >= -0.5 and hi <= 1.0,
        "must be in [-0.5, 1.0]",
        Fixed(0.0),
    ),
}

_EVOLVING_MET_PARAMS = {
    "met_logzsol_0": (
        "Initial metallicity log10(Z/Zsun) (oldest stars)",
        lambda lo, hi: True,
        "",
        Uniform(-2.0, 0.2),
    ),
    "met_logzsol_final": (
        "Final metallicity log10(Z/Zsun) (present-day)",
        lambda lo, hi: True,
        "",
        Uniform(-2.0, 0.2),
    ),
}

_CHEM_EVOL_PARAMS = {
    "chem_yield": (
        "Nucleosynthetic yield (mass of metals per unit stellar mass locked). "
        "Typical 0.02-0.04 for solar neighborhood with Chabrier IMF.",
        lambda lo, hi: lo > 0 and hi <= 0.2,
        "must be in (0, 0.2]",
        Fixed(0.03),
    ),
    "chem_eta_outflow": (
        "Mass loading factor (Mdot_out / SFR). 0 = closed box, >0 = leaky box with outflows.",
        lambda lo, hi: lo >= 0,
        "must be >= 0",
        Fixed(0.0),
    ),
    "chem_f_gas_init": (
        "Initial gas fraction at earliest cosmic time. Default 0.9 (galaxy starts gas-dominated).",
        lambda lo, hi: lo > 0 and hi <= 1,
        "must be in (0, 1]",
        Fixed(0.9),
    ),
    "chem_return_frac": (
        "Stellar mass return fraction (instantaneous recycling). Default 0.4 for Chabrier IMF.",
        lambda lo, hi: lo >= 0 and hi < 1,
        "must be in [0, 1)",
        Fixed(0.4),
    ),
}


# Dust emission priors now live in :mod:`tengri.components.dust._params`
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

_SHOCK_PARAMS = {
    "shock_frac": (
        "Fraction of nebular Halpha replaced by shock emission [0, 1]",
        lambda lo, hi: lo >= 0 and hi <= 1,
        "must be in [0, 1]",
        Fixed(0.0),
    ),
    "shock_velocity": (
        "Shock velocity in km/s (100-1000 for MAPPINGS III; 200-1000 for MAPPINGS V)",
        lambda lo, hi: lo >= 100 and hi <= 1000,
        "must be in [100, 1000]",
        Fixed(300.0),
    ),
    "shock_log_density": (
        "Log10 pre-shock density in cm^-3; snapped to nearest grid point",
        lambda lo, hi: True,
        "",
        Fixed(0.0),
    ),
    "shock_b_over_sqrt_n": (
        "B/sqrt(n) in uG cm^(3/2) (MAPPINGS III) or absolute B in uG (MAPPINGS V); "
        "snapped to nearest grid point",
        lambda lo, hi: True,
        "",
        Fixed(1.0),
    ),
    "shock_abundance": (
        "Abundance set: solar | 2xsolar | dopita2005 | lmc | smc",
        lambda lo, hi: True,
        "",
        Fixed("solar"),
    ),
    "shock_component": (
        "Emission component: shock | precursor | combined",
        lambda lo, hi: True,
        "",
        Fixed("combined"),
    ),
}


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

    # CB_19-specific extra axes (density, C/O, ΔN/O, HbFrac)
    if nebular == "cb19":
        for pname, (desc, check, err, default) in _CB19_PARAMS.items():
            registry[pname] = (desc, check, err)
            defaults[pname] = default

    # Alpha-element enhancement
    if alpha_fe_evolving:
        # Evolving [α/Fe]: old stars more α-enhanced than young.
        # Replaces global met_alpha_fe with per-age ramp.
        for pname, (desc, check, err, default) in _EVOLVING_ALPHA_PARAMS.items():
            registry[pname] = (desc, check, err)
            defaults[pname] = default
    else:
        # Global [α/Fe] (same for all ages — defaults to Fixed(0) = no-op)
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

    # Shock emission params (only when shock=True)
    if shock:
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

    # Emission line velocity parameters (registered when eline_mode is active)
    if eline_mode in ("marginalized", "fitted"):
        for pname, (desc, check, err, default) in _ELINE_PARAMS.items():
            registry[pname] = (desc, check, err)
            defaults[pname] = default

    # Broad emission line component (AGN)
    if eline_broad:
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
