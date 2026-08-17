# SPDX-License-Identifier: BSD-3-Clause
"""Build orchestrator-compatible component chains from a single call.

Public-API helper that lets users assemble the ``run_components`` chain
from a flat set of keyword arguments without constructing each
:class:`SEDComponent` subclass by hand::

    from tengri.forward.component_factory import build_components
    from tengri.forward.orchestrator import run_components
    from tengri.protocols.component import ForwardState

    components = build_components(
        ssp_data=ssp,
        sfh_model="tsnorm",
        metallicity_model="ramp",
        dust_law_bc="calzetti",
        dust_emission_model="dale2014",
        agn_model="standard",
        use_radio=True,
        use_xray=True,
        use_igm=True,
    )
    state = run_components(components, ForwardState(wave=ssp.ssp_wave), params)

This is the **public-facing** orchestrator entry point for users who
want the raw component chain without a :class:`tengri.SEDModel`.
``SEDModel`` itself routes every prediction through the same
orchestrator internally — there is one forward path.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, NamedTuple

import jax.numpy as jnp

from tengri.components.agn.component import AGNSEDComponentConfig

# Attenuator component CLASSES are resolved from _REGISTRY via the dispatch
# seam (single dispatch, #844) — only their config dataclasses are imported here.
from tengri.components.dust.component import (
    DustAttenuationSEDComponentConfig,
)
from tengri.components.dust.two_component import (
    DustSEDComponentConfig,
)
from tengri.components.dust.wg00_model import (
    WG00AttenuationSEDComponentConfig,
)

# Non-stellar component CLASSES (nebular/dust/radio/xray/igm/agn) are resolved
# from _REGISTRY via the dispatch seam (single dispatch, #844/#845) — only their
# config dataclasses are imported. Stellar stays a direct import (the permanent
# exception: rich SFH+SSP orchestrator, never registry-dispatched).
from tengri.components.nebular.component import NebularSEDComponentConfig
from tengri.components.sed_model_component import _REGISTRY, SEDModelComponent
from tengri.components.stellar import StellarSEDComponent
from tengri.components.stellar.component import StellarSEDComponentConfig
from tengri.protocols.component import SEDComponent

__all__ = [
    "IonizingQuantities",
    "RadioQuantities",
    "XRayQuantities",
    "build_components",
    "chain_summary",
    "state_to_emission_lines",
    "state_to_ionizing_quantities",
    "state_to_radio_quantities",
    "state_to_sed_components",
    "state_to_sed_quantities",
    "state_to_sfh_quantities",
    "state_to_xray_quantities",
]

# Tiny floor used as a zero-guard when computing ratios, sSFR, and
# weighted averages where the denominator can legitimately be exactly
# zero (no stellar mass, no AGN contribution, no UV luminosity, …).
# 1e-30 sits far below any physical magnitude in M⊙, erg/s, or
# photons/s encountered in SED fitting, well above the float64
# underflow boundary, and well below double-precision rounding noise.
_TINY = 1e-30


# Alias map: grammar types -> _REGISTRY keys
_EMISSION_TYPE_ALIASES = {
    "dl07": "draine_li2007",
    "dl07_tabulated": "draine_li2007",
    "dl14": "draine_li2014",
    "mbb": "modified_blackbody",
    "draine2021_pah": "draine2021_pah_ir",
}


def _resolve_registry_component(
    domain: str,
    type_str: str,
    *,
    config: Any = None,
    **config_kwargs: Any,
) -> SEDModelComponent:
    """Resolve and construct a SEDModelComponent from the registry.

    Parameters
    ----------
    domain : str
        Component domain (e.g., "dust_emission"). Used for error messages.
    type_str : str
        Grammar type name, resolved via ``_EMISSION_TYPE_ALIASES`` to a
        registry key.
    config : SEDComponentConfig, optional
        Pre-constructed config object (not typically used for components).
    **config_kwargs
        Construction-time keyword arguments forwarded to the component's
        ``__init__``. For analytic/grid components, typically empty; some may
        accept data-path or backend overrides.

    Returns
    -------
    SEDModelComponent
        Instantiated component ready to thread into the orchestrator chain.

    Raises
    ------
    ValueError
        If ``type_str`` (after aliasing) is not found in ``_REGISTRY``. The
        error message lists every registered name, **not** only those valid for
        ``domain`` — so a mistyped ``dust_emission`` type is answered with
        ``igm``, ``radio`` and ``xray`` among the suggestions. Narrowing it
        needs a domain-to-base-class map covering all seven domains; tracked
        separately rather than guessed at here.

    Notes
    -----
    This seam is construction-time only, never traced through JAX. Template
    HDF5 loading (for grid components) happens on first ``apply()`` invocation.
    """
    # Resolve grammar type → registry key via alias map
    registry_key = _EMISSION_TYPE_ALIASES.get(type_str, type_str)

    if registry_key not in _REGISTRY:
        # Collect available components and format a helpful error
        available = sorted(_REGISTRY.keys())
        raise ValueError(
            f"{domain} component {registry_key!r} (resolved from grammar type "
            f"{type_str!r}) not found in registry. Available names: {available}"
        )

    # Instantiate the registered component class
    component_cls = _REGISTRY[registry_key]
    if config is not None:
        component = component_cls(config=config, **config_kwargs)
    else:
        component = component_cls(**config_kwargs)

    # Dust emission components (modified_blackbody, dale2014, etc.) publish to a
    # unified "dust_emission" key so they map to the canonical DerivedState
    # field. Override the instance name so _apply_precomp publishes to
    # "dust_emission_phot_lnu_precomp" instead of template-specific keys.
    if domain == "dust_emission":
        component.name = "dust_emission"

    return component


class RadioQuantities(NamedTuple):
    """Orchestrator-path mirror of the legacy
    :class:`tengri.forward.prediction.RadioProperties` accessor.

    Fields:

    - ``l_1p4ghz`` (erg/s/Hz) — radio luminosity at 1.4 GHz, integrated
      from ``state.derived["sed_radio"]`` at 21 cm rest-frame.
    - ``l_thermal`` (erg/s/Hz) — free-free thermal contribution
      computed from the published ``nion`` via
      :func:`tengri.utils.sed_quantities.compute_l_radio_thermal`.
    - ``l_nonthermal`` (erg/s/Hz) — synchrotron component
      (l_1p4ghz − l_thermal).
    - ``q_ir`` — FIR-radio correlation parameter from L_TIR and l_1p4ghz.

    All fields are JAX scalars; the NamedTuple is a JAX pytree.
    """

    l_1p4ghz: jnp.ndarray
    l_thermal: jnp.ndarray
    l_nonthermal: jnp.ndarray
    q_ir: jnp.ndarray


class XRayQuantities(NamedTuple):
    """Orchestrator-path mirror of the legacy
    :class:`tengri.forward.prediction.XRayProperties` accessor.

    Fields:

    - ``l_x_xrb`` (erg/s) — X-ray-binary luminosity (Lehmer 2010, 2016)
      computed from ``sfh_quantities.sfr_100myr`` and
      ``sfh_quantities.stellar_mass``.
    - ``l_x_agn`` (erg/s) — AGN X-ray luminosity from the published
      ``L_agn_bol`` via :func:`compute_l_x_agn`.
    - ``l_x_total`` (erg/s) — sum of the two.

    """

    l_x_xrb: jnp.ndarray
    l_x_agn: jnp.ndarray
    l_x_total: jnp.ndarray


class IonizingQuantities(NamedTuple):
    """Orchestrator-path mirror of the legacy
    :class:`tengri.forward.prediction.IonizingProperties` accessor.

    Fields:

    - ``q_h`` (photons/s) — total ionizing photon production rate;
      sourced directly from ``state.derived["nion"]``.
    - ``xi_ion`` (Hz/erg) — production efficiency q_h / νLν(1500 Å).

    """

    q_h: jnp.ndarray
    xi_ion: jnp.ndarray


def build_components(
    *,
    ssp_data: Any,
    # Stellar
    sfh_model: str = "tsnorm",
    field: bool = False,
    metallicity_model: str = "delta",
    n_grid: int = 256,
    lgmet_scatter: float = 0.2,
    # SFH -> SSP age-weight kernel: "cic" (dense cloud-in-cell integrand),
    # "dsps" (DSPS's histogram kernel), or None to auto-select (#964).
    age_kernel: str | None = None,
    # GP-field parameterization: 1.0 = non-centered (shipped), a < 1 moves
    # amplitude dependence out of the xi -> SFH map (#1355).
    field_centering: float = 1.0,
    # Nebular
    nebular_backend: str | None = "baked_in",
    nebular_backend_instance: Any | None = None,
    # When ``True`` and ``nebular_backend == "cue"``, the orchestrator
    # asks the Cue backend for the full ~271-species line catalog
    # instead of the default 128 CLOUDY/FSPS subset. See #303.
    cue_full_catalog: bool = False,
    # Shock nebular emission (MAPPINGS V) — an ADDITIVE component that
    # composes with any photoionized ``nebular_backend`` (#851). Gated by
    # the top-level ``shock={...}`` grammar group / ``Parameters(shock=True)``.
    # ``shock_norm`` selects the relative (``"frac"``) or absolute
    # (``"lhalpha"``) Halpha normalization; the categorical knobs are static.
    use_shock: bool = False,
    shock_norm: str = "frac",
    shock_abundance: str = "solar",
    shock_component: str = "combined",
    # AGN
    agn_model: str | None = None,
    # Composable-AGN block selectors (only consulted when agn_model="composable").
    # They are static Python strings, threaded into the AGNSEDComponent's
    # config so the runner can pick the right per-stage callable at trace-
    # build time. For other AGN models the registered function absorbs
    # them via ``**kwargs`` and they have no effect.
    agn_disc_block: str = "none",
    agn_nlr_block: str = "none",
    agn_blr_block: str = "none",
    agn_torus_block: str = "none",
    agn_feii_block: str = "none",
    agn_attenuation_block: str = "none",
    agn_norm: str = "cigale_joint",
    # Dust two-component
    dust_law_bc: str = "power_law",
    dust_law_diff: str = "power_law",
    dust_law_neb: str | None = None,
    dust_law_overrides: dict | None = None,
    dust_lyman_cutoff_aa: float = 0.0,
    dust_lyc_absorb_all: bool = False,
    dust_eb_include_lyc: bool = False,
    dust_emission_model: str = "modified_blackbody",
    astrodust_spinning_dust: bool = False,
    astrodust_f_cnm: float = 0.28,
    use_dust: bool = True,
    # Single-component dust (Calzetti-style screen). Picks
    # ``DustAttenuationSEDComponent`` instead of the two-component
    # ``DustSEDComponent``. ``dust_law_diff`` is reused as the screen law.
    dust_model: str = "two_component",
    # Shape parameters of the selected attenuation law that somebody actually
    # asked for (user-set or freed), resolved from spec provenance by
    # SEDModel._build_component_chain. Empty means "nobody asked", and each
    # law's own published default then stands — see
    # DustAttenuationSEDComponentConfig.live_shape_params (#1808).
    #
    # None means "no spec was consulted" and is NOT the same as empty: the
    # two-component screen keeps its historical pass-all in that case, since a
    # caller with no provenance to read cannot conclude that nobody asked
    # (#1833). The single screen treats both alike — passing nothing is its
    # historical behavior.
    dust_live_shape_params: frozenset[str] | None = None,
    # Witt & Gordon (2000) screen (dust_model="wg00", FSPS dust_type=3).
    # Static structural selectors threaded into the WG00 screen component.
    wg00_dust_curve: str = "mw",
    wg00_geometry: str = "shell",
    wg00_structure: str = "homogeneous",
    # Multiwavelength
    use_radio: bool = False,
    radio_sfr_mode: str = "bell2003",
    radio_agn_model: str = "powerlaw",
    xray_model: str = "yang20",
    use_xray: bool = False,
    use_igm: bool = False,
    igm_model: str = "inoue",
    igm_patchy: bool = False,
    use_dla: bool = False,
) -> list[SEDComponent]:
    r"""Construct an ordered :class:`SEDComponent` list for the orchestrator.

    The component order is the **canonical pipeline order**, which any
    orchestrator-driven prediction should follow:

    1. ``StellarSEDComponent`` — emits the stellar SED, publishes
       ``lnu_age``, ``ssp_ages_yr``, ``log_metallicity_history``,
       ``nion``, etc.
    2. ``NebularSEDComponent`` — adds nebular emission to
       ``sed_intrinsic`` (no-op for the BakedIn backend).
    3. ``AGNSEDComponent`` — adds AGN disc + torus + lines and
       publishes ``L_agn_bol``.
    4. ``DustSEDComponent`` — applies two-component attenuation to
       the per-age cube, integrates absorbed luminosity, adds IR
       re-emission, publishes ``L_ir``.
    5. ``RadioSEDComponent`` — synchrotron, reads ``L_ir``,
       ``log_mstar``, ``L_agn_bol`` with documented fallbacks.
    6. ``XRaySEDComponent`` — XRBs + AGN corona, reads ``sfr``,
       ``log_mstar``, ``L_agn_bol``.
    7. ``IGMSEDComponent`` — multiplies ``sed_observed`` by Inoue+2014
       transmission (no-op if no observed-frame SED yet).

    Parameters
    ----------
    ssp_data : SSPData
        Stellar-population templates, required by stellar.
    sfh_model : str
        Registered SFH model — currently ``"tsnorm"`` or ``"dpl"``.
    field : bool
        Add a stochastic GP field on top of the mean SFH.
    metallicity_model : str
        ``"delta"`` (constant Z) or ``"ramp"`` (linear log10(Z) ramp).
    n_grid : int
        SFH lookback-time grid resolution.
    lgmet_scatter : float
        Gaussian σ in log10(Z) for the DSPS triweight kernel [dex].
    age_kernel : str or None
        SFH→SSP age-weight kernel: ``"cic"`` (dense cloud-in-cell integrand),
        ``"dsps"`` (DSPS's histogram kernel), or ``None`` (default) to
        auto-select. See :class:`~tengri.components.stellar.component.StellarSEDComponentConfig`
        for the accuracy/cost tradeoff (#964).
    nebular_backend : str | None
        ``"baked_in"`` (default), ``"cloudy_grid"``, ``"cb19"``,
        ``"mappings"``, ``"cue"``, ``"shock"``, or ``None`` to omit
        nebular entirely.
    nebular_backend_instance : object | None
        Pre-constructed backend object for ``cloudy_grid`` / ``cb19`` /
        ``mappings`` / ``cue`` / ``shock`` (which need HDF5 / weights
        paths). Required for those backends.
    agn_model : str | None
        AGN model registry key (``"simple"``, ``"standard"``, …) or
        ``None`` to omit AGN.
    dust_law_bc, dust_law_diff : str
        Birth-cloud / diffuse-ISM attenuation-law registry keys.
    dust_law_neb : str or None
        Nebular birth-cloud attenuation-law key. ``None`` inherits
        ``dust_law_bc`` (nebular reddened like the youngest stars).
    dust_emission_model : str
        IR emission template registry key.
    use_dust : bool
        If ``False`` no dust component is added (no attenuation, no IR).
    use_radio, use_xray, use_igm : bool
        Add the corresponding adapter to the chain.

    Returns
    -------
    list[SEDComponent]
        Ordered component list ready to feed
        :func:`tengri.forward.orchestrator.run_components`.

    Notes
    -----
    **JIT-compatible**: yes — the returned components flow through
    ``jax.jit`` once :class:`tengri.protocols.ForwardState` is registered
    as a pytree.

    The ``StellarSEDComponent`` carries ``ssp_data`` on its instance.
    All other adapters are stateless except their config knobs.
    """
    components: list[SEDComponent] = []

    # 1. Stellar (always required — it publishes the cross-component
    #    inputs that every later adapter reads).
    components.append(
        StellarSEDComponent(
            config=StellarSEDComponentConfig(
                sfh_model=sfh_model,
                field=field,
                metallicity_model=metallicity_model,
                n_grid=n_grid,
                lgmet_scatter=lgmet_scatter,
                age_kernel=age_kernel,
                field_centering=field_centering,
            ),
            ssp_data=ssp_data,
        )
    )

    # 2. Dust (optional) — runs BEFORE AGN so the AGN component can
    # read ``state.derived["L_absorbed"]`` for the CIGALE-style
    # ``agn_power = L_abs × fracAGN/(1-fracAGN)`` cross-component
    # coupling (see ``agn/component.py`` and ``agn/_params.py:
    # agn_ir_frac``). Note: although appended here, the topological sort
    # places dust AFTER the nebular component (DustSEDComponent declares
    # ``sed_nebular`` an optional input) so the nebular continuum is
    # reddened by the HII-region dust, matching bagpipes/FSPS/CIGALE.
    # AGN/radio/xray SEDs are still passed through unattenuated by stellar
    # dust (they are added after dust runs).
    if use_dust:
        # Build the per-model attenuator config (parameterization). Class
        # SELECTION is single-dispatch: _resolve_registry_component looks the
        # attenuator up in _REGISTRY (registered in each attenuator module), so
        # build_components no longer hardcodes the component classes (#844).
        if dust_model == "wg00":
            atten_type = "wg00"
            atten_config = WG00AttenuationSEDComponentConfig(
                dust_curve=wg00_dust_curve,
                geometry=wg00_geometry,
                structure=wg00_structure,
            )
        elif dust_model == "single_component":
            atten_type = "single_component"
            atten_config = DustAttenuationSEDComponentConfig(
                law=dust_law_diff,
                live_shape_params=frozenset(dust_live_shape_params or ()),
            )
        else:
            atten_type = "two_component"
            _overrides = dust_law_overrides or {}
            atten_config = DustSEDComponentConfig(
                law_bc=dust_law_bc,
                law_diff=dust_law_diff,
                law_neb=dust_law_neb,
                # #1833: without this the shared Fixed(0.0) dust_bump_strength /
                # dust_delta overwrote each law's published default. Only reaches
                # here from SEDModel, which is the only caller that knows who
                # asked; a direct build leaves it None and keeps pass-all.
                live_shape_params=dust_live_shape_params,
                bc_law_overrides=tuple(_overrides.get("bc", {}).items()),
                diff_law_overrides=tuple(_overrides.get("diff", {}).items()),
                neb_law_overrides=tuple(_overrides.get("neb", {}).items()),
                lyman_cutoff_aa=dust_lyman_cutoff_aa,
                lyc_absorb_all=dust_lyc_absorb_all,
                eb_include_lyc=dust_eb_include_lyc,
            )

        components.append(
            _resolve_registry_component("dust_attenuation", atten_type, config=atten_config)
        )

        # Energy-balanced IR re-emission. The two-component attenuator re-emits
        # inside its own apply(); the single-screen path publishes L_ir (absorbed
        # UV/optical/NIR luminosity) and relies on a downstream emission component
        # to re-radiate it — without one, L_ir is computed but never re-emitted,
        # silently dropping the dust IR (#565). The emission component reads L_ir
        # as an optional input and produces sed_dust_ir; the topological sort places
        # it after attenuation. Route through the same single dispatch seam. WG00
        # keeps its historical behavior of appending no separate emission component.
        if atten_type != "wg00" and dust_emission_model is not None:
            # Astrodust+PAH (HD23) supports optional spinning-dust (AME) emission
            # and phase-mix configuration. Other dust-emission models do not.
            emission_config = None
            emission_kwargs = {}
            if dust_emission_model == "astrodust":
                from tengri.components.dust.emission.templates.astrodust import (
                    AstrodustIRConfig,
                )

                emission_config = AstrodustIRConfig(
                    spinning_dust=astrodust_spinning_dust,
                    f_cnm=astrodust_f_cnm,
                )

            components.append(
                _resolve_registry_component(
                    "dust_emission", dust_emission_model, config=emission_config, **emission_kwargs
                )
            )

    # 3. Nebular (optional)
    if nebular_backend is not None:
        components.append(
            _resolve_registry_component(
                "nebular",
                "nebular",
                config=NebularSEDComponentConfig(
                    backend=nebular_backend,
                    cue_full_catalog=cue_full_catalog,
                ),
                backend=nebular_backend_instance,
            )
        )

    # 3b. Shock (optional, #851) — MAPPINGS V shock emission as a separate
    # additive component. Composes with the photoionized nebular backend
    # above: both accumulate into ``sed_intrinsic`` (so both are reddened by
    # the dust screen) and publish distinct diagnostic keys (``sed_nebular``
    # vs ``sed_shock``). Resolved through the single dispatch seam.
    if use_shock:
        from tengri.components.nebular.shock_model import ShockNebularConfig

        # ShockNebular is a SEDModelComponent (config lives as a class
        # attribute, not a constructor arg), so resolve the class through the
        # seam and set the per-model config as an instance attribute — the same
        # post-construction configuration pattern the seam uses for ``name``.
        shock_component_obj = _resolve_registry_component("nebular", "shock")
        shock_component_obj.config = ShockNebularConfig(
            norm=shock_norm,
            abundance=shock_abundance,
            component=shock_component,
        )
        components.append(shock_component_obj)

    # 4. AGN (optional) — placed after dust so ``state.derived["L_absorbed"]``
    # is available for the CIGALE-coupled ``agn_ir_frac`` flow.
    if agn_model is not None:
        # NOTE (#721): ``dust_frac_agn`` (Dale2014's embedded quasar template) and
        # the composable AGN's ``agn_ir_frac`` are two distinct AGN surfaces, both
        # keyed off the same stellar ``L_absorbed`` — using both with positive
        # values double-counts AGN MIR. A *value-aware* guard cannot live here:
        # ``build_components`` sees only structural selectors, not the resolved
        # ``dust_frac_agn``/``agn_ir_frac`` values (which may be FREE), so a
        # construction-time check would false-positive on legitimate models such
        # as ``recipes.composable_agn()`` (Dale2014 + composable AGN with
        # ``dust_frac_agn=0``). The value-aware guard therefore lives at
        # ``SEDModel.build`` (``_warn_agn_dust_double_count``), where the resolved
        # spec is available: it warns only when both surfaces are positive-active
        # (FREE, or Fixed > 0), emitting a filterable ``AGNDustDoubleCountWarning``
        # (ADR-0018 §5).
        # Dispatch through the single ``_REGISTRY`` seam, like every other
        # domain (nebular/radio/xray/dust). AGN is a *composite* — one
        # component whose config selects the disc/torus/nlr/blr/feii/atten
        # sub-blocks (ADR-0018) — but its top-level dispatch is still a single
        # registered component (``_REGISTRY["agn"] = AGNSEDComponent``), so it
        # is routed and manifest-guarded uniformly (#846).
        components.append(
            _resolve_registry_component(
                "agn",
                "agn",
                config=AGNSEDComponentConfig(
                    model=agn_model,
                    agn_disc_block=agn_disc_block,
                    agn_nlr_block=agn_nlr_block,
                    agn_blr_block=agn_blr_block,
                    agn_feii_block=agn_feii_block,
                    agn_torus_block=agn_torus_block,
                    agn_attenuation_block=agn_attenuation_block,
                    agn_norm=agn_norm,
                ),
            )
        )

    # 5-7. Multiwavelength + IGM (each optional)
    if use_radio:
        from tengri.components.radio.component import RadioSEDComponentConfig

        components.append(
            _resolve_registry_component(
                "radio",
                "radio",
                config=RadioSEDComponentConfig(
                    sfr_mode=radio_sfr_mode,
                    agn_radio_model=radio_agn_model,
                ),
            )
        )
    if use_xray:
        from tengri.components.xray.component import XRaySEDComponentConfig

        # A name that registers its own component class builds that class.
        # ``xray_aird`` and ``agn_xray_corona`` each ship one -- with their own
        # config and their own ``predict`` -- and this resolved the key "xray"
        # unconditionally, passing the name as a config field instead.
        # ``XRaySEDComponent`` branches on ``config.model`` for ``lopez24`` and
        # falls through to the yang20 corona otherwise, so both names produced a
        # bit-identical SED to ``yang20``: #1684, the unfinished half of #1120,
        # which closed after adding the names to the grammar allowlist but not
        # here, turning that issue's loud ValueError into silence.
        #
        # Derived from the registry rather than a hand-written list, so a corona
        # registered later is wired by existing -- but only if the group can
        # actually feed it. A component is routed here only when its
        # ``parameter_prefix`` matches the prefix the ``xray`` group declares
        # its parameters under. ``xray_aird`` uses ``xray_`` and is fed;
        # ``agn_xray_corona`` declares ``gamma`` / ``e_cut`` /
        # ``delta_alpha_ox`` under ``agn_xray_``, which no group supplies, so
        # building it raises ``KeyError: 'gamma'`` inside ``predict``. Wiring it
        # means adding those names to the parameter space -- a public-surface
        # change, not a factory one -- so it stays on the shared component until
        # that is decided. Left on the shared component rather than swapped for
        # an internal KeyError; #1684 tracks the remaining half.
        _xray_cls = _REGISTRY.get(xray_model) if xray_model != "xray" else None
        if _xray_cls is not None and getattr(_xray_cls, "parameter_prefix", None) == "xray_":
            components.append(_resolve_registry_component("xray", xray_model))
        else:
            components.append(
                _resolve_registry_component(
                    "xray",
                    "xray",
                    config=XRaySEDComponentConfig(model=xray_model),
                )
            )
    if use_igm or use_dla:
        from tengri.components.igm.component import IGMSEDComponentConfig

        components.append(
            _resolve_registry_component(
                "igm",
                "igm",
                config=IGMSEDComponentConfig(
                    # 'none' disables the mean IGM when only a DLA is requested.
                    igm_model=igm_model if use_igm else "none",
                    igm_patchy=igm_patchy,
                    use_dla=use_dla,
                ),
            )
        )

    from tengri.forward.orchestrator import topological_sort, validate_pipeline

    # ADR-0006: derive the dependency-respecting order from declared
    # publishes/requires. The sort is stable (preserves input order
    # among components with no ordering constraint), so the canonical
    # pipeline reproduces the previous hand-coded order byte-for-byte —
    # verified by tests/contract/test_topological_sort.py. This cited
    # tests/integration/test_derived_contract_snapshots.py until that file was
    # deleted in #1029; the SED-output half of the check went with it, so the
    # surviving guarantee is the ordering one.
    components = topological_sort(components)

    # ADR-0004: construction-time contract check. After the sort,
    # validate_pipeline's "out-of-order publisher" path becomes
    # dead code (the sort guarantees publisher-before-consumer);
    # the remaining checks are duplicate-publish and units mismatch.
    validate_pipeline(components)

    return components


def chain_summary(components: Sequence[SEDComponent]) -> str:
    """Pretty-print a component chain for diagnostics.

    Returns a one-line ``→``-separated string of component names.
    Useful for log lines and error messages where the chain identity
    matters more than the parameter values.
    """
    return " → ".join(c.name for c in components)


# ─────────────────────────────────────────────────────────────────────
# ForwardState → legacy Quantities bridges
# ─────────────────────────────────────────────────────────────────────
#
# Build out the legacy ``SFHQuantities`` / ``SEDQuantities`` NamedTuples
# from a fresh :class:`ForwardState`. Lets users with code that
# expects the legacy types swap in the orchestrator path without
# rewriting downstream call sites. Full :class:`Prediction` parity
# (lines, radio, X-ray, ionizing-photon properties) is follow-up.


def state_to_sfh_quantities(state: Any):
    """Convert orchestrator :class:`ForwardState` → :class:`SFHQuantities`.

    Pulls the per-galaxy SFH derived quantities published by
    :class:`StellarSEDComponent` (``log_mstar``, ``log_mstar_formed``,
    ``sfr_10myr``, ``sfr_100myr``, ``sfh_grid_lbt_yr``, ``sfr_history``,
    ``log_metallicity_history``) and packages them in the legacy
    NamedTuple shape so existing code reading
    ``predict_sfh_quantities(...).stellar_mass`` keeps working when
    sourced from the orchestrator path.

    Parameters
    ----------
    state : ForwardState
        Output of :func:`run_components` on a chain that includes
        :class:`StellarSEDComponent`.

    Returns
    -------
    SFHQuantities
        ``stellar_mass``, ``stellar_mass_surviving``, ``sfr_100myr``,
        ``sfr_10myr``, ``ssfr``, ``mass_weighted_age_gyr``,
        ``mass_weighted_metallicity``.

    Notes
    -----
    Mass-weighted age and metallicity are computed from
    ``sfh_grid_lbt_yr`` × ``sfr_history`` (mass per bin) and
    ``log_metallicity_history``. ``ssfr`` uses the **surviving** mass
    in the denominator to match the legacy convention.

    **JIT-compatible**: yes — pure JAX.
    """
    from tengri.forward.prediction import SFHQuantities

    derived = state.derived
    log_mstar = jnp.asarray(derived["log_mstar"])
    log_mstar_formed = jnp.asarray(derived["log_mstar_formed"])
    # The *honest* surviving mass: NaN when the SSP grid has no mass-remaining
    # table, rather than ``log_mstar``'s silent fallback to the formed mass (which
    # asserts zero mass loss). ``predict_sfh_quantities`` already returned NaN
    # here; this path returned the formed mass, and the two had drifted apart
    # under one name (#1131). sSFR below keeps the fallback on purpose — see there.
    log_mstar_surviving = jnp.asarray(derived["log_mstar_surviving"])
    stellar_mass_surviving = jnp.power(10.0, log_mstar_surviving)
    stellar_mass = jnp.power(10.0, log_mstar_formed)

    sfh_lbt = jnp.asarray(derived["sfh_grid_lbt_yr"])
    sfr_history = jnp.asarray(derived["sfr_history"])
    log_z_history = jnp.asarray(derived["log_metallicity_history"])

    # Mass-weighted age on the SSP age grid — the stars the SED actually contains.
    # Integrating the raw SFH grid instead counts mass the SED truncates (an SFH
    # can form stars before the Big Bang at the model's redshift), and the two
    # answers differed by ~4.6% under this one name until #1131. Shared with the
    # property catalog and Prediction.sfh so they cannot drift apart again.
    from tengri.utils.conversions import log_z_abs_to_logzsol
    from tengri.utils.sed_quantities import compute_mass_weighted_age

    mw_age_gyr = compute_mass_weighted_age(
        jnp.asarray(derived["age_weights"]), jnp.asarray(derived["ssp_ages_yr"])
    )

    # Mass per SFH bin (∫ SFR dt locally) — the weight for the mass-weighted metallicity.
    bin_widths = jnp.gradient(sfh_lbt)
    bin_mass = jnp.maximum(sfr_history * bin_widths, 0.0)
    bin_mass_total = jnp.maximum(jnp.sum(bin_mass), _TINY)
    # ``log_metallicity_history`` is absolute log10(Z) — the SSP grid's
    # convention — and every user-facing metallicity is log10(Z/Zsun), so the
    # weighted mean is converted before it leaves. This publish point and
    # ``_mass_weighted_metallicity_fn`` are separate implementations of the
    # same average, pinned bit-equal by test_property_catalog, so they have to
    # convert together.
    mw_z = log_z_abs_to_logzsol(jnp.sum(log_z_history * bin_mass) / bin_mass_total)

    sfr_100myr = jnp.asarray(derived["sfr_100myr"])
    sfr_10myr = jnp.asarray(derived["sfr_10myr"])
    # sSFR keeps ``log_mstar``'s fallback: "how much mass survives" has no answer
    # without a mass-remaining table, but sSFR against the formed mass still does,
    # and going NaN here would be a regression. Same asymmetry as the ``ssfr``
    # property and as the method this replaces.
    ssfr = sfr_100myr / jnp.maximum(jnp.power(10.0, log_mstar), _TINY)

    return SFHQuantities(
        stellar_mass=stellar_mass,
        stellar_mass_surviving=stellar_mass_surviving,
        sfr_100myr=sfr_100myr,
        sfr_10myr=sfr_10myr,
        ssfr=ssfr,
        mass_weighted_age_gyr=mw_age_gyr,
        mass_weighted_metallicity=mw_z,
    )


def state_to_sed_quantities(state: Any):
    """Convert orchestrator :class:`ForwardState` → :class:`SEDQuantities`.

    Maps the directly-available SED quantities and computes the
    UV/break diagnostics from ``state.sed_intrinsic`` and
    ``state.wave``. Fields that the orchestrator does not yet publish
    or that require luminosity-weighting infrastructure are returned
    as ``NaN`` — callers that need them should keep using
    :meth:`SEDModel.predict_sed_quantities` until the orchestrator
    bridge is extended.

    Parameters
    ----------
    state : ForwardState
        Output of :func:`run_components` on a chain that includes
        :class:`StellarSEDComponent` (and ideally
        :class:`DustSEDComponent` for ``l_tir`` / ``l_dust_absorbed``).

    Returns
    -------
    SEDQuantities
        ``l_bol``, ``l_tir``, ``l_dust_absorbed`` populated;
        UV-slope / Dn4000 / Balmer-break / M_UV / luminosity-weighted
        quantities returned as ``NaN`` until the legacy
        :func:`compute_uv_slope_beta` machinery is migrated (see
        ``docs/dev/TODO.md``).

    Notes
    -----
    **JIT-compatible**: yes — pure JAX.
    **Units**: bolometric luminosities returned in Lsun (matches the
    legacy NamedTuple convention).
    """
    from tengri.forward.prediction import SEDQuantities
    from tengri.utils.sed_quantities import (
        compute_balmer_break,
        compute_bolometric_luminosity,
        compute_dn4000,
        compute_fuv_flux,
        compute_irx,
        compute_l_tir,
        compute_log_uv_luminosity_1600,
        compute_m_uv,
        compute_nuv_flux,
        compute_rest_uv_color,
        compute_uv_slope_beta,
        derived_luminosity_lsun,
        derived_weights_peak_relative,
    )

    sed = state.sed_intrinsic
    wave = state.wave

    # Bolometric luminosity in Lsun. Wave is ascending, so nu is descending and
    # the signed area is negative; ``abs(...)`` recovers the positive luminosity.
    # Delegating to the canonical reduction keeps this bit-identical to the
    # ``l_bol`` property and folds 1/L_sun into the integral, so the ~1e43 erg/s
    # value is never formed (float32-safe, #1206).
    l_bol = jnp.abs(compute_bolometric_luminosity(sed, wave))

    # Dust-absorbed luminosity from the orchestrator's energy-balance
    # bookkeeping — exact match for legacy ``compute_l_dust_absorbed``.
    # Reads the ``log_L_ir`` companion when the chain publishes it: the linear
    # ``L_absorbed`` is ~3.6e43 erg/s and is ``inf`` in float32, while the
    # answer here (~9.5e9 Lsun) is representable, and the attenuator computes
    # the log form first anyway (#1837).
    derived = state.derived
    l_dust_absorbed = derived_luminosity_lsun(derived, "L_absorbed", "log_L_ir")
    # L_TIR uses the legacy semantics (integration of the SED over the
    # 8–1000 μm window) for parity with ``predict_sed_quantities``,
    # not the orchestrator's energy-balance ``L_ir`` derived key —
    # the two agree when the wavelength grid extends to FIR but
    # differ on UV/optical-only grids (where the IR window is empty
    # and the energy-balance value lives outside the SED).
    l_tir = compute_l_tir(sed, wave)

    # UV / break diagnostics — pull from the legacy helpers, which
    # take ``(sed, wave)`` arrays directly.
    fuv = compute_fuv_flux(sed, wave)
    nuv = compute_nuv_flux(sed, wave)

    # IRX against the monochromatic 1600 A anchor (Meurer+99), the same
    # definition as the ``irx`` property. This used to read
    # ``compute_irx(l_tir, fuv * 2.998e15 / 1500.0)`` — the speed of light 1000x
    # too small in [A/s], inflating IRX by exactly 3 dex. ``C_AA`` was already
    # imported in this very function. See #1131; the band-averaged FUV variant is
    # published separately as ``irx_fuv``.
    # The UV anchor is passed in the log domain: nu*L_nu(1600 A) is ~5e42 erg/s
    # and is not float32-representable at all, so the linear signature returned
    # NaN there for an IRX of order -0.8 dex (#1837).
    irx = compute_irx(l_tir, log_l_uv_erg=compute_log_uv_luminosity_1600(sed, wave))

    # Pre-dust stellar SED reconstructed from the per-age cube
    # ``lnu_age``, which StellarSEDComponent publishes before
    # DustSEDComponent overwrites ``sed_intrinsic``. If no per-age
    # cube is present (chain has no stellar component), fall back to
    # NaN.
    from tengri.utils.conversions import log_z_abs_to_logzsol as _log_z_abs_to_logzsol

    nan_scalar = jnp.asarray(jnp.nan)
    if "lnu_age" in derived:
        sed_stellar_intrinsic = jnp.sum(jnp.asarray(derived["lnu_age"]), axis=0)
        fuv_intrinsic = compute_fuv_flux(sed_stellar_intrinsic, wave)
        nuv_intrinsic = compute_nuv_flux(sed_stellar_intrinsic, wave)
    else:
        fuv_intrinsic = nan_scalar
        nuv_intrinsic = nan_scalar

    # Luminosity-weighted age and metallicity. Stellar publishes
    # ``L_age`` (bolometric luminosity per SSP age bin, erg/s) and
    # ``ssp_ages_yr`` (n_age,). For metallicity we use the SFH-grid
    # log_metallicity_history evaluated at each SSP age via linear
    # interpolation — the metallicity ramp is monotonic in lookback
    # time so this is sound for the supported (delta, ramp) modes.
    #
    # The weights are used only inside ``sum(x*w)/sum(w)``, so any common
    # factor cancels exactly. Normalizing by the peak bin keeps every weight in
    # [0, 1]: raw ``L_age`` peaks at ~3.3e42 erg/s, so 85 of 93 bins were
    # ``inf`` in float32 and ``ssp_ages_yr * L_age`` overflowed a second time on
    # top (~1e10 x). Both weighted means are of order 1 (#1837).
    if "L_age" in derived and "ssp_ages_yr" in derived:
        ssp_ages_yr = jnp.asarray(derived["ssp_ages_yr"])
        L_age = derived_weights_peak_relative(derived, "L_age", "log_L_age")
        L_total = jnp.maximum(jnp.sum(L_age), _TINY)
        lw_age_yr = jnp.sum(ssp_ages_yr * L_age) / L_total
        lw_age_gyr = lw_age_yr / 1e9
        if "log_metallicity_history" in derived and "sfh_grid_lbt_yr" in derived:
            lz_per_ssp = jnp.interp(
                ssp_ages_yr,
                jnp.asarray(derived["sfh_grid_lbt_yr"]),
                jnp.asarray(derived["log_metallicity_history"]),
            )
            # Absolute log10(Z) on the grid → log10(Z/Zsun) on the way out;
            # mirrors ``_luminosity_weighted_metallicity_fn``.
            lw_z = _log_z_abs_to_logzsol(jnp.sum(lz_per_ssp * L_age) / L_total)
        else:
            lw_z = nan_scalar
    else:
        lw_age_gyr = nan_scalar
        lw_z = nan_scalar

    return SEDQuantities(
        l_bol=l_bol,
        l_tir=l_tir,
        l_dust_absorbed=l_dust_absorbed,
        irx=irx,
        uv_slope_beta=compute_uv_slope_beta(sed, wave),
        dn4000=compute_dn4000(sed, wave),
        balmer_break=compute_balmer_break(sed, wave),
        m_uv=compute_m_uv(sed, wave),
        fuv_flux=fuv,
        nuv_flux=nuv,
        fuv_flux_intrinsic=fuv_intrinsic,
        nuv_flux_intrinsic=nuv_intrinsic,
        rest_uv_color=compute_rest_uv_color(sed, wave),
        luminosity_weighted_age_gyr=lw_age_gyr,
        luminosity_weighted_metallicity=lw_z,
    )


def state_to_radio_quantities(state: Any) -> RadioQuantities:
    """Convert :class:`ForwardState` → :class:`RadioQuantities`.

    Reads ``state.derived["sed_radio"]`` (the radio-component-published
    SED in erg/s/Hz on the rest-frame wave grid) and interpolates at
    21 cm (= 1.4 GHz) to populate ``l_1p4ghz``. Thermal / non-thermal
    split uses the published ``log_nion`` and the log-domain
    :func:`tengri.utils.sed_quantities.compute_l_radio_thermal_from_log_qh`.

    Returns
    -------
    RadioQuantities
        ``l_1p4ghz``, ``l_thermal``, ``l_nonthermal``, ``q_ir``.

    Notes
    -----
    Returns ``NaN`` fields when the chain did not include
    :class:`RadioSEDComponent` (no ``L_radio`` published).
    """
    from tengri.utils.sed_quantities import (
        compute_l_radio_thermal_from_log_qh,
        compute_q_ir,
        derived_luminosity_lsun,
    )

    derived = state.derived
    nan_scalar = jnp.asarray(jnp.nan)
    if "sed_radio" not in derived:
        return RadioQuantities(
            l_1p4ghz=nan_scalar,
            l_thermal=nan_scalar,
            l_nonthermal=nan_scalar,
            q_ir=nan_scalar,
        )
    L_radio = jnp.asarray(derived["sed_radio"])  # erg/s/Hz on rest-frame wave grid
    wave = state.wave
    # 21 cm = 2.1e9 Å. The radio component computes at all wavelengths.
    wave_21cm = 21.106e8  # Å — 1.4 GHz exactly
    l_1p4ghz = jnp.interp(wave_21cm, wave, L_radio)

    log_nion = jnp.asarray(derived.get("log_nion", -jnp.inf))
    l_thermal = compute_l_radio_thermal_from_log_qh(log_nion)
    l_nonthermal = l_1p4ghz - l_thermal

    # Same seam as ``l_dust_absorbed``: the linear ``L_ir`` is ~3.6e43 erg/s and
    # is ``inf`` in float32, while q_IR is a dex ratio of order 2 (#1837).
    l_tir_lsun = derived_luminosity_lsun(derived, "L_ir", "log_L_ir")
    q_ir = compute_q_ir(l_tir_lsun, l_1p4ghz)

    return RadioQuantities(
        l_1p4ghz=l_1p4ghz,
        l_thermal=l_thermal,
        l_nonthermal=l_nonthermal,
        q_ir=q_ir,
    )


def state_to_xray_quantities(state: Any) -> XRayQuantities:
    """Convert :class:`ForwardState` → :class:`XRayQuantities`.

    Uses the SFH-derived SFR and stellar mass to compute the XRB
    luminosity (Lehmer+10/16) and the published ``L_agn_bol`` to
    compute the AGN corona luminosity (Duras+20).

    Returns
    -------
    XRayQuantities
        ``l_x_xrb``, ``l_x_agn``, ``l_x_total``.
    """
    from tengri.utils.sed_quantities import compute_l_x_agn, compute_l_x_xrb

    derived = state.derived
    sfr = jnp.asarray(derived.get("sfr_100myr", derived.get("sfr", 0.0)))
    log_mstar = jnp.asarray(derived.get("log_mstar", 0.0))
    mstar = jnp.power(10.0, log_mstar)
    l_x_xrb = compute_l_x_xrb(sfr, mstar)

    L_agn_bol = jnp.asarray(derived.get("L_agn_bol", 0.0))
    # ``compute_l_x_agn`` uses log10 internally — protect against the
    # zero-AGN case where the conversion would produce -inf/NaN.
    l_x_agn = jnp.where(L_agn_bol > 0.0, compute_l_x_agn(jnp.maximum(L_agn_bol, _TINY)), 0.0)

    return XRayQuantities(
        l_x_xrb=l_x_xrb,
        l_x_agn=l_x_agn,
        l_x_total=l_x_xrb + l_x_agn,
    )


def state_to_ionizing_quantities(state: Any) -> IonizingQuantities:
    """Convert :class:`ForwardState` → :class:`IonizingQuantities`.

    Reads ``state.derived["nion"]`` for q_h (ionizing photon rate, photons/s;
    deferred to #1206 items 2/3) and computes ``xi_ion`` from ``log_nion``
    using the log-domain helper for float32 safety.

    Returns
    -------
    IonizingQuantities
        ``q_h``, ``xi_ion``.
    """
    from tengri.utils.sed_quantities import compute_xi_ion_from_log_qh

    derived = state.derived
    nan_scalar = jnp.asarray(jnp.nan)
    q_h = jnp.asarray(derived.get("nion", nan_scalar))

    sed = state.sed_intrinsic
    if sed is None:
        xi_ion = nan_scalar
    else:
        log_nion = jnp.asarray(derived.get("log_nion", -jnp.inf))
        xi_ion = compute_xi_ion_from_log_qh(log_nion, sed, state.wave)

    return IonizingQuantities(q_h=q_h, xi_ion=xi_ion)


def state_to_sed_components(state: Any) -> dict:
    r"""Convert :class:`ForwardState` → per-component SED decomposition.

    Reads the per-component SED arrays every adapter publishes into
    ``state.derived`` under the ADR-0009 typed contract — the single
    source both :attr:`Prediction.sed.components
    <tengri.forward.prediction.SEDProperties.components>` and
    :meth:`Posterior.sed_components
    <tengri.inference.posterior.Posterior.sed_components>` decompose from.

    Parameters
    ----------
    state : ForwardState
        Output of :func:`run_components` on any component chain
        (missing components decompose to zeros).

    Returns
    -------
    dict
        ``wavelength`` — rest-frame grid [Angstrom], shape ``(n_wave,)`` —
        plus per-component rest-frame :math:`L_\nu` [erg/s/Hz], each of
        shape ``(n_wave,)``:

        - ``sed_total`` — accumulated post-chain total
          (``state.sed_intrinsic`` after every adapter ran);
        - ``sed_intrinsic`` — stellar pre-attenuation,
          ``sum(lnu_age, axis=0)``;
        - ``sed_attenuated`` — stellar post-attenuation
          (``sed_dust_attenuated``; falls back to intrinsic when no
          dust adapter ran);
        - ``sed_nebular``, ``sed_shock``, ``sed_dust_ir``, ``sed_agn``,
          ``sed_radio``, ``sed_xray`` — each component's own published
          contribution (zeros when the component is absent).

    Notes
    -----
    **JIT-compatible**: yes — pure reads of published arrays with
    ``jnp`` fallbacks; no Python branching on traced values.
    """
    derived = state.derived
    wave = jnp.asarray(state.wave)
    zeros = jnp.zeros(wave.shape[0])

    # Stellar pre-attenuation: sum the per-age cube. Always present
    # because StellarSEDComponent is mandatory in every chain.
    lnu_age = derived.get("lnu_age")
    sed_intrinsic_stellar = jnp.sum(jnp.asarray(lnu_age), axis=0) if lnu_age is not None else zeros

    # Stellar post-attenuation. ``sed_dust_attenuated`` is the canonical
    # key (DustSEDComponent); fall back to stellar intrinsic when no
    # dust adapter ran.
    sed_attenuated_stellar = jnp.asarray(derived.get("sed_dust_attenuated", sed_intrinsic_stellar))

    return {
        "wavelength": wave,
        "sed_total": jnp.asarray(state.sed_intrinsic)
        if state.sed_intrinsic is not None
        else zeros,
        "sed_attenuated": sed_attenuated_stellar,
        "sed_intrinsic": sed_intrinsic_stellar,
        "sed_nebular": jnp.asarray(derived.get("sed_nebular", zeros)),
        "sed_shock": jnp.asarray(derived.get("sed_shock", zeros)),
        "sed_dust_ir": jnp.asarray(derived.get("sed_dust_ir", zeros)),
        "sed_agn": jnp.asarray(derived.get("sed_agn", zeros)),
        "sed_radio": jnp.asarray(derived.get("sed_radio", zeros)),
        "sed_xray": jnp.asarray(derived.get("sed_xray", zeros)),
    }


def state_to_emission_lines(state: Any):
    """Convert :class:`ForwardState` → :class:`EmissionLines`.

    Reads the discrete line catalog
    ``state.derived["line_waves"]`` / ``state.derived["line_lums"]``
    published by :class:`NebularSEDComponent` (when the active backend is
    Cue or CloudyGrid) and extracts the 11 headline survey-diagnostic
    lines via the legacy nearest-wavelength matcher
    :func:`tengri.utils.sed_quantities.extract_line_luminosity`. The full
    backend catalog (typically ~138–271 species) is also exposed via
    ``all_waves`` / ``all_lums`` for downstream lookups of species the
    headline NamedTuple does not name explicitly (HeII 1640, HeI 10830,
    [O III] 4363, ...).

    Dust attenuation: the published luminosities already include the
    attenuation regime selected by the SEDModel's ``_neb_dust_mode``
    when ``predict_emission_lines`` routes through
    :meth:`SEDModel.predict_emission_lines`. Direct callers of this
    helper see the *intrinsic* line luminosities — apply
    :func:`tengri.forward.emission_helpers.attenuate_emission` (or call
    via ``model.predict_emission_lines``) for the observed values.

    Returns
    -------
    EmissionLines
        Headline scalars (``halpha``, ``hbeta``, ``oiii_5007``, ...) plus
        the full ``all_waves`` / ``all_lums`` arrays — all in **[erg/s]**,
        passed through unconverted from ``state.derived["line_lums"]``, whose
        ``DerivedKey`` declares that unit and which
        :meth:`~tengri.forward.sed_model.SEDModel.predict_line_fluxes` consumes
        as such.

        This said "Lsun" until #1559. It was wrong then too — the bridge has
        never converted anything — and it was the documentation three backends
        were written against, which is how they came to publish [Lsun] into an
        [erg/s] key and emit line fluxes a factor 3.839e33 too faint.

    Notes
    -----
    Returns all-NaN headlines and empty ``all_*`` arrays when the
    chain's nebular backend did not publish a line catalog (BakedIn —
    emission baked into SSP grid; shock — publishes a continuous line
    SED, not a discrete list). For those cases callers should query
    ``state.derived["sed_nebular"]`` and perform their own narrow-band
    integration.
    """
    from tengri.forward.prediction import EmissionLines
    from tengri.utils.sed_quantities import KEY_LINES, extract_line_luminosity

    derived = state.derived
    nan_scalar = jnp.asarray(jnp.nan)
    empty = jnp.asarray([], dtype=jnp.float64)
    if "line_waves" not in derived or "line_lums" not in derived:
        # No discrete catalog published. Return all-NaN + empty all_*.
        kw = {k: nan_scalar for k in EmissionLines._fields if k not in ("all_waves", "all_lums")}
        return EmissionLines(all_waves=empty, all_lums=empty, **kw)

    line_waves = jnp.asarray(derived["line_waves"])
    line_lums = jnp.asarray(derived["line_lums"])

    return EmissionLines(
        lya=extract_line_luminosity(line_waves, line_lums, KEY_LINES["lya"]),
        civ_1549=extract_line_luminosity(line_waves, line_lums, KEY_LINES["civ_1549"]),
        oii=extract_line_luminosity(line_waves, line_lums, KEY_LINES["oii"]),
        hbeta=extract_line_luminosity(line_waves, line_lums, KEY_LINES["hbeta"]),
        oiii_4959=extract_line_luminosity(line_waves, line_lums, KEY_LINES["oiii_4959"]),
        oiii_5007=extract_line_luminosity(line_waves, line_lums, KEY_LINES["oiii_5007"]),
        nii_6548=extract_line_luminosity(line_waves, line_lums, KEY_LINES["nii_6548"]),
        halpha=extract_line_luminosity(line_waves, line_lums, KEY_LINES["halpha"]),
        nii_6584=extract_line_luminosity(line_waves, line_lums, KEY_LINES["nii_6584"]),
        sii_6717=extract_line_luminosity(line_waves, line_lums, KEY_LINES["sii_6717"]),
        sii_6731=extract_line_luminosity(line_waves, line_lums, KEY_LINES["sii_6731"]),
        all_waves=line_waves,
        all_lums=line_lums,
    )
