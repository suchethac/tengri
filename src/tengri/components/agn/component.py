# SPDX-License-Identifier: BSD-3-Clause
"""AGNSEDComponent: unified AGN emission (disc + torus + lines + jets/corona).

Dispatches to any registered AGN model (``"multicolor_agn"``,
``"kubota_done_full"``, ``"adaf"``, ``"unified_nlr_blr"``, ``"skirtor"``,
``"silva04"``, ``"cat3d_wind"``, ``"relagn"``, ``"qsogen"``) via
:func:`tengri.components.agn.unified.resolve_agn_model`.

Cross-component reads
---------------------
Nothing required — AGN is a self-contained additive emitter. ``redshift``
is read from :data:`BARE_NAME_ALLOWLIST`.

Cross-component publications
----------------------------

- ``state.derived["L_agn_bol"]`` (scalar, erg/s) — bolometric AGN
  luminosity. Consumed by
  :class:`tengri.components.xray.XRaySEDComponent` and
  :class:`tengri.components.radio.RadioSEDComponent` via their
  documented fallback (``state.derived.get("L_agn_bol", 0.0)``).
- ``state.derived["sed_agn"]`` — the AGN SED contribution
  (erg/s/Hz, shape n_wave) for diagnostics.

Architectural notes
-------------------
``agn_torus_frac`` is **never** auto-derived from ``agn_cos_inc`` /
``theta_torus`` (CLAUDE.md gotcha — gradient discontinuity). It is
read directly from ``params`` as an independent free parameter.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import jax.numpy as jnp

from tengri.components.agn._params import PARAMS as _AGN_PARAMS
from tengri.components.agn.unified import resolve_agn_model
from tengri.components.xray.xray import COS_INC_REF_30DEG as _XRAY_COS_INC_REF_30DEG
from tengri.parameters.resolve import require_redshift
from tengri.protocols.component import (
    DerivedKey,
    ForwardState,
    ParamDeclaration,
    SEDComponentConfig,
    SEDComponentState,
)
from tengri.utils.physics_constants import L_SUN

__all__ = ["AGNSEDComponent", "AGNSEDComponentConfig"]


@dataclass(frozen=True)
class AGNSEDComponentConfig(SEDComponentConfig):
    """Frozen knobs for :class:`AGNSEDComponent`.

    Parameters
    ----------
    name : str
        Diagnostic identifier. Default ``"agn"``.
    model : str
        AGN model registry key. One of ``"multicolor_agn"`` (Kubota & Done
        outer-zone disc + 2-T torus), ``"kubota_done_full"`` (full 3-zone
        disc), ``"adaf"``, ``"unified_nlr_blr"``, ``"skirtor"``,
        ``"silva04"``, ``"cat3d_wind"``, ``"relagn"``, ``"qsogen"``, or
        ``"composable"`` (block-composed via the selectors below).
        Default ``"multicolor_agn"``.
    agn_disc_block, agn_nlr_block, agn_blr_block, agn_feii_block,
    agn_torus_block, agn_attenuation_block : str
        Composable-AGN block selectors. Only consulted when
        ``model == "composable"`` — the runner reads them from this config
        (they are static strings, not traced JAX values, so they cannot
        ride in ``params``). Each defaults to ``"none"`` so non-composable
        AGN models receive harmless no-op selectors that the underlying
        registry function absorbs via ``**kwargs``.
    agn_norm : str
        Cross-block normalization policy (#556). ``"cigale_joint"``
        (default) ties the disc, torus and polar to CIGALE's single
        ``agn_power`` reference via the fixed SKIRTOR template ratios
        (energy-conserving; only active for ``agn_torus_block="skirtor"`` +
        ``agn_ir_frac>0``). ``"independent"`` keeps each component on its own
        luminosity scale (disc on ``agn_log_lbol``, torus on ``agn_power``,
        polar via the legacy face-on proxy) — the GRAHSP/AGNfitter-style
        bookkeeping. A static string, read by the runner like the block
        selectors.
    """

    name: str = "agn"
    model: str = "multicolor_agn"
    agn_disc_block: str = "none"
    agn_nlr_block: str = "none"
    agn_blr_block: str = "none"
    agn_feii_block: str = "none"
    agn_torus_block: str = "none"
    agn_attenuation_block: str = "none"
    agn_norm: str = "cigale_joint"


@dataclass(frozen=True)
class AGNSEDComponentState(SEDComponentState):
    """State for AGN component.

    Holds optional filter passbands when ``approx=WavePrecomp()``
    is set on the parent SEDModel. The component uses them at
    :meth:`AGNSEDComponent.apply` time to filter-integrate the
    analytically computed AGN SED and publish ``agn_phot_lnu_precomp``.

    Also optionally caches SKIRTOR torus template grids for JIT threading
    so they become Parameter ops rather than baked Constants.
    """

    name: str = "agn"
    filter_waves: Any | None = None
    filter_trans: Any | None = None
    skirtor_templates: Any | None = None


@dataclass(frozen=True)
class AGNSEDComponent:
    """SEDComponent adapter for the unified AGN model registry.

    Notes
    -----
    **JIT-compatible**: yes — :meth:`apply` delegates to a registered
    AGN function which is pure JAX.
    **Additive**: writes ``sed_intrinsic = sed_intrinsic + L_AGN(λ)``,
    matching the convention used by :class:`RadioSEDComponent` and
    :class:`XRaySEDComponent`.
    """

    config: AGNSEDComponentConfig = field(default_factory=AGNSEDComponentConfig)
    name: str = "agn"
    parameter_prefix: str = "agn_"
    _state: AGNSEDComponentState | None = None

    def citations(self) -> tuple[str, ...]:
        """AGN sub-block citations (disc, torus, BLR) flow from
        :mod:`tengri.citations.associations`; the wrapper has no
        always-required paper."""
        return ()

    def declared_parameters(self) -> list[ParamDeclaration]:
        """Free parameters this component owns.

        Returns the canonical :data:`PARAMS` tuple from
        :mod:`tengri.components.agn._params`. The legacy ``_AGN_PARAMS``
        bucket in :mod:`tengri.parameters._param_defs` is a derived view
        of the same tuple (plus the ``neb_xid`` orphan kept in the
        registry for the Feltre NLR backend).

        The full ``agn_*`` parameter superset is declared so that any
        registered model can run without missing keys. Users freely
        ``Fixed`` whatever a particular model does not consume; the
        AGN registry functions accept ``**kwargs`` and ignore unused
        names.
        """
        return list(_AGN_PARAMS)

    def outputs(self) -> tuple[DerivedKey, ...]:
        """Cross-component derived keys this AGN component publishes.

        See :func:`tengri.forward.orchestrator.validate_pipeline`.
        """
        return (
            DerivedKey("L_agn_bol", "erg/s", "AGN bolometric luminosity"),
            DerivedKey("sed_agn", "erg/s/Hz", "AGN SED contribution on pipeline wave grid"),
            DerivedKey(
                "L_2500_intrinsic",
                "erg/s/Hz",
                "AGN intrinsic disc L_nu at 2500 A (un-reddened); drives X-ray alpha_ox",
            ),
            DerivedKey(
                "L_4400_intrinsic",
                "erg/s/Hz",
                "AGN intrinsic disc L_nu at 4400 A (un-reddened); drives radio loudness",
            ),
            DerivedKey(
                "agn_cos_inc",
                "dimensionless",
                "AGN cos(i) — the X-ray corona tilts its Yang+2022 anisotropy "
                "to the same sightline (X-CIGALE yang20.py cosi; #980)",
            ),
        )

    def precompute(
        self,
        ssp_data: Any | None = None,
        wave_grid: jnp.ndarray | None = None,
        approx: Mapping[str, bool] | None = None,
        filters: tuple[tuple[jnp.ndarray, jnp.ndarray], ...] | None = None,
    ) -> AGNSEDComponentState:
        r"""Cache filter passbands and SKIRTOR templates for precomputation.

        When ``approx=WavePrecomp()`` is set:

        - Stores filter passbands so :meth:`apply` can publish
          ``agn_phot_lnu_precomp`` (filter-integrated AGN photometry).
        - If the AGN model is "skirtor", pre-loads the template grids
          so they thread through JIT as Parameter ops.

        Parameters
        ----------
        ssp_data : Any | None
            Unused; accepted for Protocol uniformity.
        wave_grid : ndarray | None
            Unused; accepted for Protocol uniformity.
        approx : Mapping[str, bool] | None
            Approximation flags. When ``approx.get('wave_precomp')``
            is ``True``, cache templates and filters.
        filters : tuple of (wave, trans) pairs | None
            Filter wavelengths and transmissions to cache.

        Returns
        -------
        AGNSEDComponentState
            State with optionally-populated filter_waves, filter_trans,
            and skirtor_templates.
        """
        del ssp_data, wave_grid
        approx = approx or {}

        filter_waves = None
        filter_trans = None
        skirtor_templates = None

        if approx.get("wave_precomp") and filters is not None:
            filter_waves = tuple(jnp.asarray(fw) for fw, _ in filters)
            filter_trans = tuple(jnp.asarray(ft) for _, ft in filters)

        # SKIRTOR torus models require template grids. Load them here so
        # they're available for JIT threading.
        if self.config.model == "skirtor":
            try:
                from tengri.components.agn.skirtor import _load_skirtor_default_grid

                # Store the template ARRAYS (a SKIRTORGrid pytree), not the
                # interpolation closure — arrays thread through jax.jit as a
                # runtime input; a closure cannot (#1198).
                skirtor_templates = _load_skirtor_default_grid()
            except Exception:
                # If SKIRTOR template loading fails (file not found),
                # gracefully continue without threading. The apply() path
                # will fall back to lazy loading.
                pass

        return AGNSEDComponentState(
            name=self.name,
            filter_waves=filter_waves,
            filter_trans=filter_trans,
            skirtor_templates=skirtor_templates,
        )

    def apply(
        self,
        state: ForwardState,
        params: Mapping[str, jnp.ndarray],
        ssp_data: Any | None = None,
        template_data: Any | None = None,
    ) -> ForwardState:
        """Add AGN emission to ``state.sed_intrinsic`` and publish ``L_agn_bol``.

        Parameters
        ----------
        state : ForwardState
            Must carry rest-frame ``wave`` (Å). If ``sed_intrinsic`` is
            ``None`` it is initialized to zeros of the same shape.
        params : mapping
            Receives ``agn_*`` keys plus the bare ``redshift``.
        template_data : dict | None
            Nested dict with component namespaces ("nebular", "agn", etc)
            carrying template grids/weights for JIT threading. When present,
            SKIRTOR templates are read from ``template_data["agn"]["skirtor"]``
            as a JIT runtime input instead of module-level lazy-loaded cache.

        Returns
        -------
        ForwardState
            New state with ``sed_intrinsic`` updated and ``L_agn_bol``
            published to ``derived``.
        """
        wave = state.wave

        # Convert log10(L_bol/Lsun) → erg/s once for both the model
        # call and the L_agn_bol publication.
        agn_log_lbol = jnp.asarray(params["agn_log_lbol"])
        L_agn_bol = jnp.power(10.0, agn_log_lbol) * L_SUN

        # CIGALE-faithful cross-component coupling
        # ────────────────────────────────────────────────────────────
        # When ``agn_ir_frac > 0`` we follow CIGALE skirtor2016's
        # bookkeeping: the AGN dust-IR power is derived from the
        # stellar dust-absorbed luminosity via
        # ``agn_power = L_absorbed × fracAGN / (1 − fracAGN)``
        # (skirtor2016.py:498). Below we OVERRIDE ``agn_torus_frac`` so
        # the existing block normalization ``l_scale = L_bol × frac``
        # produces ``l_scale = agn_power``. This way the torus block
        # API stays unchanged while the higher-level driver matches
        # CIGALE bit-for-bit. ``lambda_fracAGN="0/0"`` (CIGALE's
        # whole-IR default) is assumed; the alternative wavelength-
        # window flow is not yet wired.
        agn_ir_frac = jnp.asarray(params.get("agn_ir_frac", 0.0))
        L_absorbed = jnp.asarray(state.derived.get("L_absorbed", 0.0))
        # Avoid divide-by-zero / negative leak when fracAGN ≥ 1.
        _one_minus_frac = jnp.maximum(1.0 - agn_ir_frac, 1e-6)
        agn_power_from_stellar = L_absorbed * agn_ir_frac / _one_minus_frac
        agn_torus_frac_user = jnp.asarray(params.get("agn_torus_frac", 0.5))
        agn_torus_frac_effective = jnp.where(
            agn_ir_frac > 0.0,
            agn_power_from_stellar / jnp.maximum(L_agn_bol, 1e-30),
            agn_torus_frac_user,
        )

        # Resolve the AGN model from the registry. resolve_agn_model is
        # a factory-time lookup but the returned callable is pure JAX
        # so it folds into a JIT trace cleanly. If template_data is
        # provided, extract AGN SKIRTOR template and thread it.
        agn_fn = resolve_agn_model(self.config.model)

        # Thread the SKIRTOR template as a JIT runtime input
        skirtor_template = None
        if template_data is not None and isinstance(template_data, dict):
            agn_data = template_data.get("agn")
            if agn_data is not None and isinstance(agn_data, dict):
                skirtor_template = agn_data.get("skirtor")
        # Build kwargs, adding _template for SKIRTOR threading if available.
        # The agn_kwargs dict is built in two passes:
        #   1. Explicit defaults for the AGN params the registered models
        #      most commonly read (kept here so missing keys don't break
        #      multicolor / skirtor / kubota_done at trace time).
        #   2. Forward every additional ``agn_*`` key from ``params`` so
        #      block-specific params (skirtor, grahsp, cat3d_wind, etc.)
        #      reach the registered function via its ``**kwargs`` tail.
        agn_kwargs = {
            "agn_lum_ratio": jnp.asarray(params.get("agn_lum_ratio", 1.0)),
            "agn_alpha": jnp.asarray(params.get("agn_alpha", -1.0)),
            "agn_log_mbh": jnp.asarray(params.get("agn_log_mbh", 8.0)),
            "agn_log_ledd": jnp.asarray(params.get("agn_log_ledd", -1.0)),
            "agn_a_spin": jnp.asarray(params.get("agn_a_spin", 0.0)),
            # CIGALE-coupled override: see ``agn_torus_frac_effective``
            # block above. When ``agn_ir_frac > 0`` this carries the
            # stellar-derived agn_power; otherwise it's the user value.
            "agn_torus_frac": agn_torus_frac_effective,
            "agn_T_torus": jnp.asarray(params.get("agn_T_torus", 1000.0)),
            "agn_tau_torus": jnp.asarray(params.get("agn_tau_torus", 3.0)),
            "agn_T_hot": jnp.asarray(params.get("agn_T_hot", 1500.0)),
            "agn_T_warm": jnp.asarray(params.get("agn_T_warm", 300.0)),
            "agn_frac_hot": jnp.asarray(params.get("agn_frac_hot", 0.5)),
            "agn_cos_inc": jnp.asarray(params.get("agn_cos_inc", 0.5)),
            "agn_polar_ebv": jnp.asarray(params.get("agn_polar_ebv", 0.0)),
            "agn_polar_oa": jnp.asarray(params.get("agn_polar_oa", 40.0)),
            "agn_ebv_disc": jnp.asarray(params.get("agn_ebv_disc", 0.0)),
        }
        # Sweep every remaining ``agn_*`` key in params (skirtor / grahsp /
        # kubota_done full-disc / cat3d_wind / radiation-physics extras)
        # through to the model function. ``agn_log_lbol`` is passed
        # explicitly below and so excluded here.
        for key, val in params.items():
            if key.startswith("agn_") and key != "agn_log_lbol" and key not in agn_kwargs:
                agn_kwargs[key] = jnp.asarray(val)
        # Composable-AGN block selectors are static Python strings (not
        # traced JAX values). They live on the config so the runner can
        # close over them at trace-build time and pick the right per-stage
        # callable. For non-composable AGN models the registered function
        # absorbs them via ``**kwargs`` and they have no effect.
        agn_kwargs["agn_disc_block"] = self.config.agn_disc_block
        agn_kwargs["agn_nlr_block"] = self.config.agn_nlr_block
        agn_kwargs["agn_blr_block"] = self.config.agn_blr_block
        agn_kwargs["agn_feii_block"] = self.config.agn_feii_block
        agn_kwargs["agn_torus_block"] = self.config.agn_torus_block
        agn_kwargs["agn_attenuation_block"] = self.config.agn_attenuation_block
        agn_kwargs["agn_norm"] = self.config.agn_norm
        if skirtor_template is not None:
            agn_kwargs["_template"] = skirtor_template

        # Call AGN function. For composable models, request L_2500_intrinsic
        # and L_4400_intrinsic. For monolithic models, both default to 0.0
        # (X-ray falls back to L_bol BC or SKIRTOR's published L_2500_30deg;
        # radio falls back to L_bol bolometric correction).
        if self.config.model == "composable":
            L_agn, L_2500_intrinsic, L_4400_intrinsic = agn_fn(
                wave, agn_log_lbol=agn_log_lbol, return_l2500=True, **agn_kwargs
            )
        else:
            L_agn = agn_fn(wave, agn_log_lbol=agn_log_lbol, **agn_kwargs)
            L_2500_intrinsic = jnp.asarray(0.0)
            L_4400_intrinsic = jnp.asarray(0.0)

        # Filter-integrate L_agn through the cached filter
        # passbands and publish ``agn_phot_lnu_precomp`` so predict_via_precomp
        # can include the AGN contribution in the LUT sum.
        derived_overrides = dict(
            L_agn_bol=L_agn_bol,
            sed_agn=L_agn,
            L_2500_intrinsic=L_2500_intrinsic,
            L_4400_intrinsic=L_4400_intrinsic,
            # X-CIGALE tilts the corona with the AGN viewing angle
            # (yang20.py: cosi = cos(agn.i) for SKIRTOR, sin(psy) for
            # Fritz); publish cos(i) so the X-ray block shares this
            # sightline at every inclination, not only the 30° anchor
            # (#980). The shared ``agn_cos_inc`` knob wins; the
            # AGNfitter-style SKIRTOR block carries degrees instead;
            # models with neither publish the Yang+2020 anchor
            # (anisotropy factor exactly 1).
            agn_cos_inc=(
                jnp.asarray(params["agn_cos_inc"])
                if "agn_cos_inc" in params
                else jnp.cos(jnp.deg2rad(jnp.asarray(params["agn_incl_skirtor"])))
                if "agn_incl_skirtor" in params
                else jnp.asarray(_XRAY_COS_INC_REF_30DEG)
            ),
        )
        if (
            self._state is not None
            and self._state.filter_waves is not None
            and self._state.filter_trans is not None
        ):
            from tengri.observation.photometry import lnu_filter_integral

            z = jnp.asarray(require_redshift(params, "components.agn.component.apply"))
            # Filter-integrate L_agn directly via ``lnu_filter_integral``
            # (ADR-0016, #398.e). Replaces the previous
            # ``compute_flux_density(..., dl_cm=1) × inv_cosmology`` dance
            # that applied and immediately undid the (1+z)/(4π d_L²)
            # dimming. The new helper returns the bare filter-integrated
            # rest-frame L_ν — matching the publish convention of
            # ``stellar_phot_lnu_precomp``.
            agn_phot_lnu_precomp = jnp.asarray(
                [
                    lnu_filter_integral(L_agn, state.wave, fw, ft, redshift=z)
                    for fw, ft in zip(
                        self._state.filter_waves,
                        self._state.filter_trans,
                        strict=False,
                    )
                ]
            )
            derived_overrides["agn_phot_lnu_precomp"] = agn_phot_lnu_precomp
            # The REST band (#1148). ``phot_rest_fnu`` is the SED reprojected at
            # z=0, so the filter sits in the REST frame and samples the rest SED at
            # its own pivot — the SAME integral with redshift=0, not the observed-band
            # value reused. Reusing it is what made the LUT report a different
            # physical quantity from the exact path (769 % in des_g at z=0.5).
            derived_overrides["agn_restband_lnu_precomp"] = jnp.asarray(
                [
                    lnu_filter_integral(L_agn, state.wave, fw, ft, redshift=0.0)
                    for fw, ft in zip(
                        self._state.filter_waves,
                        self._state.filter_trans,
                        strict=False,
                    )
                ]
            )

        # Spectrum LUT family (SpectrumPrecomp): a spectrum pixel is a single
        # wavelength, so point-sampling the rest-frame AGN SED at the pixel
        # wavelengths is exact (mirrors radio/X-ray ``_emit(spec_eff)`` and the
        # dust-IR ``jnp.interp`` projection). Without this, ``predict_spectrum``
        # under ``approx=SpectrumPrecomp()`` silently dropped the AGN
        # contribution — ``predict_spectrum_via_precomp`` only sums the
        # ``*_spec_lnu_precomp`` families that components actually publish, and
        # AGN previously published only the photometry family.
        spec_eff = state.derived.get("spec_eff_waves")
        if spec_eff is not None:
            derived_overrides["agn_spec_lnu_precomp"] = jnp.interp(spec_eff, state.wave, L_agn)

        return state.add_intrinsic(L_agn).with_(
            derived=state.derived.with_(**derived_overrides),
        )


# Register in the unified component dispatch table so build_components resolves
# the AGN component via _resolve_registry_component (single dispatch, #846)
# instead of importing the class directly. AGN is a composite — its config
# selects the disc/torus/nlr/blr/feii/atten sub-blocks (ADR-0018) — but its
# top-level dispatch is a single registered component, exactly like the other
# composite domain (nebular).
from tengri.components.sed_model_component import _REGISTRY

_REGISTRY["agn"] = AGNSEDComponent
