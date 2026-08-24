# SPDX-License-Identifier: BSD-3-Clause
"""AGNSEDComponent: unified AGN emission (disc + torus + lines + jets/corona).

Dispatches to any registered AGN model (``"multicolor_agn"``,
``"kubota_done_full"``, ``"adaf"``, ``"unified_nlr_blr"``, ``"skirtor"``,
``"silva04"``, ``"cat3d_wind"``, ``"relagn"``, ``"qsogen"``) via
:func:`tengri.components.agn.unified.resolve_agn_model`.

Cross-component reads
---------------------
Nothing required; AGN is a self-contained additive emitter. ``redshift``
is read from :data:`BARE_NAME_ALLOWLIST`.

Cross-component publications
----------------------------

- ``state.derived["L_agn_bol"]`` (scalar, erg/s): bolometric AGN
  luminosity. Consumed by
  :class:`tengri.components.xray.component.XRaySEDComponent` and
  :class:`tengri.components.radio.component.RadioSEDComponent` via their
  documented fallback (``state.derived.get("L_agn_bol", 0.0)``).
- ``state.derived["sed_agn"]``: the AGN SED contribution
  (erg/s/Hz, shape n_wave) for diagnostics.

Architectural notes
-------------------
``agn_torus_frac`` is **never** auto-derived from ``agn_cos_inc`` /
``theta_torus`` (CLAUDE.md gotcha: gradient discontinuity). It is
read directly from ``params`` as an independent free parameter.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import jax.numpy as jnp

from tengri.components.agn._params import PARAMS as _AGN_PARAMS
from tengri.components.agn.blocks._protocol import collect_block_templates
from tengri.components.agn.unified import resolve_agn_model
from tengri.components.template_threading import TemplateThreading
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

#: log10 of the solar luminosity [dex], for folding the AGN bolometric scale
#: into log space (float32 safety, #1206). L_SUN ~3.828e33 erg/s.
_LOG10_L_SUN: float = float(jnp.log10(L_SUN))

#: Reference AGN ``agn_log_lbol`` (= log10(L_bol/L_sun)) at which every block is
#: evaluated for the float32 factoring (#1206). Chosen so L_bol = 1e10 erg/s: low
#: enough that the *squares* of the internal bolometric integrals stay in float32
#: range (``(1e10)**2 = 1e20 << 3.4e38``), yet high enough that the runner's
#: ``max(faceon/L_sun, 1e-30)`` zero-protection floor never engages (faceon/L_sun
#: ~1e-23, seven decades clear). The true 10^agn_log_lbol is re-applied afterward.
_AGN_LBOL_REF: float = 10.0 - _LOG10_L_SUN

#: AGN disc blocks that are NOT yet float32-safe (#1206). Each returns NaN/inf in
#: pure float32 (JAX-Metal) from a distinct grid-dependent overflow (a ``0*inf`` in
#: the block/runner, *not* the L_bol magnitude the shape-class fixes address). This
#: used to name ``forward_dtype="float32"`` as a second way in; that knob casts
#: nothing (#1433), so it cannot reach float32 arithmetic here. See
#: ``docs/dev/float32-tier-b-boundary.md`` §8 and
#: ``tests/regression/precision/test_agn_disc_float32_inventory.py``. The
#: float32-safe discs are ``multicolor``, ``kubota_done``, ``adaf`` (physical,
#: L_bol-dependent shape) and ``powerlaw`` / ``richards2006`` / ``skirtor`` /
#: ``qsogen`` / ``schartmann2005`` / ``adaf_lopez2024`` (shape-invariant).
#: ``grahsp_sbpl`` is blocked on a linear erg/s *parameter* (``agn_grahsp_l5100``
#: is ``inf`` in float32), not a kernel overflow: it needs a log-space parameter.
_NON_FLOAT32_SAFE_DISCS: frozenset[str] = frozenset({"grahsp_sbpl"})


class Float32UnsafeAGNWarning(UserWarning):
    """A non-float32-safe AGN disc block is being evaluated in float32 (#1206)."""


__all__ = [
    "AGNSEDComponent",
    "AGNSEDComponentConfig",
    "Float32UnsafeAGNWarning",
]


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
        ``model == "composable"``, the runner reads them from this config
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
        polar via the legacy face-on proxy), the GRAHSP/AGNfitter-style
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

    Also caches the template libraries of whichever composable blocks the
    recipe selected, so they become Parameter ops rather than baked
    Constants.
    """

    name: str = "agn"
    filter_waves: Any | None = None
    filter_trans: Any | None = None
    skirtor_templates: Any | None = None
    #: ``{"<category>/<name>": pytree}`` for every selected block that
    #: declared a ``template_loader``. See ``collect_block_templates``.
    block_templates: Any | None = None


@dataclass(frozen=True)
class AGNSEDComponent(TemplateThreading):
    """SEDComponent adapter for the unified AGN model registry.

    Notes
    -----
    **JIT-compatible**: yes, :meth:`apply` delegates to a registered
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
        ``tengri.components.agn._params``. The legacy ``_AGN_PARAMS``
        bucket in ``tengri.parameters._builders`` is a derived view
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
            DerivedKey(
                "log_L_agn_bol",
                "dex",
                "log10(L_agn_bol / (erg/s)); float32-safe form "
                "(L_agn_bol ~1e46 overflows float32)",
            ),
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
                "AGN cos(i), the X-ray corona tilts its Yang+2022 anisotropy "
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
                # interpolation closure: arrays thread through jax.jit as a
                # runtime input; a closure cannot (#1198).
                skirtor_templates = _load_skirtor_default_grid()
            except Exception:
                # If SKIRTOR template loading fails (file not found),
                # gracefully continue without threading. The apply() path
                # will fall back to lazy loading.
                pass

        # Composable-block template libraries. Driven by the block recipe, NOT
        # by ``config.model``: ``composable`` is the default in the build
        # grammar, so a gate on ``model == "skirtor"`` publishes nothing for
        # the recommended surface and every torus library bakes (#1383).
        block_templates = collect_block_templates(self.block_recipe()) or None

        return AGNSEDComponentState(
            name=self.name,
            filter_waves=filter_waves,
            filter_trans=filter_trans,
            skirtor_templates=skirtor_templates,
            block_templates=block_templates,
        )

    def block_recipe(self) -> dict[str, str]:
        """Map each pipeline stage to the block name this config selected.

        Returns
        -------
        dict
            ``{"disc": ..., "nlr": ..., "blr": ..., "feii": ..., "torus": ...,
            "attenuation": ...}``.
        """
        return {
            "disc": self.config.agn_disc_block,
            "nlr": self.config.agn_nlr_block,
            "blr": self.config.agn_blr_block,
            "feii": self.config.agn_feii_block,
            "torus": self.config.agn_torus_block,
            "attenuation": self.config.agn_attenuation_block,
        }

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

        import jax as _jax

        from tengri.utils.scale import pow10 as _pow10

        agn_log_lbol = jnp.asarray(params["agn_log_lbol"])

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
        #
        # Every luminosity here (``L_absorbed`` ~1e43, ``L_agn_bol`` ~1e46) is
        # ``inf`` in pure float32, so the ratio ``agn_power / L_agn_bol`` is
        # formed in LOG space (from the float32-safe ``log_L_ir``), never as
        # inf/inf. ``agn_ir_frac == 0`` leaves the coupling inert (the user
        # torus_frac), with the log branch still finite so its unused-branch
        # gradient cannot leak (#1206).
        agn_ir_frac = jnp.asarray(params.get("agn_ir_frac", 0.0))
        # Avoid divide-by-zero / negative leak when fracAGN ≥ 1.
        _one_minus_frac = jnp.maximum(1.0 - agn_ir_frac, 1e-6)
        agn_torus_frac_user = jnp.asarray(params.get("agn_torus_frac", 0.5))
        _log_L_absorbed = state.derived.get("log_L_ir")
        if _log_L_absorbed is None:
            _log_L_absorbed = jnp.log10(
                jnp.maximum(jnp.asarray(state.derived.get("L_absorbed", 0.0)), 1e-30)
            )
        # log10(agn_power / L_agn_bol)
        #   = log10(L_absorbed) + log10(frac/(1−frac)) − agn_log_lbol − log10(L_sun)
        _log_torus_frac = (
            _log_L_absorbed
            + jnp.log10(jnp.maximum(agn_ir_frac, 1e-30))
            - jnp.log10(_one_minus_frac)
            - agn_log_lbol
            - _LOG10_L_SUN
        )
        agn_torus_frac_effective = jnp.where(
            agn_ir_frac > 0.0, _pow10(_log_torus_frac), agn_torus_frac_user
        )
        # Published byproduct only [erg/s]; ~1e46 and ``inf`` in float32.
        # stop_gradient so that inf cannot poison the reverse pass via 0*inf;
        # consumers needing it linearly run in float64.
        L_agn_bol = _jax.lax.stop_gradient(_pow10(agn_log_lbol + _LOG10_L_SUN))

        # Resolve the AGN model from the registry. resolve_agn_model is
        # a factory-time lookup but the returned callable is pure JAX
        # so it folds into a JIT trace cleanly. If template_data is
        # provided, extract AGN SKIRTOR template and thread it.
        agn_fn = resolve_agn_model(self.config.model)

        # Thread the SKIRTOR template as a JIT runtime input
        skirtor_template = None
        block_templates = None
        if template_data is not None and isinstance(template_data, dict):
            agn_data = template_data.get("agn")
            if agn_data is not None and isinstance(agn_data, dict):
                skirtor_template = agn_data.get("skirtor")
                block_templates = agn_data.get("blocks")
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
        # Per-block template libraries for the composable runner. The runner
        # reads this under the name ``template_state`` and hands each stage
        # its OWN family's bundle; without it every template-backed block
        # loads its grid at trace time and bakes it in.
        if block_templates:
            agn_kwargs["template_state"] = block_templates

        # Call AGN function. For composable models, request L_2500_intrinsic
        # and L_4400_intrinsic. For monolithic models, both default to 0.0
        # (X-ray falls back to L_bol BC or SKIRTOR's published L_2500_30deg;
        # radio falls back to L_bol bolometric correction).
        #
        # Float32 boundary (#1206). Evaluating every AGN block at the true
        # ``agn_log_lbol`` overflows float32: the CIGALE-joint disc renorm forms
        # ``trapz(L_torus)`` and ``trapz(L_disc)`` ~ L_bol (~1e44 erg/s, past
        # float32 max 3.4e38) at ``blocks/runner.py``. So in **float32 only** we
        # evaluate every block at a low reference L_bol (``_AGN_LBOL_REF`` →
        # 1e10 erg/s, comfortably in range) and re-apply the
        # ``10^(agn_log_lbol − _AGN_LBOL_REF)`` scale in log space via
        # apply_log10_scale. That output factoring is EXACT for shape-invariant
        # blocks, the SKIRTOR torus template and the power-law disc scale
        # linearly with L_bol: and, since the ``agn_log_lbol_shape`` hand-off
        # below, exact for the multicolor disc too: the true L_bol drives the
        # temperature and geometry while only the MAGNITUDE is factored, and the
        # magnitude is linear by construction. Measured in float64, where float32
        # round-off cannot mask a shape error: max relative deviation from a
        # direct evaluation at the true L_bol is 2.2e-16 (one ulp) at
        # log L_bol = 11-14. Without the shape hand-off the same comparison is
        # off by 100% at log L_bol = 11 and 3685% at 14, that is what this
        # comment used to describe. In **float64** we evaluate at the true
        # ``agn_log_lbol`` and
        # publish the block outputs unchanged, the reference implementation,
        # bit-for-bit identical to pre-#1206 main for every disc type.
        from tengri.utils.scale import apply_log10_scale

        _use_ref = wave.dtype == jnp.float32
        if _use_ref and self.config.agn_disc_block in _NON_FLOAT32_SAFE_DISCS:
            # Warns once per trace (Python side-effect at trace time). These discs
            # produce NaN/inf in float32; the fit will silently corrupt. #1206.
            warnings.warn(
                f"AGN disc_block={self.config.agn_disc_block!r} is not float32-safe "
                "(#1206): it returns NaN/inf in pure float32 (JAX-Metal). "
                "For float32 use a supported disc: "
                "'multicolor', 'kubota_done', 'adaf' (physical), or 'powerlaw' / "
                "'richards2006' / 'skirtor' / 'qsogen' / 'schartmann2005' "
                "(shape-invariant), or run in float64. See "
                "docs/dev/float32-tier-b-boundary.md §8.",
                Float32UnsafeAGNWarning,
                stacklevel=2,
            )
        _lbol_eval = (
            jnp.full_like(jnp.asarray(agn_log_lbol, dtype=wave.dtype), _AGN_LBOL_REF)
            if _use_ref
            else agn_log_lbol
        )
        if _use_ref:
            # Multicolor-disc shape depends on L_bol (temperature), so evaluating
            # the whole runner at the reference L_bol would give the WRONG disc
            # shape. Hand the disc its TRUE L_bol for the temperature/geometry
            # (``agn_log_lbol_shape``) while everything else: including the disc's
            # output MAGNITUDE: stays on the reference so the runner's L_lambda
            # arithmetic stays in float32 range. Shape-invariant blocks (torus
            # template, power-law disc) ignore the kwarg. The disc's internals are
            # float32-hardened (log-space) so the true-L_bol temperature computes
            # without overflow. (#1206)
            agn_kwargs = {**agn_kwargs, "agn_log_lbol_shape": agn_log_lbol}
        if self.config.model == "composable":
            L_agn_unit, L_2500_unit, L_4400_unit = agn_fn(
                wave, agn_log_lbol=_lbol_eval, return_l2500=True, **agn_kwargs
            )
        else:
            L_agn_unit = agn_fn(wave, agn_log_lbol=_lbol_eval, **agn_kwargs)
            L_2500_unit = jnp.asarray(0.0)
            L_4400_unit = jnp.asarray(0.0)
        if _use_ref:
            _offset = agn_log_lbol - _AGN_LBOL_REF
            L_agn = apply_log10_scale(L_agn_unit, _offset)
            L_2500_intrinsic = apply_log10_scale(L_2500_unit, _offset)
            L_4400_intrinsic = apply_log10_scale(L_4400_unit, _offset)
        else:
            L_agn = L_agn_unit
            L_2500_intrinsic = L_2500_unit
            L_4400_intrinsic = L_4400_unit

        # Filter-integrate L_agn through the cached filter
        # passbands and publish ``agn_phot_lnu_precomp`` so predict_via_precomp
        # can include the AGN contribution in the LUT sum.
        derived_overrides = dict(
            L_agn_bol=L_agn_bol,
            log_L_agn_bol=(agn_log_lbol + _LOG10_L_SUN),
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
            # rest-frame L_ν: matching the publish convention of
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
            # its own pivot, the SAME integral with redshift=0, not the observed-band
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
        # contribution: ``predict_spectrum_via_precomp`` only sums the
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
# instead of importing the class directly. AGN is a composite: its config
# selects the disc/torus/nlr/blr/feii/atten sub-blocks (ADR-0018), but its
# top-level dispatch is a single registered component, exactly like the other
# composite domain (nebular).
from tengri.components.sed_model_component import _REGISTRY

_REGISTRY["agn"] = AGNSEDComponent
