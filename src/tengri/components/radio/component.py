# SPDX-License-Identifier: BSD-3-Clause
"""RadioSEDComponent: SEDComponent adapter around radio physics.

Radio synchrotron and free-free emission from AGN and star formation.
Implements the SEDComponent protocol over the radio bands (3 mm to 30 cm).
Supports multiple AGN radio models (power-law and double power-law with
aging cutoff) and star-formation-driven thermal emission.

AGN-radio model selection
-------------------------
The AGN radio component is selected via
:attr:`RadioSEDComponentConfig.agn_radio_model`:

- ``"powerlaw"`` (default) — single power-law (:func:`radio_total`).
  Backwards-compatible default; behavior bit-identical to pre-aging
  releases.
- ``"dpl"`` — AGNfitter-rx broken double power-law with phenomenological
  ``exp(-nu/nu_cut)`` aging cutoff (:func:`radio_total_dpl`,
  Martinez-Ramirez+2024 Eq. 9-10). Uses ``radio_alpha_thin``,
  ``radio_alpha_thick``, ``radio_log_nu_t``, ``radio_log_nu_cut``.

Physical synchrotron-aging kernels (Jaffe & Perola 1973;
Kardashev/Pacholczyk; Tribble 1993) — ``"JP"``, ``"KP"``, ``"tribble"``
— are not yet implemented. Selecting them raises :class:`ValueError` at
construction. The physics + precomputed pitch-angle integrals (validated
against BRATS, Harwood+2013) land together in a follow-up PR alongside
the two free parameters they consume (``radio_alpha_inj``,
``radio_log_nu_break``).

Cross-component reads
---------------------
Radio depends on quantities owned by other components:

- ``L_ir`` (erg/s) — produced by the dust component as the integrated
  absorbed luminosity. Read from ``state.derived["L_ir"]`` with a
  fallback to 0.0 when no dust component has run yet.
- ``L_agn_bol`` (erg/s) — produced by the AGN component. Read from
  ``state.derived["L_agn_bol"]`` with a fallback to 0.0.
- ``log_mstar`` (log10 M_⊙) — produced by the stellar component. Read
  from ``state.derived["log_mstar"]`` with a fallback to 10.0.
- ``redshift`` — bare parameter from :data:`BARE_NAME_ALLOWLIST`.

This is the canonical pattern documented in
:class:`tengri.protocols.ForwardState`'s "Cross-component reads" section:
**published derived quantity + documented fallback**, not a free
parameter snooped from another component's namespace.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar

import jax.numpy as jnp

from tengri.components._term_response import term_band_response as _term_band_response
from tengri.components.radio._params import PARAMS as _RADIO_PARAMS
from tengri.components.radio.radio import (
    radio_freefree,
    radio_total_dpl_terms,
    radio_total_terms,
)
from tengri.components.template_threading import TemplateThreading
from tengri.parameters.resolve import require_redshift
from tengri.protocols.component import (
    DerivedKey,
    ForwardState,
    ParamDeclaration,
    SEDComponentConfig,
    SEDComponentState,
)

__all__ = ["RadioSEDComponent", "RadioSEDComponentConfig"]

# Mode strings for the AGN radio sub-model. Kept as a module-level
# constant so tests and downstream code can import it without
# instantiating the dataclass.
#
# - ``"none"`` (new 2026-06) — AGN radio turned off; SF component only.
# - ``"powerlaw"`` (default) — single power-law AGN radio
# - ``"dpl"`` — double power-law AGN radio with aging cutoff
#
# JP / KP / tribble (Jaffe & Perola 1973; Kardashev/Pacholczyk; Tribble
# 1993) are NOT in this tuple — they require precomputed pitch-angle
# integrals validated against BRATS (Harwood+2013). When that physics
# lands, the kernel names and their two free parameters (radio_alpha_inj,
# radio_log_nu_break) get added together in the same PR.
AGN_RADIO_MODELS: tuple[str, ...] = ("none", "powerlaw", "dpl")

# Mode strings for the star-formation radio sub-model, the sibling axis to
# :data:`AGN_RADIO_MODELS`. Named here rather than spelled inline at each
# check so the grammar validator (``parameters.groups._translate_radio``) and
# the discovery menu (``registry.list_radio_blocks``) read the *same* tuple —
# a hand-copied second list is how the dust menu and the radio error message
# both drifted out of agreement with what the builder actually accepts.
#
# - ``"none"`` — SF synchrotron turned off; AGN radio only.
# - ``"bell2003"`` (default) — fixed-q FIR-radio correlation.
# - ``"delvecchio2021"`` — mass- and z-dependent FIRRC at 1.4 GHz.
# - ``"mccheyne2022"`` — mass- and z-dependent FIRRC at 150 MHz.
SF_RADIO_MODELS: tuple[str, ...] = ("none", "bell2003", "delvecchio2021", "mccheyne2022")


@dataclass(frozen=True)
class RadioSEDComponentConfig(SEDComponentConfig):
    r"""Frozen knobs for :class:`RadioSEDComponent`.

    Attributes
    ----------
    name : str
        Diagnostic identifier. Default ``"radio"``.
    sfr_mode : str
        Star-formation synchrotron mode. One of
        ``{"none", "bell2003", "delvecchio2021", "mccheyne2022"}``. The
        ``"none"`` mode turns off the SF component entirely (pure AGN radio).
        Default ``"bell2003"``.
    include_freefree : bool
        Add Murphy+2011 thermal free-free component. Default ``True``
        (matches :func:`radio_total`'s default).
    agn_radio_model : str
        AGN radio sub-model. One of :data:`AGN_RADIO_MODELS` —
        ``{"none", "powerlaw", "dpl"}``. The ``"none"`` mode disables
        the AGN radio component (SF synchrotron + optional free-free only).
        Default ``"powerlaw"`` preserves the pre-aging-cutoff behavior
        bit-identically. Physical-aging kernels ``"JP"``, ``"KP"``,
        ``"tribble"`` are reserved names rejected at construction with a
        :class:`ValueError`; the physics lands in a follow-up PR.
    """

    name: str = "radio"
    sfr_mode: str = "bell2003"
    include_freefree: bool = True
    agn_radio_model: str = "powerlaw"

    def __post_init__(self) -> None:
        if self.agn_radio_model not in AGN_RADIO_MODELS:
            raise ValueError(
                f"Unknown agn_radio_model {self.agn_radio_model!r}. "
                f"Choose one of {AGN_RADIO_MODELS}."
            )
        # ``sfr_mode`` went unchecked here while its AGN sibling was validated.
        # A typo did still raise, but only later and further away: inside
        # ``radio._dispatch_sfr`` during a forward pass, naming a function
        # argument rather than the config field the user actually set. Checking
        # at construction keeps both radio axes failing at the same boundary.
        if self.sfr_mode not in SF_RADIO_MODELS:
            raise ValueError(
                f"Unknown sfr_mode {self.sfr_mode!r}. Choose one of {SF_RADIO_MODELS}."
            )


@dataclass(frozen=True)
class RadioSEDComponentState(SEDComponentState):
    r"""Radio has no precomputed tensors — typed marker only."""

    name: str = "radio"


@dataclass(frozen=True)
class RadioSEDComponent(TemplateThreading):
    r"""SEDComponent adapter around the radio physics module.

    Notes
    -----
    **JIT-compatible**: yes for every model in :data:`AGN_RADIO_MODELS`.

    **Additive**: writes ``sed_intrinsic = sed_intrinsic + L_radio(λ)``.
    Initializes ``sed_intrinsic`` from zeros if upstream did not.
    """

    config: RadioSEDComponentConfig = field(default_factory=RadioSEDComponentConfig)
    name: str = "radio"
    parameter_prefix: str = "radio_"

    def citations(self) -> tuple[str, ...]:
        """No structurally-mandatory paper for this wrapper; the FIR-radio
        correlation citations are config-driven via
        :mod:`tengri.citations.associations`."""
        return ()

    def declared_parameters(self) -> list[ParamDeclaration]:
        r"""Free parameters this component owns.

        Returns the canonical :data:`PARAMS` tuple from
        ``tengri.components.radio._params``. The legacy
        ``_RADIO_PARAMS`` bucket in ``tengri.parameters._builders``
        is derived from the same tuple, so the two registration paths
        are guaranteed to agree.

        DPL parameters (``radio_alpha_thin``, ``radio_alpha_thick``,
        ``radio_log_nu_t``, ``radio_log_nu_cut``) are declared but
        ``Fixed`` by default, so the component is a no-op extension when
        ``agn_radio_model="powerlaw"``.
        """
        return list(_RADIO_PARAMS)

    def outputs(self) -> tuple[DerivedKey, ...]:
        """Cross-component derived keys this radio component publishes.

        See :func:`tengri.forward.orchestrator.validate_pipeline`.

        """
        return (
            DerivedKey(
                "sed_radio",
                "erg/s/Hz",
                "Radio luminosity contribution on pipeline wave grid",
            ),
        )

    def optional_inputs(self) -> tuple[DerivedKey, ...]:
        """Cross-component derived keys radio reads *opportunistically*.

        Read from ``state.derived`` with documented fallbacks so radio
        remains usable in pipelines that omit the upstream publisher
        (photometry-only fits without dust or AGN). The validator does
        NOT require an upstream publisher for these, but it WILL check
        that if one is present, its units match. Catches a future
        publisher rename or unit drift. Phase B of #21 — see ADR-0004.
        """
        return (
            DerivedKey("L_ir", "erg/s", "Read from dust if present; falls back to 0.0"),
            DerivedKey("L_agn_bol", "erg/s", "Read from AGN if present; falls back to 0.0"),
            DerivedKey(
                "L_4400_intrinsic",
                "erg/s/Hz",
                "Read from AGN if present; drives radio loudness normalization",
            ),
            DerivedKey("log_mstar", "dex", "Read from stellar if present; falls back to 10.0"),
        )

    #: Cross-component scalars this emitter reads off ``state.derived``, with two
    #: deliberately distant probe draws. The build-time band-response builder
    #: (``tengri.SEDModel._additive_term_band_response``) evaluates
    #: :meth:`emission_terms` at both and requires every term to come back
    #: *proportional*: a term whose spectral **shape** — not merely its amplitude —
    #: responds to a runtime input is not rank-1, so no constant band response
    #: exists for it and the emitter drops to the dense per-call filter integral.
    #: Verifying the property beats declaring it (#1107: BOSA's template shape
    #: tracked its own luminosity and returned fluxes 13 % wrong, silently).
    #:
    #: Every value is nonzero on purpose. A term that stays identically zero across
    #: both draws is zero for a *structural* reason — ``include_freefree=False``, or
    #: ``radio_loudness = 0`` (the default, i.e. radio-quiet) — never because a probe
    #: happened to zero it out.
    EMITTER_PROBE_INPUTS: ClassVar[tuple[dict[str, float], dict[str, float]]] = (
        {
            "L_ir": 1.0e44,
            "L_agn_bol": 1.0e45,
            "L_4400_intrinsic": 1.0e28,
            "log_mstar": 10.0,
        },
        {
            "L_ir": 8.0e45,
            "L_agn_bol": 6.0e46,
            "L_4400_intrinsic": 5.0e30,
            "log_mstar": 11.5,
        },
    )

    def emitter_inputs(self, derived: Mapping[str, Any]) -> dict[str, jnp.ndarray]:
        """Read this emitter's cross-component scalars off the derived state.

        Parameters
        ----------
        derived : mapping
            ``state.derived``. Missing keys take the documented fallbacks — a model
            with no dust block publishes no ``L_ir``, and radio must still build.

        Returns
        -------
        dict
            Keyword arguments for :meth:`emission_terms`: ``L_ir`` [erg/s],
            ``L_agn_bol`` [erg/s], ``L_4400_intrinsic`` [erg/s/Hz], ``log_mstar``
            [dex Msun]. When the dust / AGN components publish the float32-safe
            log companions ``log_L_ir`` / ``log_L_agn_bol`` (#1206), they are
            forwarded too so :meth:`emission_terms` can bypass the ~1e43 / ~1e46
            linear luminosities that overflow float32.
        """
        inputs = {
            "L_ir": jnp.asarray(derived.get("L_ir", 0.0)),
            "L_agn_bol": jnp.asarray(derived.get("L_agn_bol", 0.0)),
            "L_4400_intrinsic": jnp.asarray(derived.get("L_4400_intrinsic", 0.0)),
            "log_mstar": jnp.asarray(derived.get("log_mstar", 10.0)),
        }
        log_L_ir = derived.get("log_L_ir")
        if log_L_ir is not None:
            inputs["log_L_ir"] = jnp.asarray(log_L_ir)
        log_L_agn_bol = derived.get("log_L_agn_bol")
        if log_L_agn_bol is not None:
            inputs["log_L_agn_bol"] = jnp.asarray(log_L_agn_bol)
        return inputs

    def emission_terms(
        self,
        params: Mapping[str, jnp.ndarray],
        wave: jnp.ndarray,
        *,
        L_ir: jnp.ndarray,
        L_agn_bol: jnp.ndarray,
        L_4400_intrinsic: jnp.ndarray,
        log_mstar: jnp.ndarray,
        log_L_ir: jnp.ndarray | None = None,
        log_L_agn_bol: jnp.ndarray | None = None,
    ) -> dict[str, jnp.ndarray]:
        r"""The additive terms of the radio SED, unsummed.

        Single source of truth: :meth:`apply` sums these to build the SED, and the
        build-time band-response precompute integrates *these same terms* through the
        filters. Two paths, one definition — so they cannot drift, which is exactly
        how #1107 nearly shipped a band response built from the wrong parameters.

        Parameters
        ----------
        params : mapping
            Full (un-sliced) parameter dict; reads the ``radio_*`` keys and the bare
            ``redshift``.
        wave : array_like, shape (n_wave,)
            Rest-frame wavelength grid [Angstrom]. Any grid: the full model grid in
            :meth:`apply`, or the per-term reference wavelengths under the LUT.
        L_ir : array_like, scalar
            Dust-reradiated IR luminosity [erg/s], drives the SF synchrotron and
            free-free amplitudes via the FIR-radio correlation.
        L_agn_bol : array_like, scalar
            AGN bolometric luminosity [erg/s].
        L_4400_intrinsic : array_like, scalar
            Un-reddened AGN B-band luminosity [erg/s/Hz], the radio-loudness reference.
        log_mstar : array_like, scalar
            log10 stellar mass [dex Msun]; used by the mass-evolving FIRRC modes.

        Returns
        -------
        dict of ndarray, each shape (n_wave,)
            ``{"sf", "ff", "agn"}`` — star-forming synchrotron, free-free, and the AGN
            jet [erg/s/Hz]. ``"agn"`` is zeros when ``agn_radio_model == "none"``, and
            ``"ff"`` is zeros when ``include_freefree`` is off.

        Notes
        -----
        **JIT/grad/vmap-safe.** Pure ``jnp``; the model branch is a static config read.

        Each term is *rank-1* in wavelength — a scalar amplitude times a spectral shape
        set only by the shape parameters (``alpha_sf``, ``alpha_agn``/``alpha_thin``/
        ``alpha_thick``, ``log_nu_t``, ``log_nu_cut``, ``T_e``, ``alpha_ff``). The
        *total* is not: three power laws of different index do not share a shape. That
        is why the terms are exposed separately — the band integral factorizes per term
        and not on the sum (#1109).
        """
        model = self.config.agn_radio_model
        z = jnp.asarray(require_redshift(params, "components.radio.component.emission_terms"))

        # Float32 routing (#1206): only when the forward grid is float32 do we
        # bypass the linear ``L_ir`` (~1e43) / ``L_agn_bol`` (~1e46) — which
        # overflow float32 max (3.4e38) — by handing the log companions to the
        # radio kernels (they form the representable ~1e28 radio luminosity via
        # ``pow10``). In float64 the logs are withheld so the exact linear path
        # is bit-for-bit unchanged. Both require the producer to have published
        # the log (dust → ``log_L_ir``, AGN → ``log_L_agn_bol``).
        _use_log = wave.dtype == jnp.float32
        _log_L_ir = log_L_ir if _use_log else None
        _log_L_agn = log_L_agn_bol if _use_log else None

        # FIRRC evolution coefficients for the evolving SF-radio models. The
        # active ``sfr_mode`` (static config) selects which model-specific
        # triplet is consumed; bell2003 / none ignore them (pass None → the
        # SF functions fall through to their literature defaults / zeros).
        firrc_q0, firrc_mass_slope, firrc_z_slope = self._firrc_overrides(params)

        if model == "none":
            from tengri.components.radio.radio import _dispatch_sfr

            sf = _dispatch_sfr(
                wave,
                L_ir=L_ir,
                sfr_mode=self.config.sfr_mode,
                q_ir=jnp.asarray(params["radio_q_ir"]),
                alpha_sf=jnp.asarray(params["radio_alpha_sf"]),
                log_mstar=log_mstar,
                redshift=z,
                q0=firrc_q0,
                mass_slope=firrc_mass_slope,
                z_slope=firrc_z_slope,
                apply_suppression=True,
                log_L_ir=_log_L_ir,
            )
            ff = (
                radio_freefree(
                    wave,
                    L_ir,
                    jnp.asarray(params["radio_T_e"]),
                    jnp.asarray(params["radio_alpha_ff"]),
                    log_L_ir=_log_L_ir,
                )
                if self.config.include_freefree
                else jnp.zeros_like(wave)
            )
            return {"sf": sf, "ff": ff, "agn": jnp.zeros_like(wave)}

        if model == "powerlaw":
            return radio_total_terms(
                wave,
                L_ir=L_ir,
                L_agn_bol=L_agn_bol,
                q_ir=jnp.asarray(params["radio_q_ir"]),
                alpha_sf=jnp.asarray(params["radio_alpha_sf"]),
                radio_loudness=jnp.asarray(params["radio_loudness"]),
                alpha_agn=jnp.asarray(params["radio_alpha_agn"]),
                sfr_mode=self.config.sfr_mode,
                log_mstar=log_mstar,
                redshift=z,
                q0=firrc_q0,
                mass_slope=firrc_mass_slope,
                z_slope=firrc_z_slope,
                include_freefree=self.config.include_freefree,
                T_e=jnp.asarray(params["radio_T_e"]),
                alpha_ff=jnp.asarray(params["radio_alpha_ff"]),
                l_bband=L_4400_intrinsic,
                log_L_ir=_log_L_ir,
                log_L_agn_bol=_log_L_agn,
            )

        # model == "dpl"
        return radio_total_dpl_terms(
            wave,
            L_ir=L_ir,
            L_agn_bol=L_agn_bol,
            q_ir=jnp.asarray(params["radio_q_ir"]),
            alpha_sf=jnp.asarray(params["radio_alpha_sf"]),
            radio_loudness=jnp.asarray(params["radio_loudness"]),
            alpha1=jnp.asarray(params["radio_alpha_thin"]),
            alpha2=jnp.asarray(params["radio_alpha_thick"]),
            log_nu_t=jnp.asarray(params["radio_log_nu_t"]),
            log_nu_cut=jnp.asarray(params["radio_log_nu_cut"]),
            sfr_mode=self.config.sfr_mode,
            log_mstar=log_mstar,
            redshift=z,
            q0=firrc_q0,
            mass_slope=firrc_mass_slope,
            z_slope=firrc_z_slope,
            include_freefree=self.config.include_freefree,
            T_e=jnp.asarray(params["radio_T_e"]),
            alpha_ff=jnp.asarray(params["radio_alpha_ff"]),
            l_bband=L_4400_intrinsic,
            log_L_ir=_log_L_ir,
            log_L_agn_bol=_log_L_agn,
        )

    def precompute(
        self,
        ssp_data: Any | None = None,
        wave_grid: jnp.ndarray | None = None,
        approx: Mapping[str, bool] | None = None,
        filters: tuple[tuple[jnp.ndarray, jnp.ndarray], ...] | None = None,
    ) -> RadioSEDComponentState:
        r"""No-op precompute. Radio is a closed-form function of (λ, params)."""
        del ssp_data, wave_grid, filters
        return RadioSEDComponentState(name=self.name)

    def _firrc_overrides(
        self, params: Mapping[str, jnp.ndarray]
    ) -> tuple[jnp.ndarray | None, jnp.ndarray | None, jnp.ndarray | None]:
        r"""Resolve the FIRRC ``(q0, mass_slope, z_slope)`` triplet for the
        active ``sfr_mode``.

        Returns the model-specific ``radio_delv_*`` or ``radio_mcch_*``
        parameters when an evolving SF-radio model is selected, or
        ``(None, None, None)`` for ``bell2003`` / ``none`` (the SF functions
        then use their own literature defaults). Selection is on the static
        ``self.config.sfr_mode`` string, so this introduces no traced branch.

        Parameters
        ----------
        params : mapping
            Receives all declared ``radio_*`` keys.

        Returns
        -------
        tuple
            ``(q0, mass_slope, z_slope)`` as JAX arrays, or ``(None, None,
            None)`` when the active mode does not consume them.
        """
        mode = self.config.sfr_mode
        if mode == "delvecchio2021":
            prefix = "radio_delv_"
        elif mode == "mccheyne2022":
            prefix = "radio_mcch_"
        else:
            return None, None, None
        return (
            jnp.asarray(params[f"{prefix}q0"]),
            jnp.asarray(params[f"{prefix}mass_slope"]),
            jnp.asarray(params[f"{prefix}z_slope"]),
        )

    def apply(
        self,
        state: ForwardState,
        params: Mapping[str, jnp.ndarray],
        ssp_data: Any | None = None,
        template_data: Any | None = None,
    ) -> ForwardState:
        r"""Add radio emission to ``state.sed_intrinsic``.

        Dispatches to :func:`radio_total` (powerlaw) or
        :func:`radio_total_dpl` (dpl) based on
        :attr:`RadioSEDComponentConfig.agn_radio_model`.

        Parameters
        ----------
        state : ForwardState
            Must carry rest-frame ``wave`` (Å). If ``sed_intrinsic`` is
            ``None`` it is initialized to zeros of the same shape.
        params : mapping
            Receives ``radio_*`` keys plus the bare ``redshift`` from
            the allowlist. Cross-component scalars (``L_ir``,
            ``L_agn_bol``, ``log_mstar``) are read from
            ``state.derived`` with documented fallbacks.

        Returns
        -------
        ForwardState
            New state with ``sed_intrinsic`` updated.
        """
        # JP / KP / tribble were previously dispatched here with a runtime
        # NotImplementedError. They now fail earlier — at construction —
        # because they are no longer in AGN_RADIO_MODELS. That validation
        # lives in RadioSEDComponentConfig.__post_init__.
        wave = state.wave
        z = jnp.asarray(require_redshift(params, "components.radio.component.apply"))
        inputs = self.emitter_inputs(state.derived)

        def _emit(w):
            t = self.emission_terms(params, w, **inputs)
            return t["sf"] + t["ff"] + t["agn"]

        L_radio = _emit(wave)
        derived_overrides = {"sed_radio": L_radio}

        # Precompute LUT families (#624): radio is additive and unattenuated,
        # summed by predict_via_precomp / predict_spectrum_via_precomp.
        # Spectroscopy: a pixel is a point-sample, so evaluating at the pixel
        # wavelength is exact.
        from tengri.components._band_projection import project_additive_onto_photometry

        filter_eff = state.derived.get("filter_eff_waves")
        if filter_eff is not None:
            band = _term_band_response(template_data, "radio")
            fw_pad = state.derived.get("phot_filter_waves_padded")
            ft_pad = state.derived.get("phot_filter_trans_padded")

            # Compute precomputed photometry if band response is available.
            precomputed = None
            if band is not None:
                # Exact fast path. Radio is a sum of rank-1 terms — SF synchrotron,
                # free-free, AGN jet — each a scalar amplitude times a spectral shape
                # fixed by the (fixed) shape parameters. The filter integral is linear,
                # so the band flux is sum_k A_k * R_kf with R_kf integrated once at
                # build time through the true filter transmission. Recover each A_k
                # from a single-wavelength evaluation rather than the dense grid: that
                # is what lets XLA dead-code-eliminate the full-resolution SED, which
                # is where the whole WavePrecomp speedup lives (#1109).
                ref = self.emission_terms(params, band["lam_ref"], **inputs)
                amps = jnp.stack([t[i] for i, t in enumerate(ref.values())]) / band["S_ref"]
                precomputed = amps @ band["R"]

            # Project onto observed-frame photometric filters.
            derived_overrides["radio_phot_lnu_precomp"] = project_additive_onto_photometry(
                precomputed,
                L_radio,
                wave,
                filter_eff,
                fw_pad,
                ft_pad,
                z,
                fallback_fn=_emit,
            )

            # The REST band (#1148): ``phot_rest_fnu`` projects at z=0, so its
            # filter samples the pivot itself, not pivot/(1+z). Same emission,
            # different wavelengths — reusing the observed-band value here is
            # what made the LUT report a different quantity from the exact path.
            _rb_eff = state.derived.get("filter_restband_eff_waves")
            if _rb_eff is not None:
                derived_overrides["radio_restband_lnu_precomp"] = _emit(_rb_eff)
        spec_eff = state.derived.get("spec_eff_waves")
        if spec_eff is not None:
            derived_overrides["radio_spec_lnu_precomp"] = _emit(spec_eff)

        return state.add_intrinsic(L_radio).with_(
            derived=state.derived.with_(**derived_overrides),
        )


# ─────────────────────────────────────────────────────────────────────
# Radio group property registration (Phase 1B)
# ─────────────────────────────────────────────────────────────────────

_TINY = 1e-30  # Floor for safe division


from tengri.utils.physics_constants import WAVE_1P4GHZ_AA


def _l_1p4ghz_fn(state, params):
    """1.4 GHz radio flux [erg/s/Hz]."""
    derived = state.derived
    if "sed_radio" not in derived:
        return jnp.asarray(jnp.nan)

    L_radio = jnp.asarray(derived["sed_radio"])
    wave = state.wave
    return jnp.interp(WAVE_1P4GHZ_AA, wave, L_radio)


def _l_thermal_fn(state, params):
    """Radio thermal (free-free) luminosity [erg/s/Hz]."""
    from tengri.utils.sed_quantities import compute_l_radio_thermal_from_log_qh

    derived = state.derived
    log_nion = jnp.asarray(derived.get("log_nion", -jnp.inf))
    return compute_l_radio_thermal_from_log_qh(log_nion)


def _l_nonthermal_fn(state, params):
    """Radio non-thermal synchrotron luminosity [erg/s/Hz]."""
    from tengri.utils.sed_quantities import compute_l_radio_thermal_from_log_qh

    derived = state.derived
    if "sed_radio" not in derived:
        return jnp.asarray(jnp.nan)

    L_radio = jnp.asarray(derived["sed_radio"])
    wave = state.wave
    l_1p4ghz = jnp.interp(WAVE_1P4GHZ_AA, wave, L_radio)

    log_nion = jnp.asarray(derived.get("log_nion", -jnp.inf))
    l_thermal = compute_l_radio_thermal_from_log_qh(log_nion)
    return l_1p4ghz - l_thermal


def _q_ir_fn(state, params):
    """Radio-infrared correlation parameter [dimensionless]."""
    from tengri.utils.sed_quantities import compute_q_ir, derived_luminosity_lsun

    derived = state.derived
    if "sed_radio" not in derived:
        return jnp.asarray(jnp.nan)

    L_radio = jnp.asarray(derived["sed_radio"])
    wave = state.wave
    l_1p4ghz = jnp.interp(WAVE_1P4GHZ_AA, wave, L_radio)

    # Same seam as ``l_dust_absorbed``: the linear ``L_ir`` is ~3.6e43 erg/s and
    # is ``inf`` in float32, while q_IR is a dex ratio of order 2 (#1837).
    l_tir_lsun = derived_luminosity_lsun(derived, "L_ir", "log_L_ir")
    return compute_q_ir(l_tir_lsun, l_1p4ghz)


from tengri.forward.properties import Property, register_properties

_RADIO_PROPERTIES = {
    "l_1p4ghz": Property(
        units="erg/s/Hz",
        group="radio",
        doc="1.4 GHz radio flux",
        fn=_l_1p4ghz_fn,
    ),
    "l_thermal": Property(
        units="erg/s/Hz",
        group="radio",
        doc="Radio thermal (free-free) luminosity",
        fn=_l_thermal_fn,
    ),
    "l_nonthermal": Property(
        units="erg/s/Hz",
        group="radio",
        doc="Radio non-thermal synchrotron luminosity",
        fn=_l_nonthermal_fn,
    ),
    "q_ir": Property(
        units="",
        group="radio",
        doc="Radio-infrared correlation parameter",
        fn=_q_ir_fn,
    ),
}

register_properties("radio", _RADIO_PROPERTIES)

del Property, register_properties, _RADIO_PROPERTIES


# Register in the unified component dispatch table so build_components resolves
# the radio component via _resolve_registry_component (single dispatch, #845)
# instead of importing the class directly.
from tengri.components.sed_model_component import _REGISTRY

_REGISTRY["radio"] = RadioSEDComponent
