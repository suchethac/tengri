# SPDX-License-Identifier: BSD-3-Clause
"""RadioSEDComponent: SEDComponent adapter around radio physics.

Phase II-1 first-adapter exercise. The physics in
:mod:`tengri.components.radio.radio` is unchanged; this is a thin
wrapper that satisfies :class:`tengri.core.SEDComponent` so the
orchestrator can run radio alongside other Phase-II adapters.

AGN-radio model selection
-------------------------
The AGN radio component is selected via
:attr:`RadioSEDComponentConfig.agn_radio_model`:

- ``"powerlaw"`` (default) — single power-law (:func:`radio_total`).
  Backwards-compatible default; behaviour bit-identical to pre-aging
  releases.
- ``"dpl"`` — AGNfitter-rx broken double power-law with phenomenological
  ``exp(-nu/nu_cut)`` aging cutoff (:func:`radio_total_dpl`,
  Martinez-Ramirez+2024 Eq. 9-10). Uses ``radio_alpha_thin``,
  ``radio_alpha_thick``, ``radio_log_nu_t``, ``radio_log_nu_cut``.
- ``"JP"``, ``"KP"``, ``"tribble"`` — physical synchrotron-aging
  kernels from Jaffe & Perola (1973), Kardashev/Pacholczyk, and Tribble
  (1993). Validated against BRATS (Harwood et al. 2013, 2015).
  **Currently stubs**: raise :class:`NotImplementedError`. Physics +
  precomputed kernel tables land in a follow-up PR; the parameters
  ``radio_alpha_inj`` and ``radio_log_nu_break`` are already declared in
  :mod:`tengri.parameters._param_defs` per the reserved-params pattern.

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
:class:`tengri.core.PipelineState`'s "Cross-component reads" section:
**published derived quantity + documented fallback**, not a free
parameter snooped from another component's namespace.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import jax.numpy as jnp

from tengri.components.radio.radio import radio_total, radio_total_dpl
from tengri.core.component import (
    DerivedKey,
    ParamDeclaration,
    PipelineState,
    SEDComponentConfig,
    SEDComponentState,
)
from tengri.parameters.priors import Fixed

__all__ = ["RadioSEDComponent", "RadioSEDComponentConfig"]

# Mode strings for the AGN radio sub-model. Kept as a module-level
# constant so tests and downstream code can import it without
# instantiating the dataclass.
AGN_RADIO_MODELS: tuple[str, ...] = ("powerlaw", "dpl", "JP", "KP", "tribble")
_AGING_KERNELS_NOT_YET_IMPLEMENTED = ("JP", "KP", "tribble")


@dataclass(frozen=True)
class RadioSEDComponentConfig(SEDComponentConfig):
    r"""Frozen knobs for :class:`RadioSEDComponent`.

    Attributes
    ----------
    name : str
        Diagnostic identifier. Default ``"radio"``.
    sfr_mode : str
        FIR-radio correlation mode passed to :func:`radio_total`. One of
        ``{"bell2003", "delvecchio2021", "mccheyne2022"}``. Default
        ``"bell2003"``.
    include_freefree : bool
        Add Murphy+2011 thermal free-free component. Default ``True``
        (matches :func:`radio_total`'s default).
    agn_radio_model : str
        AGN radio sub-model. One of ``{"powerlaw", "dpl", "JP", "KP",
        "tribble"}``. Default ``"powerlaw"`` preserves current behaviour
        bit-identically. ``"JP"``, ``"KP"``, ``"tribble"`` are reserved
        and currently raise :class:`NotImplementedError` at apply time.
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


@dataclass(frozen=True)
class RadioSEDComponentState(SEDComponentState):
    r"""Radio has no precomputed tensors — typed marker only."""

    name: str = "radio"


@dataclass(frozen=True)
class RadioSEDComponent:
    r"""SEDComponent adapter around the radio physics module.

    Notes
    -----
    **JIT-compatible**: yes for ``agn_radio_model in {"powerlaw", "dpl"}``.
    JP/KP/Tribble are not yet implemented.

    **Additive**: writes ``sed_intrinsic = sed_intrinsic + L_radio(λ)``.
    Initialises ``sed_intrinsic`` from zeros if upstream did not.
    """

    config: RadioSEDComponentConfig = field(default_factory=RadioSEDComponentConfig)
    name: str = "radio"
    parameter_prefix: str = "radio_"

    def declared_parameters(self) -> list[ParamDeclaration]:
        r"""Free parameters this component owns.

        Mirrors the ``radio_*`` entries in
        :mod:`tengri.parameters._param_defs` so registration via this
        list and via the legacy registry produce the same priors.

        DPL parameters (``radio_alpha_thin``, ``radio_alpha_thick``,
        ``radio_log_nu_t``, ``radio_log_nu_cut``) are declared but
        ``Fixed`` by default, so the component is a no-op extension when
        ``agn_radio_model="powerlaw"``. Likewise the JP/KP/Tribble
        parameters (``radio_alpha_inj``, ``radio_log_nu_break``).
        """
        return [
            ParamDeclaration(
                "radio_q_ir",
                Fixed(2.64),
                "FIR-radio correlation parameter (bell2003 mode) [dimensionless]",
            ),
            ParamDeclaration(
                "radio_alpha_sf",
                Fixed(0.8),
                "Star-forming synchrotron spectral index [dimensionless]",
            ),
            ParamDeclaration(
                "radio_loudness",
                Fixed(0.0),
                "AGN radio loudness log10(L_5GHz / L_B) [dimensionless]",
            ),
            ParamDeclaration(
                "radio_alpha_agn",
                Fixed(0.7),
                "AGN radio spectral index (powerlaw model) [dimensionless]",
            ),
            ParamDeclaration(
                "radio_T_e",
                Fixed(1e4),
                "Free-free electron temperature [K]",
            ),
            ParamDeclaration(
                "radio_alpha_ff",
                Fixed(-0.1),
                "Free-free spectral index L_nu ∝ nu^alpha [dimensionless]",
            ),
            ParamDeclaration(
                "radio_alpha_thin",
                Fixed(-0.75),
                "AGN-DPL optically-thin (steep) spectral slope [dimensionless]",
            ),
            ParamDeclaration(
                "radio_alpha_thick",
                Fixed(-0.1),
                "AGN-DPL optically-thick (flat/inverted) spectral slope [dimensionless]",
            ),
            ParamDeclaration(
                "radio_log_nu_t",
                Fixed(10.0),
                "AGN-DPL log10(transition frequency / Hz)",
            ),
            ParamDeclaration(
                "radio_log_nu_cut",
                Fixed(13.0),
                "AGN-DPL log10(aging cutoff frequency / Hz)",
            ),
            ParamDeclaration(
                "radio_alpha_inj",
                Fixed(0.6),
                "JP/KP/Tribble injection spectral index (reserved) [dimensionless]",
            ),
            ParamDeclaration(
                "radio_log_nu_break",
                Fixed(10.0),
                "JP/KP/Tribble log10(spectral break frequency / Hz) (reserved)",
            ),
        ]

    def publishes(self) -> tuple[DerivedKey, ...]:
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

    def requires_optional(self) -> tuple[DerivedKey, ...]:
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
            DerivedKey("log_mstar", "dex", "Read from stellar if present; falls back to 10.0"),
        )

    def precompute(
        self,
        ssp_data: Any | None = None,
        wave_grid: jnp.ndarray | None = None,
    ) -> RadioSEDComponentState:
        r"""No-op precompute. Radio is a closed-form function of (λ, params)."""
        del ssp_data, wave_grid
        return RadioSEDComponentState(name=self.name)

    def apply(
        self,
        state: PipelineState,
        params: Mapping[str, jnp.ndarray],
    ) -> PipelineState:
        r"""Add radio emission to ``state.sed_intrinsic``.

        Dispatches to :func:`radio_total` (powerlaw) or
        :func:`radio_total_dpl` (dpl) based on
        :attr:`RadioSEDComponentConfig.agn_radio_model`. JP/KP/Tribble
        modes raise :class:`NotImplementedError` with a pointer to the
        follow-up PR.

        Parameters
        ----------
        state : PipelineState
            Must carry rest-frame ``wave`` (Å). If ``sed_intrinsic`` is
            ``None`` it is initialised to zeros of the same shape.
        params : mapping
            Receives ``radio_*`` keys plus the bare ``redshift`` from
            the allowlist. Cross-component scalars (``L_ir``,
            ``L_agn_bol``, ``log_mstar``) are read from
            ``state.derived`` with documented fallbacks.

        Returns
        -------
        PipelineState
            New state with ``sed_intrinsic`` updated.
        """
        model = self.config.agn_radio_model
        if model in _AGING_KERNELS_NOT_YET_IMPLEMENTED:
            raise NotImplementedError(
                f"agn_radio_model={model!r} is reserved but not yet implemented. "
                "Physical synchrotron-aging kernels (Jaffe & Perola 1973; "
                "Kardashev 1962 / Pacholczyk 1970; Tribble 1993) require "
                "precomputed pitch-angle integrals validated against BRATS "
                "(Harwood et al. 2013). Use 'powerlaw' or 'dpl' for now; "
                "the parameters radio_alpha_inj and radio_log_nu_break are "
                "already registered for the follow-up PR."
            )

        wave = state.wave

        L_ir = jnp.asarray(state.derived.get("L_ir", 0.0))
        L_agn_bol = jnp.asarray(state.derived.get("L_agn_bol", 0.0))
        log_mstar = jnp.asarray(state.derived.get("log_mstar", 10.0))
        z = jnp.asarray(params.get("redshift", 0.0))

        if model == "powerlaw":
            L_radio = radio_total(
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
                include_freefree=self.config.include_freefree,
                T_e=jnp.asarray(params["radio_T_e"]),
                alpha_ff=jnp.asarray(params["radio_alpha_ff"]),
            )
        else:  # model == "dpl"
            L_radio = radio_total_dpl(
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
                include_freefree=self.config.include_freefree,
                T_e=jnp.asarray(params["radio_T_e"]),
                alpha_ff=jnp.asarray(params["radio_alpha_ff"]),
            )

        if state.sed_intrinsic is None:
            new_sed = L_radio
        else:
            new_sed = state.sed_intrinsic + L_radio

        new_derived = dict(state.derived)
        new_derived["sed_radio"] = L_radio
        return state.with_(sed_intrinsic=new_sed, derived=new_derived)
