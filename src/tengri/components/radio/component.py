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
  Backwards-compatible default; behaviour bit-identical to pre-aging
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
:class:`tengri.protocols.PipelineState`'s "Cross-component reads" section:
**published derived quantity + documented fallback**, not a free
parameter snooped from another component's namespace.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import jax.numpy as jnp

from tengri.components.radio._params import PARAMS as _RADIO_PARAMS
from tengri.components.radio.radio import radio_total, radio_total_dpl
from tengri.protocols.component import (
    DerivedKey,
    ParamDeclaration,
    PipelineState,
    SEDComponentConfig,
    SEDComponentState,
)

__all__ = ["RadioSEDComponent", "RadioSEDComponentConfig"]

# Mode strings for the AGN radio sub-model. Kept as a module-level
# constant so tests and downstream code can import it without
# instantiating the dataclass.
#
# JP / KP / tribble (Jaffe & Perola 1973; Kardashev/Pacholczyk; Tribble
# 1993) are NOT in this tuple — they require precomputed pitch-angle
# integrals validated against BRATS (Harwood+2013). When that physics
# lands, the kernel names and their two free parameters (radio_alpha_inj,
# radio_log_nu_break) get added together in the same PR.
AGN_RADIO_MODELS: tuple[str, ...] = ("powerlaw", "dpl")


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
        AGN radio sub-model. One of :data:`AGN_RADIO_MODELS` —
        ``{"powerlaw", "dpl"}``. Default ``"powerlaw"`` preserves the
        pre-aging-cutoff behaviour bit-identically. Physical-aging
        kernels ``"JP"``, ``"KP"``, ``"tribble"`` are reserved names
        rejected at construction with a :class:`ValueError`; the physics
        lands in a follow-up PR.
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
    **JIT-compatible**: yes for every model in :data:`AGN_RADIO_MODELS`.

    **Additive**: writes ``sed_intrinsic = sed_intrinsic + L_radio(λ)``.
    Initialises ``sed_intrinsic`` from zeros if upstream did not.
    """

    config: RadioSEDComponentConfig = field(default_factory=RadioSEDComponentConfig)
    name: str = "radio"
    parameter_prefix: str = "radio_"

    def declared_parameters(self) -> list[ParamDeclaration]:
        r"""Free parameters this component owns.

        Returns the canonical :data:`PARAMS` tuple from
        :mod:`tengri.components.radio._params`. The legacy
        ``_RADIO_PARAMS`` bucket in :mod:`tengri.parameters._param_defs`
        is derived from the same tuple, so the two registration paths
        are guaranteed to agree.

        DPL parameters (``radio_alpha_thin``, ``radio_alpha_thick``,
        ``radio_log_nu_t``, ``radio_log_nu_cut``) are declared but
        ``Fixed`` by default, so the component is a no-op extension when
        ``agn_radio_model="powerlaw"``.
        """
        return list(_RADIO_PARAMS)

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
        :attr:`RadioSEDComponentConfig.agn_radio_model`.

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
        # JP / KP / tribble were previously dispatched here with a runtime
        # NotImplementedError. They now fail earlier — at construction —
        # because they are no longer in AGN_RADIO_MODELS. That validation
        # lives in RadioSEDComponentConfig.__post_init__.
        model = self.config.agn_radio_model

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

        return state.with_(
            sed_intrinsic=new_sed,
            derived=state.derived.with_(sed_radio=L_radio),
        )
