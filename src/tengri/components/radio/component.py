# SPDX-License-Identifier: BSD-3-Clause
"""RadioSEDComponent: SEDComponent adapter around :func:`radio_total`.

Phase II-1 first-adapter exercise. The physics in
:mod:`tengri.components.radio.radio` is unchanged; this is a thin
wrapper that satisfies :class:`tengri.core.SEDComponent` so the
orchestrator can run radio alongside other Phase-II adapters.

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

from tengri.components.radio.radio import radio_total
from tengri.core.component import (
    ParamDeclaration,
    PipelineState,
    SEDComponentConfig,
    SEDComponentState,
)
from tengri.parameters.priors import Fixed

__all__ = ["RadioSEDComponent", "RadioSEDComponentConfig"]


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
    """

    name: str = "radio"
    sfr_mode: str = "bell2003"
    include_freefree: bool = True


@dataclass(frozen=True)
class RadioSEDComponentState(SEDComponentState):
    r"""Radio has no precomputed tensors — typed marker only."""

    name: str = "radio"


@dataclass(frozen=True)
class RadioSEDComponent:
    r"""SEDComponent adapter around :func:`radio_total`.

    Notes
    -----
    **JIT-compatible**: yes — :meth:`apply` is pure JAX.
    **Additive**: writes ``sed_intrinsic = sed_intrinsic + L_radio(λ)``.
    Initialises ``sed_intrinsic`` from zeros if upstream did not.
    """

    config: RadioSEDComponentConfig = field(default_factory=RadioSEDComponentConfig)
    name: str = "radio"
    parameter_prefix: str = "radio_"

    def declared_parameters(self) -> list[ParamDeclaration]:
        r"""Free parameters this component owns.

        Mirrors the ``radio_*`` entries already in
        :mod:`tengri.parameters._param_defs` so registration via this
        list and via the legacy registry produce the same priors.
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
                "AGN radio spectral index [dimensionless]",
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
        ]

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
        wave = state.wave

        L_ir = jnp.asarray(state.derived.get("L_ir", 0.0))
        L_agn_bol = jnp.asarray(state.derived.get("L_agn_bol", 0.0))
        log_mstar = jnp.asarray(state.derived.get("log_mstar", 10.0))
        z = jnp.asarray(params.get("redshift", 0.0))

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

        if state.sed_intrinsic is None:
            new_sed = L_radio
        else:
            new_sed = state.sed_intrinsic + L_radio

        new_derived = dict(state.derived)
        new_derived["sed_radio"] = L_radio
        return state.with_(sed_intrinsic=new_sed, derived=new_derived)
