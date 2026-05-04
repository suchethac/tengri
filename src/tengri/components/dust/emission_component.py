# SPDX-License-Identifier: BSD-3-Clause
"""DustEmissionSEDComponent: SEDComponent adapter for IR re-emission.

Phase II-1 sixth adapter and the **first cross-component closed-loop**:
:class:`DustAttenuationSEDComponent` publishes ``state.derived["L_ir"]``
(the integral of absorbed UV/optical/NIR luminosity); this adapter
re-emits it as a modified blackbody, closing the dust energy balance
inside the orchestrator pipeline.

Scope
-----
This adapter wraps :func:`tengri.components.dust.emission.modified_blackbody`
only — the simplest IR template with two free parameters
(``dust_T``, ``dust_beta_ir``). Richer template families
(Casey+2012, Dale+2014, Draine & Li 2007/2014, Astrodust, BOSA, THEMIS)
will be separate adapters once their precompute paths are migrated to
the :class:`tengri.core.PipelineState` protocol; their templates need
SSP-grid-shaped cached tensors that justify a real
:meth:`SEDComponent.precompute` step.

Cross-component reads
---------------------
- ``state.derived["L_ir"]`` (erg/s) — published by
  :class:`DustAttenuationSEDComponent` via the energy-balance integral
  ``∫ (L_ν_intrinsic − L_ν_attenuated) dν``. Falls back to ``0.0`` if no
  upstream attenuator has run, in which case this adapter contributes
  nothing.

Pipeline ordering
-----------------
This adapter MUST run **after** the dust attenuator so ``L_ir`` is
present in ``state.derived``. The recommended natural order is
``[Stellar, AGN, Nebular, DustAttenuation, DustEmission, IGM, Radio, XRay]``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import jax.numpy as jnp

from tengri.components.dust.emission import modified_blackbody
from tengri.core.component import (
    ParamDeclaration,
    PipelineState,
    SEDComponentConfig,
    SEDComponentState,
)
from tengri.parameters.priors import Fixed

__all__ = ["DustEmissionSEDComponent", "DustEmissionSEDComponentConfig"]


@dataclass(frozen=True)
class DustEmissionSEDComponentConfig(SEDComponentConfig):
    r"""Frozen knobs for :class:`DustEmissionSEDComponent`.

    Attributes
    ----------
    name : str
        Diagnostic identifier. Default ``"dust_emission"``.
    template : str
        IR template name. Currently only ``"modified_blackbody"`` is
        supported by this adapter. Casey/Dale/DL07/DL14/Astrodust/BOSA/
        THEMIS adapters will be separate components in Phase II-3.
    """

    name: str = "dust_emission"
    template: str = "modified_blackbody"


@dataclass(frozen=True)
class DustEmissionSEDComponentState(SEDComponentState):
    r"""Modified blackbody has no precomputed tensors — typed marker."""

    name: str = "dust_emission"


@dataclass(frozen=True)
class DustEmissionSEDComponent:
    r"""SEDComponent adapter for modified-blackbody IR re-emission.

    Notes
    -----
    **JIT-compatible**: yes — :meth:`apply` is pure JAX.
    **Additive**: writes ``sed_intrinsic = sed_intrinsic + L_ir_emitted(λ)``.
    **Closed-loop**: reads ``L_ir`` from ``state.derived`` (published by
    :class:`DustAttenuationSEDComponent`) — the first adapter that
    consumes a derived quantity from another adapter rather than from
    :class:`StellarSEDComponent` (still pending) or directly from a
    parameter.

    The adapter is a no-op when ``state.derived["L_ir"] == 0`` (no
    attenuator ran upstream, or attenuation was perfectly transparent).
    """

    config: DustEmissionSEDComponentConfig = field(default_factory=DustEmissionSEDComponentConfig)
    name: str = "dust_emission"
    parameter_prefix: str = "dust_"

    def declared_parameters(self) -> list[ParamDeclaration]:
        r"""Free parameters this component owns.

        Note that ``dust_tau_v`` is owned by
        :class:`DustAttenuationSEDComponent`, not this one — the
        :func:`merge_declared_parameters` helper will refuse to register
        the same name twice if both adapters appear in the same
        pipeline (which is by design: ``dust_tau_v`` belongs to
        attenuation; ``dust_T`` and ``dust_beta_ir`` belong to emission).
        """
        if self.config.template != "modified_blackbody":
            raise NotImplementedError(
                f"DustEmissionSEDComponent currently only supports "
                f"template='modified_blackbody'; got {self.config.template!r}. "
                f"Casey/Dale/DL07/DL14/Astrodust/BOSA/THEMIS adapters land in Phase II-3."
            )
        return [
            ParamDeclaration(
                "dust_T",
                Fixed(30.0),
                "Modified blackbody dust temperature [K]",
            ),
            ParamDeclaration(
                "dust_beta_ir",
                Fixed(1.8),
                "Modified blackbody emissivity index β [dimensionless]",
            ),
        ]

    def precompute(
        self,
        ssp_data: Any | None = None,
        wave_grid: jnp.ndarray | None = None,
    ) -> DustEmissionSEDComponentState:
        r"""No-op precompute. Modified blackbody is closed-form."""
        del ssp_data, wave_grid
        return DustEmissionSEDComponentState(name=self.name)

    def apply(
        self,
        state: PipelineState,
        params: Mapping[str, jnp.ndarray],
    ) -> PipelineState:
        r"""Add IR re-emission to ``state.sed_intrinsic``.

        Parameters
        ----------
        state : PipelineState
            Must carry rest-frame ``wave``. Reads
            ``state.derived["L_ir"]`` (energy-balance absorbed
            luminosity, erg/s); falls back to 0 if absent.
        params : mapping
            Receives ``dust_*`` keys plus ``redshift`` from the bare-name
            allowlist.

        Returns
        -------
        PipelineState
            New state with ``sed_intrinsic`` updated and
            ``derived["L_dust_emitted"]`` published (the modified
            blackbody L_nu profile, useful for diagnostics).

        Notes
        -----
        :func:`modified_blackbody` accepts a ``redshift`` argument that
        applies a CMB-heating + CMB-contrast correction
        (da Cunha et al. 2013). This adapter passes ``params["redshift"]``
        through verbatim so high-z corrections work out of the box.
        """
        L_ir = jnp.asarray(state.derived.get("L_ir", 0.0))
        z = jnp.asarray(params.get("redshift", 0.0))

        L_dust_lnu = modified_blackbody(
            state.wave,
            L_absorbed=L_ir,
            dust_T=jnp.asarray(params["dust_T"]),
            dust_beta_ir=jnp.asarray(params["dust_beta_ir"]),
            redshift=z,
        )

        if state.sed_intrinsic is None:
            new_sed = L_dust_lnu
        else:
            new_sed = state.sed_intrinsic + L_dust_lnu

        new_derived = dict(state.derived)
        new_derived["L_dust_emitted"] = L_dust_lnu
        return state.with_(sed_intrinsic=new_sed, derived=new_derived)
