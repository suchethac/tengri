# SPDX-License-Identifier: BSD-3-Clause
"""DustSEDComponent: Phase II-3 two-component attenuation + IR re-emission.

The Charlot & Fall (2000) two-component attenuation model with
energy-balanced IR re-emission. Reads the per-age stellar L_ν cube
published by :class:`tengri.components.stellar.StellarSEDComponent`,
applies an age-dependent extinction (birth cloud + diffuse ISM), and
produces a full attenuated + IR-re-emitted SED.

Sibling to :class:`tengri.components.dust.DustAttenuationSEDComponent`
(single-screen, ships in Phase II-1). The two-component variant lives
in this separate module because it requires upstream stellar
publications (``lnu_age``, ``ssp_ages_yr``) that did not exist before
Phase II-2.

Cross-component reads
---------------------
- ``state.derived["lnu_age"]``  — (n_age, n_wave) per-age L_ν cube
  (erg/s/Hz) from :class:`StellarSEDComponent`.
- ``state.derived["ssp_ages_yr"]`` — (n_age,) age axis (yr) from
  :class:`StellarSEDComponent`.

Cross-component publications
----------------------------
- ``state.derived["L_ir"]`` (scalar, erg/s) — total IR luminosity
  re-radiated by dust, consumed by
  :class:`tengri.components.radio.RadioSEDComponent` (FIR-radio
  correlation) and :class:`tengri.components.xray.XRaySEDComponent`.
- ``state.sed_intrinsic`` is overwritten with the attenuated stellar
  SED + IR re-emission added.

Architectural notes
-------------------
``ssp_data`` is **not** held on this component — it reads what it
needs from ``state.derived``. ``precompute()`` returns an empty
marker, consistent with every adapter except the stellar one.
The IR-emission backend is resolved lazily inside ``apply()`` so
template-grid HDF5 loading happens on first invocation.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import jax.numpy as jnp

from tengri.components.dust.attenuation import two_component_dust
from tengri.components.dust.emission import resolve_emission_model
from tengri.core.component import (
    ParamDeclaration,
    PipelineState,
    SEDComponentConfig,
    SEDComponentState,
)
from tengri.parameters.priors import Fixed, Uniform
from tengri.utils.physics_constants import C_AA

__all__ = [
    "DustSEDComponent",
    "DustSEDComponentConfig",
    "DustSEDComponentState",
]


@dataclass(frozen=True)
class DustSEDComponentConfig(SEDComponentConfig):
    """Frozen knobs for :class:`DustSEDComponent`.

    Parameters
    ----------
    name : str
        Diagnostic identifier. Default ``"dust"``.
    law_bc : str
        Attenuation-law registry key for the birth-cloud (young star)
        component. Default ``"power_law"``.
    law_diff : str
        Attenuation-law registry key for the diffuse ISM (old star)
        component. Default ``"power_law"``.
    emission_model : str
        IR emission template registry key. One of
        ``"modified_blackbody"``, ``"casey2012"``, ``"dale2014"``,
        ``"draine_li2007"``, ``"draine_li2014"``. Default
        ``"modified_blackbody"`` because it has no template-grid
        dependency.
    t_birth_yr : float
        Birth-cloud dispersal age (sigmoid centre, yr).
        Default 1e7 (10 Myr) per Charlot & Fall (2000).
    transition_width_dex : float
        Sigmoid width (dex) for the BC→diffuse age transition.
    """

    name: str = "dust"
    law_bc: str = "power_law"
    law_diff: str = "power_law"
    emission_model: str = "modified_blackbody"
    t_birth_yr: float = 1e7
    transition_width_dex: float = 0.3


@dataclass(frozen=True)
class DustSEDComponentState(SEDComponentState):
    """Marker state — emission templates are resolved lazily inside ``apply``."""

    name: str = "dust"


@dataclass(frozen=True)
class DustSEDComponent:
    """SEDComponent adapter for two-component dust + energy-balanced IR.

    Notes
    -----
    **JIT-compatible**: yes — :meth:`apply` is pure JAX once the
    attenuation-law registry lookup completes (these registries return
    plain JAX functions that fold cleanly into the JIT trace).
    **Pipeline ordering**: dust runs after stellar (and after nebular
    if a nebular component publishes its emission with BC dust applied
    in-place). Reads ``state.derived["lnu_age"]`` and
    ``["ssp_ages_yr"]``; overwrites ``state.sed_intrinsic`` with the
    attenuated stellar SED plus the IR re-emission.
    """

    config: DustSEDComponentConfig = field(default_factory=DustSEDComponentConfig)
    name: str = "dust"
    parameter_prefix: str = "dust_"

    def declared_parameters(self) -> list[ParamDeclaration]:
        """Free parameters this component owns.

        Mirrors the canonical ``dust_*`` priors in
        :mod:`tengri.parameters._param_defs`. Users may override any
        entry as :class:`Fixed` to drop it from the prior.
        """
        return [
            ParamDeclaration(
                "dust_tau_bc",
                Uniform(0.0, 4.0),
                "Birth-cloud V-band optical depth [dimensionless]",
            ),
            ParamDeclaration(
                "dust_tau_diff",
                Uniform(0.0, 3.0),
                "Diffuse ISM V-band optical depth [dimensionless]",
            ),
            ParamDeclaration(
                "dust_slope",
                Fixed(-0.7),
                "Power-law attenuation slope (Charlot-Fall convention) [dimensionless]",
            ),
            ParamDeclaration(
                "dust_eta_balance",
                Fixed(1.0),
                "Energy-balance relaxation factor (1.0 = strict balance) [dimensionless]",
            ),
            ParamDeclaration(
                "dust_T",
                Fixed(35.0),
                "Cold-dust temperature for MBB / Casey 2012 [K]",
            ),
            ParamDeclaration(
                "dust_beta_ir",
                Fixed(1.6),
                "Cold-dust emissivity index for MBB / Casey 2012 [dimensionless]",
            ),
            ParamDeclaration(
                "dust_alpha_dale",
                Fixed(2.0),
                "Dale 2014 template-family alpha [dimensionless, in 0.0625-4.0]",
            ),
            ParamDeclaration(
                "dust_umin",
                Fixed(1.0),
                "DL07/DL14 minimum radiation field U_min [Habing]",
            ),
            ParamDeclaration(
                "dust_qpah",
                Fixed(2.5),
                "DL07/DL14 PAH mass fraction [%]",
            ),
            ParamDeclaration(
                "dust_gamma_dl",
                Fixed(0.01),
                "DL07/DL14 photon-dominated mass fraction [dimensionless]",
            ),
            ParamDeclaration(
                "dust_alpha_dl14",
                Fixed(2.0),
                "DL14 power-law slope [dimensionless, in 1.0-3.0]",
            ),
            ParamDeclaration(
                "dust_alpha_mir",
                Fixed(2.0),
                "Casey 2012 mid-IR power-law slope [dimensionless]",
            ),
            ParamDeclaration(
                "dust_f_obscuration",
                Fixed(0.0),
                "Clumpy-geometry obscuration fraction [dimensionless, in [0, 1]]",
            ),
            ParamDeclaration(
                "dust_bump_strength",
                Fixed(0.0),
                "2175 Å UV bump strength (Kriek-Conroy) [dimensionless]",
            ),
            ParamDeclaration(
                "dust_delta",
                Fixed(0.0),
                "Kriek-Conroy attenuation slope deviation [dimensionless]",
            ),
            ParamDeclaration(
                "dust_Rv",
                Fixed(3.1),
                "Cardelli total-to-selective extinction R_V [dimensionless]",
            ),
        ]

    def precompute(
        self,
        ssp_data: Any | None = None,
        wave_grid: jnp.ndarray | None = None,
    ) -> DustSEDComponentState:
        """No-op marker — consistent with all other Phase II adapters."""
        del ssp_data, wave_grid
        return DustSEDComponentState(name=self.name)

    def apply(
        self,
        state: PipelineState,
        params: Mapping[str, jnp.ndarray],
    ) -> PipelineState:
        """Apply two-component attenuation + IR re-emission.

        Parameters
        ----------
        state : PipelineState
            Must carry ``wave`` and have stellar published
            ``lnu_age`` (n_age, n_wave) and ``ssp_ages_yr`` (n_age,)
            in its ``derived`` dict.
        params : mapping
            Receives ``dust_*`` keys plus the bare ``redshift``.

        Returns
        -------
        PipelineState
            New state with ``sed_intrinsic`` set to the attenuated
            stellar SED plus IR re-emission, and ``L_ir`` published
            to ``derived``.
        """
        if "lnu_age" not in state.derived or "ssp_ages_yr" not in state.derived:
            raise ValueError(
                "DustSEDComponent.apply requires upstream stellar publications "
                "(state.derived['lnu_age'] and ['ssp_ages_yr']). Place "
                "StellarSEDComponent before DustSEDComponent in the chain."
            )

        wave = state.wave
        lnu_age = jnp.asarray(state.derived["lnu_age"])  # (n_age, n_wave)
        ssp_ages_yr = jnp.asarray(state.derived["ssp_ages_yr"])

        # ── 1. Two-component transmission T(λ, age) ─────────────────────
        transmission = two_component_dust(
            wavelength=wave,
            age_grid=ssp_ages_yr,
            tau_v1=jnp.asarray(params["dust_tau_bc"]),
            tau_v2=jnp.asarray(params["dust_tau_diff"]),
            law_bc=self.config.law_bc,
            law_diff=self.config.law_diff,
            f_obscuration=jnp.asarray(params.get("dust_f_obscuration", 0.0)),
            t_birth=self.config.t_birth_yr,
            transition_width=self.config.transition_width_dex,
            n_slope=jnp.asarray(params.get("dust_slope", -0.7)),
            dust_bump_strength=jnp.asarray(params.get("dust_bump_strength", 0.0)),
            dust_delta=jnp.asarray(params.get("dust_delta", 0.0)),
            dust_Rv=jnp.asarray(params.get("dust_Rv", 3.1)),
        )  # (n_age, n_wave), in [0, 1]

        # ── 2. Apply transmission per age and aggregate ────────────────
        lnu_age_attenuated = lnu_age * transmission
        sed_attenuated = jnp.sum(lnu_age_attenuated, axis=0)
        sed_intrinsic_stellar = jnp.sum(lnu_age, axis=0)

        # ── 3. Energy balance: ∫ (L_nu_intrinsic - L_nu_attenuated) dν ──
        # ν = c/λ. trapezoid(integrand, x=ν) with ν descending returns a
        # negative signed area; abs() recovers the positive erg/s.
        # Mirrors forward/pipeline.py:815.
        nu = C_AA / wave
        absorbed_lnu = sed_intrinsic_stellar - sed_attenuated
        L_absorbed = jnp.abs(jnp.trapezoid(absorbed_lnu, nu))
        eta_balance = jnp.asarray(params.get("dust_eta_balance", 1.0))
        L_ir = jnp.maximum(L_absorbed * eta_balance, 0.0)

        # ── 4. IR emission template ────────────────────────────────────
        emission_fn = resolve_emission_model(self.config.emission_model)
        z = jnp.asarray(params.get("redshift", 0.0))
        sed_ir = emission_fn(
            wave,
            L_ir,
            dust_T=jnp.asarray(params.get("dust_T", 35.0)),
            dust_beta_ir=jnp.asarray(params.get("dust_beta_ir", 1.6)),
            dust_alpha_dale=jnp.asarray(params.get("dust_alpha_dale", 2.0)),
            dust_umin=jnp.asarray(params.get("dust_umin", 1.0)),
            dust_qpah=jnp.asarray(params.get("dust_qpah", 2.5)),
            dust_gamma_dl=jnp.asarray(params.get("dust_gamma_dl", 0.01)),
            dust_alpha_dl14=jnp.asarray(params.get("dust_alpha_dl14", 2.0)),
            dust_alpha_mir=jnp.asarray(params.get("dust_alpha_mir", 2.0)),
            redshift=z,
        )

        # ── 5. Combine and publish ─────────────────────────────────────
        sed_total = sed_attenuated + sed_ir
        new_derived = dict(state.derived)
        new_derived["L_ir"] = L_ir
        new_derived["L_absorbed"] = L_absorbed
        new_derived["sed_dust_attenuated"] = sed_attenuated
        new_derived["sed_dust_ir"] = sed_ir
        return state.with_(sed_intrinsic=sed_total, derived=new_derived)
