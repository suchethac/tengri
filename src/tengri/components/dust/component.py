# SPDX-License-Identifier: BSD-3-Clause
"""DustAttenuationSEDComponent: screen-style dust attenuation as a SEDComponent.

The first SED component in the pipeline that **transforms** the SED rather than
adding to it: reads ``sed_intrinsic``, writes ``sed_attenuated``.

Scope (intentionally small)
---------------------------
Wraps a *single-component screen* attenuation law — picked at construction
time from the catalogue in :mod:`tengri.components.dust.attenuation`
(default: Calzetti+2000). For two-component (birth-cloud + diffuse ISM)
attenuation that needs the per-age stellar luminosity grid, see the
sibling :class:`DustSEDComponent` in
:mod:`tengri.components.dust.two_component`.

Cross-component reads
---------------------
- ``sed_intrinsic`` (erg/s/Hz) — produced by upstream emitters (stellar +
  AGN + radio + X-ray etc.). If ``None`` this adapter is a no-op.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import jax.numpy as jnp

from tengri.components.dust.attenuation import calzetti, resolve_dust_law
from tengri.parameters.priors import Fixed
from tengri.protocols.component import (
    DerivedKey,
    ForwardState,
    ParamDeclaration,
    SEDComponentConfig,
    SEDComponentState,
)

__all__ = ["DustAttenuationSEDComponent", "DustAttenuationSEDComponentConfig"]


@dataclass(frozen=True)
class DustAttenuationSEDComponentConfig(SEDComponentConfig):
    r"""Frozen knobs for :class:`DustAttenuationSEDComponent`.

    Attributes
    ----------
    name : str
        Diagnostic identifier. Default ``"dust_attenuation"``.
    law : str
        Attenuation law name resolved by
        :func:`tengri.components.dust.attenuation.resolve_dust_law`.
        Default ``"calzetti"``. Other built-in choices include
        ``"cardelli"``, ``"smc"``, ``"lmc"``, ``"prevot_smc"``,
        ``"li08"``, etc.

    Notes
    -----
    The law is resolved eagerly into a callable ``k(λ)`` at construction
    time so :meth:`apply` does not perform a registry lookup inside the
    JIT scope.
    """

    name: str = "dust_attenuation"
    law: str = "calzetti"


@dataclass(frozen=True)
class DustAttenuationSEDComponentState(SEDComponentState):
    r"""Cached attenuation curve evaluated on the pipeline wave grid.

    Attributes
    ----------
    k_lambda : jnp.ndarray, shape (n_wave,) | None
        Pre-evaluated normalised attenuation curve k(λ) (with k(5500 Å) = 1).
        ``None`` until :meth:`DustAttenuationSEDComponent.precompute` runs.
    """

    name: str = "dust_attenuation"
    k_lambda: jnp.ndarray | None = None


@dataclass(frozen=True)
class DustAttenuationSEDComponent:
    r"""SEDComponent adapter for a single-component screen attenuation law.

    Notes
    -----
    **JIT-compatible**: yes — :meth:`apply` is pure JAX once the
    attenuation curve is precomputed.
    **Transforms**: writes ``sed_attenuated = sed_intrinsic * exp(-tau_v * k(λ))``.
    Reads ``state.sed_intrinsic`` and is a no-op if it is ``None``.

    The single-component model is intentional — see module docstring.
    Two-component (Charlot & Fall 2000) attenuation will be a separate
    adapter once the stellar component publishes per-age luminosities.
    """

    config: DustAttenuationSEDComponentConfig = field(
        default_factory=DustAttenuationSEDComponentConfig
    )
    name: str = "dust_attenuation"
    parameter_prefix: str = "dust_"
    _state: DustAttenuationSEDComponentState | None = None

    def declared_parameters(self) -> list[ParamDeclaration]:
        r"""Free parameters this component owns.

        Returns ``dust_tau_v`` only — the law shape is fixed by
        :attr:`config.law` and is not a free parameter (changing the
        attenuation law mid-fit would require re-precomputing).
        """
        return [
            ParamDeclaration(
                "dust_tau_v",
                Fixed(0.3),
                "V-band optical depth at 5500 Å [dimensionless]",
            ),
        ]

    def outputs(self) -> tuple[DerivedKey, ...]:
        """Cross-component derived keys this dust component publishes.

        See :func:`tengri.forward.orchestrator.validate_pipeline`.
        """
        return (
            DerivedKey("L_ir", "erg/s", "Integrated dust-absorbed luminosity"),
            DerivedKey("L_absorbed", "erg/s", "Alias for L_ir (energy balance)"),
            DerivedKey(
                "dust_attenuation_factor",
                "",
                "exp(-tau_v * k(lambda)) on pipeline wave grid",
            ),
            DerivedKey("sed_dust_attenuated", "erg/s/Hz", "Attenuated stellar SED"),
        )

    def precompute(
        self,
        ssp_data: Any | None = None,
        wave_grid: jnp.ndarray | None = None,
        approx: Mapping[str, bool] | None = None,
        filters: tuple[tuple[jnp.ndarray, jnp.ndarray], ...] | None = None,
    ) -> DustAttenuationSEDComponentState:
        r"""Evaluate the attenuation curve k(λ) on the pipeline wave grid.

        Parameters
        ----------
        ssp_data : object | None
            Unused (this adapter does not depend on SSP data). Kept in
            the signature to match the :class:`SEDComponent` Protocol.
        wave_grid : jnp.ndarray, shape (n_wave,) | None
            Rest-frame wavelength grid in Å. Required.

        Returns
        -------
        DustAttenuationSEDComponentState
            Holds the cached k(λ) tensor.
        """
        del ssp_data, filters
        if wave_grid is None:
            # Permissive path: contract tests call precompute() with no
            # args. Return an unprimed state; apply() will compute
            # k(λ) lazily on first call.
            return DustAttenuationSEDComponentState(name=self.name, k_lambda=None)
        if self.config.law == "calzetti":
            # Avoid a registry lookup on the most common path.
            k = calzetti(wave_grid)
        else:
            law_fn = resolve_dust_law(self.config.law)
            k = law_fn(wave_grid)
        return DustAttenuationSEDComponentState(name=self.name, k_lambda=k)

    def apply(
        self,
        state: ForwardState,
        params: Mapping[str, jnp.ndarray],
        ssp_data: Any | None = None,
    ) -> ForwardState:
        r"""Apply screen attenuation to ``state.sed_intrinsic``.

        Parameters
        ----------
        state : ForwardState
            Must carry rest-frame ``wave``. If ``sed_intrinsic`` is
            ``None`` this method is a no-op (returns the input
            unchanged).
        params : mapping
            Receives ``dust_*`` keys plus ``redshift`` (unused here).

        Returns
        -------
        ForwardState
            New state with ``sed_attenuated`` populated.

        Notes
        -----
        Computes :math:`k(\lambda)` lazily inside :meth:`apply` if
        :meth:`precompute` was not called — this keeps the adapter
        usable in tests that drive components directly without an
        orchestrator that schedules precompute.
        """
        if state.sed_intrinsic is None:
            return state

        if self._state is not None and self._state.k_lambda is not None:
            k = self._state.k_lambda
        elif self.config.law == "calzetti":
            k = calzetti(state.wave)
        else:
            law_fn = resolve_dust_law(self.config.law)
            k = law_fn(state.wave)

        tau_v = jnp.asarray(params["dust_tau_v"])
        attenuation = jnp.exp(-tau_v * k)
        attenuated = state.sed_intrinsic * attenuation

        # Energy balance — integrate the absorbed luminosity in frequency
        # space and publish for downstream consumers (DustEmissionSEDComponent
        # re-emits it as a modified blackbody; RadioSEDComponent uses it
        # to set the SF radio amplitude via the FIR-radio correlation).
        # L_ir = ∫(L_nu_intrinsic - L_nu_attenuated) dν
        # = -∫(L_nu_intrinsic - L_nu_attenuated)/λ² dλ × c × <correction>
        # We use the simpler cgs identity: dν = -c/λ² dλ; trapezoid in
        # log-wavelength would be more accurate but the simple trap is
        # what the legacy pipeline uses (forward/_kernels/compositional.py).
        from tengri.utils.physics_constants import C_AA

        absorbed_lnu = state.sed_intrinsic - attenuated  # erg/s/Hz
        # Integrate in frequency: ∫ L_nu dν, with ν = c/λ.
        # Sort wavelength ascending → frequency descending; reverse and trap.
        nu = C_AA / state.wave  # Hz
        order = jnp.argsort(nu)
        l_ir = jnp.trapezoid(absorbed_lnu[order], nu[order])  # erg/s

        # Phase 3c-3c-ii: filter-level A(λ_eff) and A'(λ_eff) LUTs.
        # Published only when an upstream component (stellar) has put
        # ``filter_eff_waves`` into ``state.derived`` — i.e. only when
        # ``approx={'wave_precomp': True}`` is set on SEDModel.
        derived_overrides = dict(
            dust_attenuation_factor=attenuation,
            L_ir=l_ir,
            L_absorbed=l_ir,
            sed_dust_attenuated=attenuated,
        )
        filter_eff = state.derived.get("filter_eff_waves")
        if filter_eff is not None:
            # Evaluate the attenuation law at the filter pivots and at a
            # finite-difference offset to compute the slope analytically.
            if self.config.law == "calzetti":
                k_at = calzetti(filter_eff)
            else:
                law_fn = resolve_dust_law(self.config.law)
                k_at = law_fn(filter_eff)
            a_lut = jnp.exp(-tau_v * k_at)
            # k'(λ) via central finite difference. δλ = 1 Å is small
            # compared with filter widths (~100–10000 Å) and gives an
            # accurate slope for smooth analytic dust laws.
            d_lambda = jnp.asarray(1.0)
            if self.config.law == "calzetti":
                k_plus = calzetti(filter_eff + d_lambda)
                k_minus = calzetti(filter_eff - d_lambda)
            else:
                law_fn = resolve_dust_law(self.config.law)
                k_plus = law_fn(filter_eff + d_lambda)
                k_minus = law_fn(filter_eff - d_lambda)
            k_slope = (k_plus - k_minus) / (2.0 * d_lambda)
            # A'(λ) = d/dλ exp(-τ·k(λ)) = -τ · k'(λ) · A(λ).
            a_slope_lut = -tau_v * k_slope * a_lut
            derived_overrides["dust_attenuation_precomp"] = a_lut
            derived_overrides["dust_attenuation_slope_precomp"] = a_slope_lut

        return state.with_(
            sed_intrinsic=attenuated,
            sed_attenuated=attenuated,
            derived=state.derived.with_(**derived_overrides),
        )
