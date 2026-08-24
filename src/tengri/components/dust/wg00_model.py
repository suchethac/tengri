# SPDX-License-Identifier: BSD-3-Clause
"""WG00AttenuationSEDComponent: Witt & Gordon (2000) screen attenuation (``dust_type=3``).

The live-pipeline SEDComponent for the WG00 Monte-Carlo radiative-transfer
attenuation curves (FSPS ``dust_type=3``). It is the WG00 analog of
:class:`tengri.components.dust.component.DustAttenuationSEDComponent`: a
single-screen transform that reads ``sed_intrinsic`` and writes
``sed_attenuated = sed_intrinsic * exp(-A(λ; τ_V))``.

Unlike a fixed ``k(λ)`` law scaled by ``τ_V``, the WG00 *curve shape* depends on
``τ_V`` (high-τ sightlines self-shield → grayer attenuation), so the full
``A(λ; τ_V)`` table is interpolated (triweight in ``τ_V``) and applied directly.
The structural choices: dust grain population (MW/SMC), large-scale geometry
(shell/cloudy/dusty), and local density (homogeneous/clumpy): are static
selectors carried on :class:`WG00AttenuationSEDComponentConfig`; the single
free parameter is ``dust_tau_v``.

Data source: Witt & Gordon 2000 (ApJ 528, 799), as reformatted and distributed
by FSPS (Conroy & Gunn 2010) in ``$SPS_HOME/dust/alldirty_{h,c}.dat``: the same
tables FSPS reads for ``dust_type=3``. Vendored into
``data/wg00_attenuation_grid.h5`` by ``scripts/build_wg00_grid.py``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import jax.numpy as jnp

from tengri.components.template_threading import TemplateThreading
from tengri.parameters.priors import Fixed
from tengri.protocols.component import (
    DerivedKey,
    ForwardState,
    ParamDeclaration,
    SEDComponentConfig,
    SEDComponentState,
)

__all__ = [
    "WG00AttenuationSEDComponent",
    "WG00AttenuationSEDComponentConfig",
]


@dataclass(frozen=True)
class WG00AttenuationSEDComponentConfig(SEDComponentConfig):
    """Frozen structural knobs for :class:`WG00AttenuationSEDComponent`.

    Attributes
    ----------
    name: str
        Diagnostic identifier. Default ``"wg00_attenuation"``.
    dust_curve: {"mw", "smc"}
        Underlying dust grain population. Default ``"mw"``.
    geometry: {"shell", "cloudy", "dusty"}
        Large-scale star-dust geometry. Default ``"shell"`` (foreground screen).
    structure: {"homogeneous", "clumpy"}
        Local density structure. Default ``"homogeneous"``.
    """

    name: str = "wg00_attenuation"
    dust_curve: str = "mw"
    geometry: str = "shell"
    structure: str = "homogeneous"


@dataclass(frozen=True)
class WG00AttenuationSEDComponentState(SEDComponentState):
    """Cached WG00 interpolation closure.

    Attributes
    ----------
    wg00_fn: callable or None
        ``fn(wave, tau_v) -> A(λ)`` from
        :func:`tengri.components.dust.wg00.create_wg00_from_grid`, or ``None``
        if the vendored grid is unavailable (apply() then no-ops).
    """

    name: str = "wg00_attenuation"
    wg00_fn: Any | None = None


@dataclass(frozen=True)
class WG00AttenuationSEDComponent(TemplateThreading):
    r"""SEDComponent for Witt & Gordon (2000) screen attenuation (FSPS ``dust_type=3``).

    Notes
    -----
    **JIT-compatible**: yes, :meth:`apply` is pure JAX once the grid closure is
    built; the structural selectors are resolved to static indices at
    construction.

    **Gradient-safe**: yes, ``dust_tau_v`` is interpolated with a C²-continuous
    triweight kernel.

    **Transforms**: writes ``sed_attenuated = sed_intrinsic * exp(-A(λ; τ_V))``
    and publishes ``L_ir`` / ``L_absorbed`` (frequency-space integral of the
    absorbed luminosity) for downstream dust-IR re-emission.

    The single-screen model is intentional (WG00 is a foreground/mixed screen on
    the integrated stellar SED, not an age-resolved two-component law).

    References
    ----------
    .. [1] A. N. Witt and K. D. Gordon, "Multiple Scattering in Clumpy Media.
       II. Galactic Environments," ApJ, 528, 799 (2000).
       https://doi.org/10.1086/308197
    """

    config: WG00AttenuationSEDComponentConfig = field(
        default_factory=WG00AttenuationSEDComponentConfig
    )
    name: str = "wg00_attenuation"
    parameter_prefix: str = "dust_"
    _state: WG00AttenuationSEDComponentState | None = None

    def citations(self) -> tuple[str, ...]:
        """WG00 attenuation cites Witt & Gordon (2000)."""
        return ("witt_gordon2000",)

    def declared_parameters(self) -> list[ParamDeclaration]:
        """Free parameters: ``dust_tau_v`` (V-band optical depth, 0.25–10)."""
        return [
            ParamDeclaration(
                "dust_tau_v",
                Fixed(1.0),
                "V-band optical depth (WG00 grid range 0.25–10) [dimensionless]",
            ),
        ]

    def outputs(self) -> tuple[DerivedKey, ...]:
        """Cross-component derived keys published by the WG00 screen."""
        return (
            DerivedKey("L_ir", "erg/s", "Integrated dust-absorbed luminosity"),
            DerivedKey("L_absorbed", "erg/s", "Alias for L_ir (energy balance)"),
            DerivedKey("log_L_ir", "dex", "log10(L_ir / (erg/s)); float32-safe form"),
            DerivedKey("dust_attenuation_factor", "", "exp(-A(lambda; tau_v)) on pipeline grid"),
            DerivedKey("sed_dust_attenuated", "erg/s/Hz", "Attenuated stellar SED"),
        )

    def optional_inputs(self) -> tuple[DerivedKey, ...]:
        """Nebular continuum read opportunistically, for ordering only.

        Declaring ``sed_nebular`` an optional input makes the pipeline
        topological sort (ADR-0006) place the nebular component *before*
        this single screen, so the WG00 attenuation reddens the nebular
        continuum (folded into ``sed_intrinsic`` by the nebular component)
        together with the stellar light: matching bagpipes/FSPS/CIGALE.
        Without it the stable sort left dust *before* nebular, leaving the
        continuum unattenuated (the single-screen analog of the
        two-component bug fixed in #668). BakedIn backends publish zeros, so
        this is a no-op there; the screen acts on the summed
        ``sed_intrinsic`` and does not read the key directly.
        """
        return (
            DerivedKey(
                "sed_nebular",
                "erg/s/Hz",
                "Nebular continuum folded into sed_intrinsic before the screen",
            ),
        )

    def _build_curve_fn(self) -> Any | None:
        """Build (or reuse) the WG00 ``A(λ; τ_V)`` interpolation closure."""
        if self._state is not None and self._state.wg00_fn is not None:
            return self._state.wg00_fn
        from tengri.components.dust.wg00 import _find_wg00_grid, create_wg00_from_grid

        try:
            grid_path = _find_wg00_grid()
            return create_wg00_from_grid(
                grid_path,
                dust_curve=self.config.dust_curve,
                geometry=self.config.geometry,
                structure=self.config.structure,
            )
        except (FileNotFoundError, OSError, KeyError):
            return None

    def precompute(
        self,
        ssp_data: Any | None = None,
        wave_grid: jnp.ndarray | None = None,
        approx: Mapping[str, bool] | None = None,
        filters: tuple[tuple[jnp.ndarray, jnp.ndarray], ...] | None = None,
    ) -> WG00AttenuationSEDComponentState:
        """Load the WG00 grid closure into component state.

        Parameters
        ----------
        ssp_data, wave_grid, approx, filters
            Unused / Protocol-conformance arguments. The WG00 closure
            interpolates onto any target grid at apply() time.

        Returns
        -------
        WG00AttenuationSEDComponentState
            Holds the cached interpolation closure (``None`` if the grid is
            unavailable: apply() then passes the SED through unchanged).
        """
        del ssp_data, wave_grid, approx, filters
        return WG00AttenuationSEDComponentState(name=self.name, wg00_fn=self._build_curve_fn())

    def apply(
        self,
        state: ForwardState,
        params: Mapping[str, jnp.ndarray],
        ssp_data: Any | None = None,
        template_data: Any | None = None,
    ) -> ForwardState:
        r"""Apply WG00 screen attenuation to ``state.sed_intrinsic``.

        Parameters
        ----------
        state: ForwardState
            Must carry rest-frame ``wave``. No-op if ``sed_intrinsic`` is
            ``None`` or the WG00 grid is unavailable.
        params: mapping
            Receives ``dust_tau_v``.

        Returns
        -------
        ForwardState
            New state with ``sed_attenuated`` populated and ``L_ir`` /
            ``L_absorbed`` / ``dust_attenuation_factor`` published.
        """
        del template_data, ssp_data
        if state.sed_intrinsic is None:
            return state

        wg00_fn = self._state.wg00_fn if self._state is not None else None
        if wg00_fn is None:
            wg00_fn = self._build_curve_fn()
        if wg00_fn is None:
            # Grid unavailable: transparent pass-through (documented).
            return state

        tau_v = jnp.asarray(params["dust_tau_v"])
        a_lambda = wg00_fn(state.wave, tau_v)
        attenuation = jnp.exp(-a_lambda)
        attenuated = state.sed_intrinsic * attenuation

        # Energy balance: L_ir = ∫ (L_nu_intrinsic − L_nu_attenuated) dν,
        # LyC-masked (λ < 912 Å ionizes H, it does not heat dust: #922).
        from tengri.forward.energy_balance import bolometric_absorbed_log10, warn_if_corrupt
        from tengri.utils.physics_constants import C_AA
        from tengri.utils.scale import pow10

        nu = C_AA / state.wave
        # Log-space integral: ~1e43 erg/s is outside float32 (#1206).
        log_l_ir, _ = bolometric_absorbed_log10(
            state.sed_intrinsic, attenuated, nu, wave=state.wave
        )
        warn_if_corrupt(log_l_ir, component="wg00")
        l_ir = pow10(log_l_ir)

        derived_overrides = dict(
            dust_attenuation_factor=attenuation,
            L_ir=l_ir,
            L_absorbed=l_ir,
            log_L_ir=log_l_ir,
            sed_dust_attenuated=attenuated,
        )
        return state.with_(
            sed_intrinsic=attenuated,
            sed_attenuated=attenuated,
            derived=state.derived.with_(**derived_overrides),
        )


# Register in the unified component dispatch table so the grammar type
# ``dust_attenuation={'type': 'wg00'}`` resolves via _resolve_registry_component
# (the single dispatch seam), not a hardcoded class in build_components (#844).
from tengri.components.sed_model_component import _REGISTRY

_REGISTRY["wg00"] = WG00AttenuationSEDComponent
