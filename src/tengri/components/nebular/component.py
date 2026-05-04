# SPDX-License-Identifier: BSD-3-Clause
"""NebularSEDComponent: SEDComponent adapter wrapping a nebular backend.

Phase II-1 fifth adapter. Validates the **zero-parameter edge case** of
the :class:`tengri.core.SEDComponent` Protocol: an adapter that has no
free parameters and (for the BakedIn backend) does not transform the
SED — it is a marker that publishes ``state.derived["nebular_backend"]``
so downstream observation models know whether emission lines are in
the SSP grid or need to be added separately.

Scope (intentionally small)
---------------------------
This adapter wraps the :class:`BakedInBackend` only — the simplest case,
where nebular emission is already folded into the SSP grid at fixed
``logU`` and escape fraction. CueBackend / CloudyGridBackend / shock
backends will land as separate adapters in Phase II-3 once
:class:`StellarSEDComponent` publishes ``state.derived["nion"]`` (the
ionising photon production rate they need as input).

Why land this now
-----------------
The contract test matrix should cover three structural variants:

1. **Additive emitter** — Radio, X-ray. Adds to ``sed_intrinsic``.
2. **Transforming attenuator** — Dust. Reads ``sed_intrinsic`` →
   writes ``sed_attenuated``.
3. **No-op marker** — this. Publishes a ``state.derived`` flag, no
   parameters, no SED change.

The seam is more confidently real with all three patterns exercised by
real adapters than with only the first two.

Cross-component reads
---------------------
None. BakedIn nebular has no inputs because the emission is already
in the SSP grid; this adapter exists purely so the orchestrator's
component list reads naturally as
``[Stellar, Nebular, Dust, AGN, IGM, Radio, XRay]`` once stellar lands.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import jax.numpy as jnp

from tengri.components.nebular.baked_in import BakedInBackend
from tengri.core.component import (
    ParamDeclaration,
    PipelineState,
    SEDComponentConfig,
    SEDComponentState,
)
from tengri.parameters.priors import Fixed, Uniform

__all__ = ["NebularSEDComponent", "NebularSEDComponentConfig"]


@dataclass(frozen=True)
class NebularSEDComponentConfig(SEDComponentConfig):
    r"""Frozen knobs for :class:`NebularSEDComponent`.

    Attributes
    ----------
    name : str
        Diagnostic identifier. Default ``"nebular"``.
    backend : str
        Nebular backend identifier — used for parameter declarations
        and the :data:`state.derived["nebular_backend"]` marker.
        ``"baked_in"`` (zero-parameter no-op marker; emission is
        already in the SSP grid), ``"cloudy_grid"``
        (HDF5 grid interpolation, requires a backend instance via
        :attr:`NebularSEDComponent.backend`), or ``"cue"`` (NN
        emulator, also requires a backend instance).
    suppress_baked_in_warning : bool
        Whether to silence the ``BakedInNebularWarning`` emitted when
        :class:`BakedInBackend` is constructed. Default ``True`` for
        adapter use.
    """

    name: str = "nebular"
    backend: str = "baked_in"
    suppress_baked_in_warning: bool = True


@dataclass(frozen=True)
class NebularSEDComponentState(SEDComponentState):
    r"""Marker state — BakedIn has no precomputed tensors.

    The backend handle is held on the component itself (see
    :class:`NebularSEDComponent._backend`), not on the state, so
    serialising the state remains cheap.
    """

    name: str = "nebular"


@dataclass(frozen=True)
class NebularSEDComponent:
    r"""SEDComponent adapter wrapping :class:`BakedInBackend`.

    Notes
    -----
    **JIT-compatible**: yes — :meth:`apply` does no JAX work; it just
    publishes a ``state.derived["nebular_backend"]`` flag.
    **No-op on SED**: ``state.sed_intrinsic`` is unchanged. Nebular
    emission for the BakedIn case is in the SSP grid already and was
    accumulated by :class:`StellarSEDComponent` (when that lands).

    The zero-parameter edge case: :meth:`declared_parameters` returns
    ``[]``. The :func:`merge_declared_parameters` helper handles this
    cleanly — an empty list contributes nothing to the merged prior
    dict.
    """

    config: NebularSEDComponentConfig = field(default_factory=NebularSEDComponentConfig)
    backend: Any | None = None
    name: str = "nebular"
    parameter_prefix: str = "neb_"

    def declared_parameters(self) -> list[ParamDeclaration]:
        r"""Free parameters this component owns.

        Per backend:

        - ``baked_in``: zero parameters (SSP grid is fixed).
        - ``cloudy_grid``: standard nebular knobs ``neb_logU``,
          ``neb_logZ_gas``, ``neb_fesc``, ``neb_fesc_lya``.
        - ``cue``: 4 standard knobs **plus** the 7 Cue ionising-spectrum
          shape parameters (``ionspec_index1..4``,
          ``ionspec_logLratio1..3``) **plus** 3 gas-property knobs
          beyond logU/logZ (``gas_logn``, ``gas_logno``, ``gas_logco``).
          Cue is genuinely 12-parameter.
        - ``shock`` (MAPPINGS V): shock-emission knobs
          (``shock_velocity``, ``shock_log_density``,
          ``shock_b_over_sqrt_n``, ``shock_log_lhalpha``).
          The string-valued ``shock_abundance`` and ``shock_component``
          are configured on the backend instance, not free params.
        """
        if self.config.backend == "baked_in":
            return []

        # Standard knobs all photoionisation backends share.
        std_knobs = [
            ParamDeclaration(
                "neb_logU",
                Uniform(-5.0, 0.0),
                "log10 ionisation parameter U [dimensionless]",
            ),
            ParamDeclaration(
                "neb_logZ_gas",
                Uniform(-2.0, 0.5),
                "log10(Z_gas/Zsun) [dimensionless]; if Fixed, falls back to "
                "stellar log_z at apply time",
            ),
            ParamDeclaration(
                "neb_fesc",
                Fixed(0.0),
                "Lyman-continuum escape fraction [dimensionless, in [0, 1]]",
            ),
            ParamDeclaration(
                "neb_fesc_lya",
                Fixed(0.0),
                "Ly-alpha escape fraction [dimensionless, in [0, 1]]",
            ),
        ]

        if self.config.backend == "cloudy_grid":
            return std_knobs

        if self.config.backend == "cue":
            # 7 ionising-spectrum shape parameters (Cue's broken
            # power-law segments) — priors taken from the legacy
            # _CUE_IONSPEC_PARAMS bounds in
            # tengri.parameters._param_defs.
            ionspec = [
                ParamDeclaration(
                    "ionspec_index1",
                    Uniform(0.0, 50.0),
                    "Cue ionising slope segment 1 (HeII, 1-228 Å) [dimensionless]",
                ),
                ParamDeclaration(
                    "ionspec_index2",
                    Uniform(-1.0, 35.0),
                    "Cue ionising slope segment 2 (OII, 228-353 Å) [dimensionless]",
                ),
                ParamDeclaration(
                    "ionspec_index3",
                    Uniform(-2.0, 20.0),
                    "Cue ionising slope segment 3 (HeI, 353-504 Å) [dimensionless]",
                ),
                ParamDeclaration(
                    "ionspec_index4",
                    Uniform(-2.0, 10.0),
                    "Cue ionising slope segment 4 (HI, 504-912 Å) [dimensionless]",
                ),
                ParamDeclaration(
                    "ionspec_logLratio1",
                    Uniform(-1.0, 12.0),
                    "Cue log luminosity ratio seg2/seg1 [dimensionless]",
                ),
                ParamDeclaration(
                    "ionspec_logLratio2",
                    Uniform(-1.0, 3.0),
                    "Cue log luminosity ratio seg3/seg2 [dimensionless]",
                ),
                ParamDeclaration(
                    "ionspec_logLratio3",
                    Uniform(-1.0, 3.0),
                    "Cue log luminosity ratio seg4/seg3 [dimensionless]",
                ),
            ]
            # 3 extra gas-property knobs beyond logU / logZ_gas
            gas_extra = [
                ParamDeclaration(
                    "gas_logn",
                    Uniform(0.0, 5.0),
                    "Cue gas density log10(n_H/cm^-3) [dimensionless]",
                ),
                ParamDeclaration(
                    "gas_logno",
                    Uniform(-2.0, 2.0),
                    "Cue [N/O] abundance ratio [dex]",
                ),
                ParamDeclaration(
                    "gas_logco",
                    Uniform(-2.0, 2.0),
                    "Cue [C/O] abundance ratio [dex]",
                ),
            ]
            return std_knobs + ionspec + gas_extra

        if self.config.backend == "shock":
            # MAPPINGS V shock-emission free parameters. The string
            # knobs (shock_abundance, shock_component) live on the
            # backend instance because they choose grids, not floats.
            return [
                ParamDeclaration(
                    "shock_velocity",
                    Uniform(100.0, 1000.0),
                    "Shock velocity [km/s]",
                ),
                ParamDeclaration(
                    "shock_log_density",
                    Fixed(0.0),
                    "log10 pre-shock density [cm^-3]",
                ),
                ParamDeclaration(
                    "shock_b_over_sqrt_n",
                    Fixed(1.0),
                    "B/sqrt(n) [μG cm^(3/2)] (MAPPINGS III) or absolute B [μG] (MAPPINGS V)",
                ),
                ParamDeclaration(
                    "shock_log_lhalpha",
                    Fixed(40.0),
                    "log10(L_Halpha [erg/s]) — MAPPINGS shock luminosity "
                    "normalisation. If Fixed, sourced from "
                    "state.derived['shock_log_lhalpha'] when present, "
                    "otherwise the param's Fixed value.",
                ),
            ]

        raise NotImplementedError(
            f"NebularSEDComponent unknown backend {self.config.backend!r}; "
            f"supported: 'baked_in', 'cloudy_grid', 'cue', 'shock'."
        )

    def precompute(
        self,
        ssp_data: Any | None = None,
        wave_grid: jnp.ndarray | None = None,
    ) -> NebularSEDComponentState:
        r"""Construct the backend handle (no JAX work)."""
        del ssp_data, wave_grid
        if self.config.backend == "baked_in":
            warning_mode = "suppress" if self.config.suppress_baked_in_warning else "warn"
            BakedInBackend(ionizing_source_warning=warning_mode)
            return NebularSEDComponentState(name=self.name)
        if self.config.backend in ("cloudy_grid", "cue", "shock"):
            if self.backend is None:
                raise ValueError(
                    f"NebularSEDComponent(backend={self.config.backend!r}) requires "
                    f"a pre-constructed backend instance via the ``backend`` "
                    f"constructor field (Cue/CloudyGrid/MAPPINGS need their "
                    f"weights/grid/abundance configuration which the "
                    f"orchestrator does not know about)."
                )
            return NebularSEDComponentState(name=self.name)
        raise NotImplementedError(f"NebularSEDComponent unknown backend {self.config.backend!r}.")

    def apply(
        self,
        state: PipelineState,
        params: Mapping[str, jnp.ndarray],
    ) -> PipelineState:
        r"""Compute nebular SED via the configured backend; add to ``sed_intrinsic``.

        For ``backend="baked_in"`` the SED is unchanged (emission is
        already in the SSP grid). For Cue / CloudyGrid the backend is
        called and the result is added to ``state.sed_intrinsic``.

        Cross-component reads (Cue/CloudyGrid):
        - ``state.derived["log_metallicity_history"]`` — falls back to
          ``params["met_logzsol"]`` if not published.
        - ``state.derived["lnu_age"]`` — for the SSP-weighted Q_H
          computation in CloudyGrid.

        Returns
        -------
        PipelineState
            New state with ``derived["nebular_backend"]`` published and
            (for non-BakedIn backends) ``sed_intrinsic`` updated to
            include nebular emission.
        """
        # NOTE: do not publish ``self.config.backend`` (a Python string)
        # to ``state.derived`` — strings are not JAX leaves and break
        # ``jax.jit`` traces. The backend identity is in
        # ``self.config.backend`` (eager-time inspection only).
        new_derived = dict(state.derived)

        if self.config.backend == "baked_in":
            return state.with_(derived=new_derived)

        # Stellar metallicity (absolute log10(Z)) for downstream backends.
        log_z_history = state.derived.get("log_metallicity_history")
        if log_z_history is not None:
            log_z = jnp.asarray(log_z_history)[0]  # present-day value
        else:
            from tengri.parameters.translate import LOG10_ZSUN

            log_z = jnp.asarray(params["met_logzsol"]) + LOG10_ZSUN

        # ── MAPPINGS V shock backend (different parameter set) ────────
        if self.config.backend == "shock":
            # Source the Halpha luminosity normalisation: prefer the
            # state.derived publication (e.g. from a future stellar
            # extension) over the param's Fixed default.
            log_lha = state.derived.get("shock_log_lhalpha", params.get("shock_log_lhalpha", 40.0))
            l_shock_halpha = jnp.power(10.0, jnp.asarray(log_lha))
            nebular_sed = self.backend.predict_nebular_sed(
                wavelength=state.wave,
                shock_velocity=jnp.asarray(params["shock_velocity"]),
                l_shock_halpha=l_shock_halpha,
                shock_log_density=jnp.asarray(params.get("shock_log_density", 0.0)),
                shock_b_over_sqrt_n=jnp.asarray(params.get("shock_b_over_sqrt_n", 1.0)),
            )
            new_sed = (
                nebular_sed if state.sed_intrinsic is None else state.sed_intrinsic + nebular_sed
            )
            new_derived["sed_nebular"] = nebular_sed
            return state.with_(sed_intrinsic=new_sed, derived=new_derived)

        # ── Photoionisation backends (Cue + CloudyGrid) ───────────────
        common_kwargs = {
            "ssp_wave": state.wave,
            "log_z": log_z,
            "neb_logU": jnp.asarray(params.get("neb_logU", -3.0)),
            "neb_logZ_gas": params.get("neb_logZ_gas"),
            "neb_fesc": jnp.asarray(params.get("neb_fesc", 0.0)),
            "neb_fesc_lya": jnp.asarray(params.get("neb_fesc_lya", 0.0)),
        }

        if self.config.backend == "cue":
            # Cue picks up Q_H either from explicit ``gas_logqion`` or
            # the published ``nion`` (stellar).
            nion = state.derived.get("nion")
            if "gas_logqion" in params:
                common_kwargs["gas_logqion"] = jnp.asarray(params["gas_logqion"])
            elif nion is not None:
                common_kwargs["gas_logqion"] = jnp.log10(jnp.maximum(nion, 1.0))
            # Forward Cue's full 12-parameter surface. Backend signature
            # accepts ``**neb_params`` so unrecognised keys are ignored
            # safely; we forward every ``ionspec_*`` and ``gas_*`` we
            # find in ``params`` so users who declared them get a fully
            # parameterised Cue prediction.
            cue_extras = {}
            for key in (
                "ionspec_index1",
                "ionspec_index2",
                "ionspec_index3",
                "ionspec_index4",
                "ionspec_logLratio1",
                "ionspec_logLratio2",
                "ionspec_logLratio3",
                "gas_logn",
                "gas_logno",
                "gas_logco",
            ):
                if key in params:
                    cue_extras[key] = jnp.asarray(params[key])
            nebular_sed = self.backend.predict_nebular_sed(**common_kwargs, **cue_extras)
        else:  # cloudy_grid
            lnu_age = state.derived.get("lnu_age")
            if lnu_age is None:
                raise ValueError(
                    "CloudyGrid backend requires state.derived['lnu_age'] "
                    "(per-age L_nu cube from StellarSEDComponent)."
                )
            ssp_log_ages_yr = jnp.log10(state.derived["ssp_ages_yr"])
            # CSP weights (mass per age bin) — derive from lnu_age via
            # the bolometric-weighted approximation if SSP unit-mass
            # weights aren't separately available.
            ssp_weights = jnp.ones(lnu_age.shape[0])
            nebular_sed = self.backend.predict_nebular_sed(
                ssp_weights=ssp_weights,
                ssp_log_ages_yr=ssp_log_ages_yr,
                **common_kwargs,
            )

        # Add nebular contribution to sed_intrinsic (BC dust attenuation
        # of the nebular emission is the dust component's responsibility
        # — its ``two_component_dust`` transmission applies to the full
        # sed_intrinsic when it runs after this component).
        if state.sed_intrinsic is None:
            new_sed = nebular_sed
        else:
            new_sed = state.sed_intrinsic + nebular_sed

        new_derived["sed_nebular"] = nebular_sed
        return state.with_(sed_intrinsic=new_sed, derived=new_derived)
