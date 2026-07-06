# SPDX-License-Identifier: BSD-3-Clause
"""NebularSEDComponent: nebular emission as a SEDComponent.

A thin SEDComponent that dispatches to a chosen backend (BakedIn,
CloudyGrid, or Cue). For ``BakedInBackend`` the nebular emission is
already folded into the SSP grid at fixed ``logU`` and escape fraction
— the component is then a no-op on the SED, and only publishes
``state.derived["nebular_backend"]`` so downstream observation models
know whether emission lines need adding separately or are already
present in the stellar templates.

CueBackend and CloudyGridBackend become free-parameter components: they
read ``state.derived["nion"]`` (the ionizing photon production rate
from the stellar block) and add the resulting line + continuum SED to
``sed_intrinsic``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import jax.numpy as jnp

from tengri.components.nebular.baked_in import BakedInBackend
from tengri.parameters.priors import Fixed, Uniform
from tengri.protocols.component import (
    DerivedKey,
    ForwardState,
    ParamDeclaration,
    SEDComponentConfig,
    SEDComponentState,
)

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
    cue_full_catalog : bool
        For the ``"cue"`` backend only. When ``True``, expose the full
        Cue-trained line catalog (~271 species) via ``state.derived
        ["line_waves"]`` / ``["line_lums"]`` so users can query HeII
        1640, HeI 10830 and other high-z diagnostics via
        :meth:`tengri.forward.prediction.EmissionLines.get`. Default
        ``False`` matches the pre-#303 behavior (128 CLOUDY/FSPS
        lines) and avoids surprising users who iterate over
        ``all_waves`` / ``all_lums``. No effect on the headline
        Hα/Hβ/etc. named accessors, which always work.
    """

    name: str = "nebular"
    backend: str = "baked_in"
    suppress_baked_in_warning: bool = True
    cue_full_catalog: bool = False


@dataclass(frozen=True)
class NebularSEDComponentState(SEDComponentState):
    r"""State for the nebular component.

    BakedIn has no precomputed tensors — the backend handle is held on
    the component itself. Non-BakedIn backends (Cue / CloudyGrid /
    Shock) optionally cache filter passbands when
    ``approx=WavePrecomp()`` is set on the parent SEDModel;
    :meth:`NebularSEDComponent.apply` uses them to filter-integrate the
    nebular SED contribution and publish ``nebular_phot_lnu_precomp``.
    """

    name: str = "nebular"
    filter_waves: Any | None = None
    filter_trans: Any | None = None


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
    _state: NebularSEDComponentState | None = None
    # Tuple prefix so the MAPPINGS shock backend (``shock_*``) and the
    # photoionization backends (``neb_*``, ``ionspec_*``, ``gas_*``) all
    # flow through the standard prefix-stripping path. Backends silently
    # ignore keys they don't consume — passing ``shock_*`` to Cue is
    # harmless, and vice versa.
    parameter_prefix: tuple[str, ...] = ("neb_", "shock_", "ionspec_", "gas_")

    def citations(self) -> tuple[str, ...]:
        """Nebular-backend citations (Cue / Cloudy / baked-in / shock) are
        config-driven via
        :data:`tengri.citations.associations.NEBULAR_BACKEND_CITATIONS`."""
        return ()

    def declared_parameters(self) -> list[ParamDeclaration]:
        r"""Free parameters this component owns.

        Per backend:

        - ``baked_in``: zero parameters (SSP grid is fixed).
        - ``cloudy_grid``: standard nebular knobs ``neb_logU``,
          ``neb_logZ_gas``, ``neb_fesc``, ``neb_fesc_lya``.
        - ``cue``: 4 standard knobs **plus** the 7 Cue ionizing-spectrum
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

        # Standard knobs all photoionization backends share.
        std_knobs = [
            ParamDeclaration(
                "neb_logU",
                # default -2.5: representative HII-region ionization parameter
                # (Cue/CLOUDY grids span logU ~ -3..-2). Explicit so '*': FIXED
                # does not silently use the prior midpoint (#477 / #478 / #845).
                Uniform(-5.0, 0.0, default=-2.5),
                "log10 ionization parameter U [dimensionless]",
            ),
            ParamDeclaration(
                "neb_logZ_gas",
                # default 0.0: solar gas-phase metallicity.
                Uniform(-2.0, 0.5, default=0.0),
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
            ParamDeclaration(
                "neb_fdust",
                Fixed(0.0),
                "Dust-absorption fraction of ionizing photons in HII regions "
                "[dimensionless, in [0, 1]]",
            ),
        ]

        if self.config.backend in ("cloudy_grid", "cb19", "mappings"):
            return std_knobs

        if self.config.backend == "cue":
            # The 7 Cue ionizing-spectrum shape parameters (broken power-law
            # segments) + 3 gas-property knobs are declared ONCE in
            # ``components/nebular/_params.py`` and consumed verbatim here —
            # the same tuples the flat-builder bucket derives from, so the two
            # construction paths cannot drift (#887). Their physical defaults
            # (1-Myr solar-Z BPASS fit; n_H=100, solar [N/O]/[C/O]) live there.
            from tengri.components.nebular._params import (
                CUE_GAS_EXTRA_PARAMS,
                CUE_IONSPEC_PARAMS,
            )

            return std_knobs + list(CUE_IONSPEC_PARAMS) + list(CUE_GAS_EXTRA_PARAMS)

        if self.config.backend == "shock":
            # MAPPINGS V shock-emission free parameters. The string
            # knobs (shock_abundance, shock_component) live on the
            # backend instance because they choose grids, not floats.
            return [
                ParamDeclaration(
                    "shock_velocity",
                    # default 250 km/s: representative MAPPINGS shock velocity.
                    Uniform(100.0, 1000.0, default=250.0),
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
                    "normalization. If Fixed, sourced from "
                    "state.derived['shock_log_lhalpha'] when present, "
                    "otherwise the param's Fixed value.",
                ),
            ]

        raise NotImplementedError(
            f"NebularSEDComponent unknown backend {self.config.backend!r}; "
            f"supported: 'baked_in', 'cloudy_grid', 'cb19', 'mappings', 'cue', 'shock'."
        )

    def outputs(self) -> tuple[DerivedKey, ...]:
        """Cross-component derived keys this nebular component publishes.

        See :func:`tengri.forward.orchestrator.validate_pipeline`.
        """
        outs = [
            DerivedKey("sed_nebular", "erg/s/Hz", "Photoionized continuum + lines"),
            DerivedKey("line_waves", "Angstrom", "Line vacuum wavelengths"),
            DerivedKey("line_lums", "erg/s", "Line luminosities"),
            DerivedKey(
                "lyc_transmission",
                "",
                "Stellar LyC survival fraction where(λ<912, neb_fesc, 1)",
            ),
        ]
        # ``sed_shock`` is owned by this component ONLY on the mutually-exclusive
        # ``backend="shock"`` path. When a photoionized backend composes with a
        # separate additive :class:`ShockNebular` component (#851), that
        # component owns ``sed_shock`` — declaring it here too would trip the
        # duplicate-publisher guard. Consumers read it via ``.get(..., zeros)``,
        # so leaving it undeclared for photoionized/baked-in backends is safe.
        if self.config.backend == "shock":
            outs.insert(1, DerivedKey("sed_shock", "erg/s/Hz", "MAPPINGS shock emission"))
        return tuple(outs)

    def inputs(self) -> tuple[DerivedKey, ...]:
        """Cross-component derived keys this nebular component reads.

        Backend-dependent:

        - ``"baked_in"`` (Sanders+2024 grid) reads nothing from
          ``state.derived`` — the backend operates on the SED in-place
          and ignores stellar-age-resolved tensors. Returns ``()``.
        - All other photoionization backends (``"cue"``,
          ``"cloudy_grid"``) and the MAPPINGS shock backend
          (``"shock"``) require the age-resolved stellar outputs to
          weight per-age ionizing rates. Returns ``lnu_age`` +
          ``ssp_ages_yr`` (and ``age_weights`` for the CloudyGrid
          path, which sums the per-bin grid lookups).

        Promotes the previously-fatal KeyError at JIT trace time
        (``state.derived["lnu_age"]`` → ``KeyError`` if Stellar is
        absent) to a construction-time
        :class:`tengri.protocols.component.ComponentIOError` with a
        "Did you mean: ..." hint. Phase A of issue #21; see
        ADR-0004 for the input/output design.
        """
        backend = self.config.backend
        if backend == "baked_in":
            return ()
        deps = [
            DerivedKey("lnu_age", "erg/s/Hz", "Per-age L_nu cube (n_age, n_wave)"),
            DerivedKey("ssp_ages_yr", "yr", "SSP age axis"),
        ]
        if backend == "cloudy_grid":
            deps.append(DerivedKey("age_weights", "Msun", "CSP mass weights per SSP age bin"))
        return tuple(deps)

    def precompute(
        self,
        ssp_data: Any | None = None,
        wave_grid: jnp.ndarray | None = None,
        approx: Mapping[str, bool] | None = None,
        filters: tuple[tuple[jnp.ndarray, jnp.ndarray], ...] | None = None,
    ) -> NebularSEDComponentState:
        r"""Construct the backend handle and cache filters when wave_precomp on.

        BakedIn nebular doesn't need filters — its emission is in the SSP
        grid and is covered by the stellar LUT. Non-BakedIn backends
        (Cue / CloudyGrid / Shock) cache the filter passbands so
        :meth:`apply` can publish ``nebular_phot_lnu_precomp`` for
        consumption by ``predict_via_precomp``.
        """
        del ssp_data, wave_grid
        approx = approx or {}
        cache_filters = (
            approx.get("wave_precomp")
            and filters is not None
            and self.config.backend != "baked_in"
        )
        cached_filter_waves = None
        cached_filter_trans = None
        if cache_filters:
            cached_filter_waves = tuple(jnp.asarray(fw) for fw, _ in filters)
            cached_filter_trans = tuple(jnp.asarray(ft) for _, ft in filters)

        if self.config.backend == "baked_in":
            warning_mode = "suppress" if self.config.suppress_baked_in_warning else "warn"
            BakedInBackend(ionizing_source_warning=warning_mode)
            return NebularSEDComponentState(name=self.name)
        if self.config.backend in ("cloudy_grid", "cb19", "mappings", "cue", "shock"):
            if self.backend is None:
                raise ValueError(
                    f"NebularSEDComponent(backend={self.config.backend!r}) requires "
                    f"a pre-constructed backend instance via the ``backend`` "
                    f"constructor field (Cue/CloudyGrid/CB19/MAPPINGS/Shock need their "
                    f"weights/grid/abundance configuration which the "
                    f"orchestrator does not know about)."
                )
            return NebularSEDComponentState(
                name=self.name,
                filter_waves=cached_filter_waves,
                filter_trans=cached_filter_trans,
            )
        raise NotImplementedError(f"NebularSEDComponent unknown backend {self.config.backend!r}.")

    def apply(
        self,
        state: ForwardState,
        params: Mapping[str, jnp.ndarray],
        ssp_data: Any | None = None,
        template_data: Any | None = None,
    ) -> ForwardState:
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
        ForwardState
            New state with ``derived["sed_nebular"]`` and
            ``derived["sed_shock"]`` always populated (zeros for the
            non-active branch), and (for non-BakedIn backends)
            ``sed_intrinsic`` updated to include the active emission.
        """
        # NOTE: do not publish ``self.config.backend`` (a Python string)
        # to ``state.derived`` — strings are not JAX leaves and break
        # ``jax.jit`` traces. The backend identity is in
        # ``self.config.backend`` (eager-time inspection only).

        # SEDModel._template_data_for_jit wraps the backend's weights/grid
        # bundle in a namespaced dict ``{"nebular": <bundle>, ...}`` so the
        # JIT closure can thread heterogeneous template payloads (nebular +
        # dust IR + AGN). Backends expect the unwrapped bundle, so peel
        # the ``"nebular"`` slot off here. ``None`` is preserved (no
        # threading active → backend falls back to the closure-captured
        # ``self.weights`` / ``self.grid``).
        if isinstance(template_data, dict) and "nebular" in template_data:
            template_data = template_data["nebular"]

        # Publish ``sed_nebular`` for every backend; publish ``sed_shock`` only
        # on the mutually-exclusive ``backend="shock"`` path (#851). When a
        # photoionized backend composes with a separate :class:`ShockNebular`
        # component, that component owns ``sed_shock``; ``Posterior`` reads it
        # via ``.get(..., zeros)`` so an unset key is harmless. ``baked_in``
        # returns zeros because emission is already in the SSP grid.
        zeros = jnp.zeros_like(state.wave)

        if self.config.backend == "baked_in":
            return state.with_(derived=state.derived.with_(sed_nebular=zeros))

        # Stellar metallicity (absolute log10(Z)) for downstream backends.
        log_z_history = state.derived.get("log_metallicity_history")
        if log_z_history is not None:
            log_z = jnp.asarray(log_z_history)[0]  # present-day value
        else:
            from tengri.parameters.translate import LOG10_ZSUN

            log_z = jnp.asarray(params["met_logzsol"]) + LOG10_ZSUN

        # ── MAPPINGS V shock backend (different parameter set) ────────
        if self.config.backend == "shock":
            # Source the Halpha luminosity normalization: prefer the
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
                line_sigma_kms=jnp.asarray(params.get("neb_eline_sigma_kms", 100.0)),
            )
            # Shock backend: shock contribution is logically separate
            # from photoionized continuum. Publish under ``sed_shock``
            # and zero out ``sed_nebular`` so the legacy Posterior dict
            # decomposition matches: photoionized vs shock are distinct.
            return state.add_intrinsic(nebular_sed).with_(
                derived=state.derived.with_(sed_shock=nebular_sed, sed_nebular=zeros),
            )

        # ── Photoionization backends (Cue + CloudyGrid) ───────────────
        # Cue / CloudyGrid both expect ``neb_logZ_gas`` in **absolute**
        # log10(Z), matching the convention used by ``log_z``. Public
        # params carry it in Z/Zsun, so apply the LOG10_ZSUN offset
        # here (mirrors the legacy ``param_map`` translation in
        # parameters/translate.py).
        from tengri.parameters.translate import LOG10_ZSUN

        _neb_logZ_gas = params.get("neb_logZ_gas")
        if _neb_logZ_gas is not None:
            _neb_logZ_gas = jnp.asarray(_neb_logZ_gas) + LOG10_ZSUN
        common_kwargs = {
            "ssp_wave": state.wave,
            "log_z": log_z,
            "neb_logU": jnp.asarray(params.get("neb_logU", -3.0)),
            "neb_logZ_gas": _neb_logZ_gas,
            "neb_fesc": jnp.asarray(params.get("neb_fesc", 0.0)),
            "neb_fesc_lya": jnp.asarray(params.get("neb_fesc_lya", 0.0)),
            "neb_fdust": jnp.asarray(params.get("neb_fdust", 0.0)),
            # Intrinsic nebular line velocity dispersion → triweight line
            # profile width (Prospector-style). Default 100 km/s.
            "line_sigma_kms": jnp.asarray(params.get("neb_eline_sigma_kms", 100.0)),
        }
        # ── Diffuse-ionized-gas (DIG) mixing (issue #259) ─────────────
        # The DIG component is a second photoionization regime with a
        # lower ionization parameter (log U_DIG = log U_HII + Δlog U,
        # where Δlog U < 0). Linear mass-fraction mix of the two
        # backend evaluations. Python-literal short-circuit on
        # ``dig_frac == 0`` keeps the no-DIG cost at one backend call.
        # Tracked values always pay two calls under JIT.
        _dig_frac = params.get("neb_dig_frac", 0.0)
        _dig_delta_logU = params.get("neb_dig_delta_logU", -1.0)
        _dig_frac_is_zero = isinstance(_dig_frac, (int, float)) and float(_dig_frac) == 0.0
        _dig_kwargs = None  # built on demand only when needed
        if not _dig_frac_is_zero:
            _dig_kwargs = dict(common_kwargs)
            _dig_kwargs["neb_logU"] = common_kwargs["neb_logU"] + jnp.asarray(_dig_delta_logU)

        if self.config.backend == "cue":
            # Forward Cue's full 12-parameter surface. Backend signature
            # accepts ``**neb_params`` so unrecognized keys are ignored
            # safely; we forward every ``ionspec_*`` and ``gas_*`` we
            # find in ``params`` so users who declared them get a fully
            # parameterized Cue prediction.
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
            # Prefer Cue's high-level path (``ssp_weights`` +
            # ``ssp_log_ages_yr``) so the ionizing-spectrum shape and
            # Q_H are both SSP-derived — matches legacy
            # ``predict_line_fluxes`` parity. Fall back to the explicit
            # ``gas_logqion`` shortcut only if upstream did not publish
            # ``age_weights`` (e.g. a chain without StellarSEDComponent).
            age_weights = state.derived.get("age_weights")
            ssp_ages_yr = state.derived.get("ssp_ages_yr")
            if "gas_logqion" in params:
                common_kwargs["gas_logqion"] = jnp.asarray(params["gas_logqion"])
            elif age_weights is not None and ssp_ages_yr is not None:
                common_kwargs["ssp_weights"] = jnp.asarray(age_weights)
                common_kwargs["ssp_log_ages_yr"] = jnp.log10(jnp.asarray(ssp_ages_yr))
            else:
                nion = state.derived.get("nion")
                if nion is not None:
                    common_kwargs["gas_logqion"] = jnp.log10(jnp.maximum(nion, 1.0))
            nebular_sed = self.backend.predict_nebular_sed(
                **common_kwargs, **cue_extras, template_data=template_data
            )
            if _dig_kwargs is not None:
                nebular_sed_dig = self.backend.predict_nebular_sed(
                    **_dig_kwargs, **cue_extras, template_data=template_data
                )
                _f = jnp.asarray(_dig_frac)
                nebular_sed = (1.0 - _f) * nebular_sed + _f * nebular_sed_dig
        else:  # cloudy_grid, cb19, mappings
            ssp_ages_yr = state.derived.get("ssp_ages_yr")
            age_weights = state.derived.get("age_weights")
            if ssp_ages_yr is None or age_weights is None:
                raise ValueError(
                    f"{self.config.backend.upper()} backend requires state.derived['ssp_ages_yr'] "
                    "and state.derived['age_weights'] (CSP mass weights from "
                    "StellarSEDComponent). Build the chain with a stellar "
                    "component upstream."
                )
            ssp_log_ages_yr = jnp.log10(jnp.asarray(ssp_ages_yr))
            ssp_weights = jnp.asarray(age_weights)
            nebular_sed = self.backend.predict_nebular_sed(
                ssp_weights=ssp_weights,
                ssp_log_ages_yr=ssp_log_ages_yr,
                template_data=template_data,
                **common_kwargs,
            )
            if _dig_kwargs is not None:
                nebular_sed_dig = self.backend.predict_nebular_sed(
                    ssp_weights=ssp_weights,
                    ssp_log_ages_yr=ssp_log_ages_yr,
                    template_data=template_data,
                    **_dig_kwargs,
                )
                _f = jnp.asarray(_dig_frac)
                nebular_sed = (1.0 - _f) * nebular_sed + _f * nebular_sed_dig

        # Publish the discrete line catalog (``line_waves`` /
        # ``line_lums``) when the backend supports it. This is what the
        # legacy ``state_to_emission_lines`` bridge consumes.
        if hasattr(self.backend, "predict_nebular_line_luminosities"):
            try:
                if self.config.backend == "cue":
                    # #303: opt into the full Cue catalog (~271 species)
                    # instead of the default 128 CLOUDY/FSPS subset, so
                    # users can read HeII 1640, HeI 10830, [OIII] 4363,
                    # etc. via pred.lines.get(wavelength).
                    cue_cloudyfsps_only = not self.config.cue_full_catalog
                    line_waves, line_lums = self.backend.predict_nebular_line_luminosities(
                        **common_kwargs,
                        **cue_extras,
                        template_data=template_data,
                        cloudyfsps_only=cue_cloudyfsps_only,
                    )
                    if _dig_kwargs is not None:
                        _, line_lums_dig = self.backend.predict_nebular_line_luminosities(
                            **_dig_kwargs,
                            **cue_extras,
                            template_data=template_data,
                            cloudyfsps_only=cue_cloudyfsps_only,
                        )
                        _f = jnp.asarray(_dig_frac)
                        line_lums = (1.0 - _f) * line_lums + _f * line_lums_dig
                else:  # cloudy_grid, cb19, mappings
                    line_waves, line_lums = self.backend.predict_nebular_line_luminosities(
                        ssp_weights=ssp_weights,
                        ssp_log_ages_yr=ssp_log_ages_yr,
                        template_data=template_data,
                        **common_kwargs,
                    )
                    if _dig_kwargs is not None:
                        _, line_lums_dig = self.backend.predict_nebular_line_luminosities(
                            ssp_weights=ssp_weights,
                            ssp_log_ages_yr=ssp_log_ages_yr,
                            template_data=template_data,
                            **_dig_kwargs,
                        )
                        _f = jnp.asarray(_dig_frac)
                        line_lums = (1.0 - _f) * line_lums + _f * line_lums_dig
                # CLAUDE.md contract: vacuum wavelengths throughout.
                #
                # Upstream Cue (yi-jia-li/cue) ships TWO disagreeing files:
                # ``lineList_wav.npy`` (what the network is keyed against —
                # **air** in optical, CLOUDY-default convention) and
                # ``cue_emlines_info.dat`` (newer parallel metadata —
                # vacuum, but in a *different ordering* that does not
                # match the network indices). The Li+2024 paper §2 states
                # vacuum intent but the .npy never got regenerated.
                #
                # ``data/cue_weights.npz`` is built from the .npy because
                # network indexing requires it. We translate at this
                # boundary so internal indexing stays upstream-faithful
                # while user-facing labels honor tengri's vacuum contract.
                #
                # Idempotency: probe the Balmer series. Vacuum and air
                # wavelengths differ by ~1.3-1.8 Å in the optical, so a
                # multi-line consensus is robust to a single near-coincidence
                # or floating-point noise around any one probe value.
                # Hα 6564.61 v / 6562.80 a, Hβ 4862.68 v / 4861.33 a,
                # Hγ 4341.68 v / 4340.47 a.
                if self.config.backend in ("cue", "cloudy_grid", "cb19", "mappings"):
                    # Trace-safe implementation: under a jitted sampler
                    # (NUTS/HMC loss), ``line_waves`` arrives as a Tracer
                    # via the threaded ``template_data``, so numpy
                    # conversion / boolean indexing / Python branches
                    # raise — and the guard below used to swallow that,
                    # silently dropping the line catalog from
                    # ``state.derived`` (joint phot+lines fits then fail
                    # with a misleading "backend did not publish" error).
                    line_waves = jnp.asarray(line_waves)
                    _BALMER_AIR_VAC = (
                        (6562.80, 6564.61),
                        (4861.33, 4862.68),
                        (4340.47, 4341.68),
                    )
                    air_votes = jnp.asarray(0.0)
                    vac_votes = jnp.asarray(0.0)
                    for air_w, vac_w in _BALMER_AIR_VAC:
                        mid = 0.5 * (air_w + vac_w)
                        probe = line_waves[jnp.argmin(jnp.abs(line_waves - mid))]
                        in_band = (probe > air_w - 1.0) & (probe < vac_w + 1.0)
                        is_air = jnp.abs(probe - air_w) < jnp.abs(probe - vac_w)
                        air_votes = air_votes + jnp.where(in_band & is_air, 1.0, 0.0)
                        vac_votes = vac_votes + jnp.where(in_band & ~is_air, 1.0, 0.0)
                    looks_air = (air_votes >= 2.0) & (air_votes > vac_votes)
                    # Ciddor (1996) air→vacuum, jnp inline (mirror of
                    # ``tengri.utils.conversions.air_to_vacuum``, which is
                    # numpy-only and not traceable).
                    sigma = 1e4 / line_waves
                    n_refr = (
                        1.0
                        + 6.4328e-5
                        + 2.94981e-2 / (146.0 - sigma**2)
                        + 2.5540e-4 / (41.0 - sigma**2)
                    )
                    in_optical = (line_waves >= 2000.0) & (line_waves <= 1.0e4)
                    converted = jnp.where(in_optical, line_waves * n_refr, line_waves)
                    line_waves = jnp.where(looks_air, converted, line_waves)

                state = state.with_(
                    derived=state.derived.with_(line_waves=line_waves, line_lums=line_lums)
                )
            except Exception as exc:
                # Backend's line-luminosity path may fail (e.g. when
                # ``cloudyfsps_only=True`` filters everything out).
                # Don't let it block the SED forward pass — but never
                # swallow silently: a dropped publish disables line-flux
                # fitting downstream.
                import warnings

                warnings.warn(
                    f"nebular backend {self.config.backend!r}: discrete "
                    f"line-catalog publish failed ({type(exc).__name__}: {exc}); "
                    "line_waves/line_lums will be absent from state.derived and "
                    "line-flux fitting will not work for this model.",
                    stacklevel=2,
                )

        # Add nebular contribution to sed_intrinsic. Birth-cloud + diffuse
        # dust attenuation of this continuum is the dust component's
        # responsibility: ``DustSEDComponent`` declares ``sed_nebular`` an
        # optional input, so the topological sort runs it AFTER this component
        # and reddens the published ``sed_nebular`` with the young-limit screen.

        # Photoionized path: ``sed_nebular`` carries continuum + lines;
        # shock contribution is zero (this branch is not the shock backend).
        # Phase 3c-3d-neb: filter-integrate nebular SED and publish
        # ``nebular_phot_lnu_precomp`` for consumption by predict_via_precomp.
        derived_overrides = dict(sed_nebular=nebular_sed, sed_shock=zeros)
        if (
            self._state is not None
            and self._state.filter_waves is not None
            and self._state.filter_trans is not None
        ):
            from tengri.observation.photometry import lnu_filter_integral

            z = jnp.asarray(params.get("redshift", 0.0))
            # Filter-integrate nebular_sed directly via ``lnu_filter_integral``
            # (ADR-0016, #398.e). Replaces the previous
            # ``compute_flux_density(..., dl_cm=1) × inv_cosmology`` dance
            # that applied and immediately undid the (1+z)/(4π d_L²)
            # dimming. Publishes the bare filter-integrated rest-frame L_ν.
            nebular_phot_lnu_precomp = jnp.asarray(
                [
                    lnu_filter_integral(nebular_sed, state.wave, fw, ft, redshift=z)
                    for fw, ft in zip(
                        self._state.filter_waves,
                        self._state.filter_trans,
                        strict=False,
                    )
                ]
            )
            derived_overrides["nebular_phot_lnu_precomp"] = nebular_phot_lnu_precomp

        # Spectrum LUT family (SpectrumPrecomp): point-sample the *un-attenuated*
        # rest-frame nebular SED at the pixel wavelengths (a pixel is a single
        # wavelength, so this is exact). ``predict_spectrum_via_precomp`` puts
        # this in the dust-attenuable bucket and reddens it by the young-limit
        # screen (T_bc·T_diff), matching the exact path. Without this family the
        # Cue/CloudyGrid nebular emission was silently dropped from
        # ``predict_spectrum`` under ``approx=SpectrumPrecomp()`` (lines
        # vanished). BakedIn nebular is carried by ``stellar_spec_lnu_precomp``.
        spec_eff = state.derived.get("spec_eff_waves")
        if spec_eff is not None:
            derived_overrides["nebular_spec_lnu_precomp"] = jnp.interp(
                spec_eff, state.wave, nebular_sed
            )

        # Issue #301: ``neb_fesc`` is the fraction of LyC photons that
        # *escape* the HII region (observed unattenuated); the absorbed
        # fraction (1 - fesc) drives the nebular emission. The current
        # code already scales the nebular continuum (Cue cue.py:1656) and
        # lines (cue.py:1214) by (1 - fesc), but the stellar LyC below
        # 912 Å was passed through untouched — overestimating the
        # observed ionizing continuum at fesc < 1 and breaking energy
        # balance against the nebular emission. Attenuate stellar LyC by
        # ``fesc`` here so the SED reflects "stellar LyC × fesc + nebular
        # ∝ (1 − fesc)" globally.
        neb_fesc = jnp.asarray(params.get("neb_fesc", 0.0))
        lyc_mask = state.wave < 912.0
        # Publish the stellar-LyC survival fraction so the two-component dust
        # component — which rebuilds the stellar SED from the *unmasked* per-age
        # ``lnu_age`` cube — can apply the same fesc absorption. Masking only
        # ``sed_intrinsic`` here is enough for the single-screen path (it
        # attenuates ``sed_intrinsic`` directly) but is bypassed by the
        # two-component path, which previously leaked / negated the LyC below
        # 912 Å (#824). ``where(λ<912, fesc, 1)`` -> at fesc=0 the stellar LyC is
        # fully absorbed, matching CIGALE / FSPS / bagpipes.
        lyc_transmission = jnp.where(lyc_mask, neb_fesc, jnp.ones_like(state.wave))
        derived_overrides["lyc_transmission"] = lyc_transmission

        sed_intrinsic = state.sed_intrinsic
        if sed_intrinsic is not None:
            sed_intrinsic = jnp.where(lyc_mask, sed_intrinsic * neb_fesc, sed_intrinsic)

        return state.with_(
            sed_intrinsic=(sed_intrinsic + nebular_sed)
            if sed_intrinsic is not None
            else nebular_sed,
            derived=state.derived.with_(**derived_overrides),
        )


# Register in the unified component dispatch table so build_components resolves
# the nebular component via _resolve_registry_component (single dispatch, #845)
# instead of importing the class directly.
from tengri.components.sed_model_component import _REGISTRY

_REGISTRY["nebular"] = NebularSEDComponent
