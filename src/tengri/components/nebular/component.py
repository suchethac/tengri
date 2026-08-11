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
read the stellar-published ionizing rate — ``log_nion`` for the log-domain
consumers (grid path and Cue fallback), ``nion`` for the deferred erg/s
line paths — and add the resulting line + continuum SED to ``sed_intrinsic``.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from functools import cache
from typing import Any

import jax.numpy as jnp
import numpy as np

from tengri.components.nebular._constants import _LSUN_ERG
from tengri.components.nebular.baked_in import BakedInBackend
from tengri.components.template_threading import TemplateThreading
from tengri.parameters.priors import Fixed, Uniform
from tengri.parameters.resolve import require_redshift
from tengri.protocols.component import (
    DerivedKey,
    ForwardState,
    ParamDeclaration,
    SEDComponentConfig,
    SEDComponentState,
)
from tengri.utils.scale import log10_magnitude

__all__ = ["NebularSEDComponent", "NebularSEDComponentConfig"]

#: Nebular parameters only some photoionization backends model. CB19 carries
#: them as three of its six interpolation axes; CLOUDY, Cue and the baked-in
#: backend have no such axes. They are threaded per backend by
#: :func:`_backend_accepted_params` rather than added to the shared kwargs
#: unconditionally.
_BACKEND_OPTIONAL_PARAMS: tuple[str, ...] = ("neb_log_nH", "neb_co", "neb_dno")

#: Backend methods the shared kwargs dict is splatted into. A parameter is
#: threaded only when *every* method that exists names it, so a backend that
#: models an axis in one call and not the other is never handed a value one of
#: them would drop.
_BACKEND_KWARG_SINKS: tuple[str, ...] = (
    "predict_nebular_sed",
    "predict_nebular_line_luminosities",
)


@cache
def _backend_accepted_params(backend_cls: type) -> frozenset[str]:
    """Which :data:`_BACKEND_OPTIONAL_PARAMS` this backend class names.

    Parameters
    ----------
    backend_cls : type
        Backend class (not instance), so the result caches per class.

    Returns
    -------
    frozenset of str
        Subset of :data:`_BACKEND_OPTIONAL_PARAMS` named by every method in
        :data:`_BACKEND_KWARG_SINKS` the class defines.

    Notes
    -----
    **JIT-compatible**: no — signature introspection, cached per class and
    evaluated at apply time before entering any JAX transform.

    Every backend's ``predict_nebular_sed`` ends in ``**kwargs``, so passing an
    unmodeled parameter raises nothing — it is silently dropped, which is
    indistinguishable from being read. The named parameters are therefore the
    only honest test of whether a backend models an axis.

    This is the general form of a defect measured on CB19: the component built
    one shared kwargs dict that never contained ``neb_log_nH``, ``neb_co`` or
    ``neb_dno``, so the backend received its signature defaults on every call
    while the sampler was free to propose values. Sweeping each across its full
    declared support moved the SED by exactly 0.0 — three free dimensions that
    could not affect the fit, on a backend whose own docstring advertises being
    differentiable through them.
    """
    accepted = set(_BACKEND_OPTIONAL_PARAMS)
    for method_name in _BACKEND_KWARG_SINKS:
        method = getattr(backend_cls, method_name, None)
        if method is None:
            continue
        try:
            named = set(inspect.signature(method).parameters)
        except (TypeError, ValueError):  # pragma: no cover - C-implemented callables
            return frozenset()
        accepted &= named
    return frozenset(accepted)


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
class NebularSEDComponent(TemplateThreading):
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
    #: Optional per-Q_H nebular grid (:class:`NebularGridTable`). When attached
    #: (via :meth:`SEDModel.enable_fast_nebular`) and it carries a photometry
    #: channel, :meth:`apply` reconstructs ``nebular_phot_lnu_precomp`` as
    #: ``Q_H x interp(grid)`` instead of the per-eval filter integration — which
    #: makes the Cue forward dead for the photometry channel (XLA prunes it).
    #: Typed loosely to avoid an import cycle; it is a ``NebularGridTable``.
    grid_table: Any | None = None
    #: ``True`` when the continuum must be computed rather than zeroed, which
    #: forbids the grid's photometry shortcut: serving photometry from the grid
    #: requires zeroing the continuum, and the dust energy balance needs it to
    #: size the absorbed budget. Set from the assembled chain by
    #: :meth:`SEDModel.enable_fast_nebular` — derived, never asserted.
    #:
    #: Named for what the component must *do*, not for the one key that drives
    #: it today: ``sed_shock`` has the same exposure and would share this gate.
    #:
    #: The census reads the component contract, so it cannot see a reader that
    #: takes a published key off ``state.derived`` without declaring an input —
    #: ``state_to_sed_components`` and the accumulated ``state.sed_intrinsic``
    #: are both such readers. They ask :meth:`materialized` for a publishing
    #: variant instead of being enumerated here (#1673).
    must_materialize_sed: bool = False
    # Tuple prefix so the MAPPINGS shock backend (``shock_*``) and the
    # photoionization backends (``neb_*``, ``ionspec_*``, ``gas_*``) all
    # flow through the standard prefix-stripping path. Backends silently
    # ignore keys they don't consume — passing ``shock_*`` to Cue is
    # harmless, and vice versa.
    parameter_prefix: tuple[str, ...] = ("neb_", "shock_", "ionspec_", "gas_")

    def materialized(self) -> NebularSEDComponent:
        """Variant that computes the nebular continuum instead of zeroing it.

        The publication hook :func:`~tengri.forward.orchestrator.materialized_chain`
        asks for. Serving photometry from the per-Q_H grid requires zeroing
        ``sed_nebular``, so any caller that reads the forward state itself needs
        this variant rather than the one the observables kernel runs (#1673).

        Returns
        -------
        NebularSEDComponent
            ``self`` when the continuum is already materialized, else a copy
            with :attr:`must_materialize_sed` set. Returning ``self`` unchanged
            keeps the chain's identity stable for callers that cache on it.

        Notes
        -----
        **JIT-compatible**: not applicable — composition-time only.
        """
        if self.must_materialize_sed:
            return self
        return replace(self, must_materialize_sed=True)

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
            DerivedKey("log_line_lums", "dex", "log10(line luminosities / (erg/s)); float32-safe"),
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

    def _grid_interp_point(self, grid, params, state):
        """Assemble the ionization interp point for :attr:`grid_table`.

        ``neb_logU`` / ``neb_logZ_gas`` are in the (prefix-sliced) ``params`` this
        component sees; ``met_logzsol`` is **not** (the stellar component owns it),
        so it is sourced from the stellar-published absolute metallicity history
        and converted back to relative ``log10(Z/Zsun)`` — the units the grid axis
        was built in. Returns a dict keyed by ``grid.axis_names`` for
        :func:`reconstruct_nebular_phot`.
        """
        point = {}
        for name in grid.axis_names:
            if name == "met_logzsol":
                from tengri.parameters.translate import LOG10_ZSUN

                log_z_hist = state.derived.get("log_metallicity_history")
                point[name] = jnp.asarray(log_z_hist)[0] - LOG10_ZSUN
            else:
                point[name] = jnp.asarray(params[name])
        return point

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
        # Axes only some backends model (CB19's log_nH / log_CO / dNO). Passed
        # only to a backend that names them; a value absent from ``params``
        # falls through to the backend's own signature default, which matches
        # the registry default for each, so the two cannot disagree.
        for _name in _backend_accepted_params(type(self.backend)):
            if _name in params:
                common_kwargs[_name] = jnp.asarray(params[_name])
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

        # FAST path (#950): with a per-Q_H grid attached, the nebular photometry
        # (this apply) and the emission lines (predict_line_fluxes) reconstruct
        # from the grid, so the expensive Cue continuum + line-catalog forwards
        # are not needed. Skipping them zeroes ``nebular_sed``, and the Cue NN
        # forward is genuinely gone (not merely pruned), which is where the
        # ~1.2 ms/eval saving comes from.
        #
        # That trade is only available when nothing downstream reads the
        # continuum. It used to be taken unconditionally, on the stated grounds
        # that the only live consumers were the exact spectrum / dust-continuum
        # paths — but the dust energy balance reads ``sed_nebular`` to size the
        # absorbed budget, so a model with dust emission re-emitted the stellar
        # half alone: 11 % low in the far-IR, gradient up to 380 % wrong, in
        # float64 and silently. ``must_materialize_sed`` is derived from the
        # assembled chain, so a future consumer disables the shortcut by
        # declaring the input, with nothing to keep in sync here — for
        # consumers that go through the contract. A reader that takes the
        # published key without declaring it stays invisible here (#1673).
        #
        # One flag, used for BOTH the continuum and the photometry publication
        # below: two checks of the same condition are free to drift apart, and
        # this defect is what that costs.
        use_grid = (
            self.grid_table is not None
            and getattr(self.grid_table, "log_phot_per_qh", None) is not None
            and not self.must_materialize_sed
        )

        if use_grid:
            nebular_sed = zeros
        elif self.config.backend == "cue":
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
                log_nion = state.derived.get("log_nion")
                if log_nion is not None:
                    common_kwargs["gas_logqion"] = jnp.maximum(log_nion, 0.0)
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
        # legacy ``state_to_emission_lines`` bridge consumes. Skipped on the
        # fast grid path — predict_line_fluxes reconstructs lines from the grid,
        # so the Cue line-catalog forward is not run.
        if not use_grid and hasattr(self.backend, "predict_nebular_line_luminosities"):
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
                    # Edlén (1953) air→vacuum, jnp inline (mirror of
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

                # THE unit seam (#1559). Every backend returns [Lsun]; the
                # published ``line_lums`` DerivedKey is [erg/s], and
                # ``predict_line_fluxes`` divides by 4 pi d_L^2 with no further
                # conversion. This one multiply is what reconciles them.
                #
                # It used to live inside CueBackend and nowhere else, so Cue
                # came out right and CloudyGrid / CB19 / MappingsPhoto came out
                # a factor L_sun too faint — invisible to every per-backend
                # test, because a global scale cancels in the line ratios and
                # monotonicity checks they all use.
                #
                # The constant is the *backend's*, not a global. Cue's network
                # was trained on L_sun = 3.839e33 and the grid backends are
                # tabulated against IAU 3.828e33; using one value for both puts
                # a systematic 0.287% on whichever backend it does not belong
                # to. That is far too small for the units test to catch, so it
                # would have shipped.
                lsun_erg = getattr(self.backend, "lsun_erg", _LSUN_ERG)

                # The log companion (#1534) is taken on the [Lsun] side, before
                # the multiply that leaves the float32 window: taken after, the
                # value is already ``inf`` and its log carries nothing.
                # ``log10_magnitude`` keeps a dark line (``-inf``) distinct from
                # a corrupt one (``+inf``), #1527.
                log_line_lums = log10_magnitude(line_lums) + float(np.log10(lsun_erg))
                state = state.with_(
                    derived=state.derived.with_(
                        line_waves=line_waves,
                        line_lums=line_lums * lsun_erg,
                        log_line_lums=log_line_lums,
                    )
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
        # Filter-integrate the nebular SED and publish
        # ``nebular_phot_lnu_precomp`` for consumption by predict_via_precomp.
        derived_overrides = dict(sed_nebular=nebular_sed, sed_shock=zeros)
        grid = self.grid_table
        # ``use_grid``, not a second re-derivation of it: when a downstream
        # consumer forces the exact continuum above, photometry must come from
        # the filter integration below, or the nebular light would be counted
        # once in ``sed_nebular`` and again from the grid.
        if use_grid:
            # FAST path (#950): reconstruct the intrinsic nebular photometry from
            # the per-Q_H grid, ``L_nu = 10^{log_nion + interp(grid)}``, instead of the
            # per-eval filter integration below. The downstream contract is
            # identical — ``predict_via_precomp`` applies the young-limit dust
            # screen (at the filter level) and the cosmology dimming to this same
            # ``nebular_phot_lnu_precomp`` key, so only the intrinsic-L_nu source
            # changes. ``nebular_sed`` (the Cue continuum) is now unread by the
            # photometry channel, so XLA prunes the Cue forward from
            # ``predict_photometry``. log10(Q_H) is the stellar-published ``log_nion``.
            from tengri.components.nebular.nebular_grid_precompute import (
                reconstruct_nebular_phot,
            )

            log_nion = state.derived["log_nion"]
            derived_overrides["nebular_phot_lnu_precomp"] = reconstruct_nebular_phot(
                log_nion, self._grid_interp_point(grid, params, state), grid
            )
        elif (
            self._state is not None
            and self._state.filter_waves is not None
            and self._state.filter_trans is not None
        ):
            from tengri.observation.photometry import lnu_filter_integral

            z = jnp.asarray(require_redshift(params, "components.nebular.component.apply"))
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
            # The REST band (#1148). ``phot_rest_fnu`` is the SED reprojected at
            # z=0, so the filter sits in the REST frame and samples the rest SED at
            # its own pivot — the SAME integral with redshift=0, not the observed-band
            # value reused. Reusing it is what made the LUT report a different
            # physical quantity from the exact path (769 % in des_g at z=0.5).
            derived_overrides["nebular_restband_lnu_precomp"] = jnp.asarray(
                [
                    lnu_filter_integral(nebular_sed, state.wave, fw, ft, redshift=0.0)
                    for fw, ft in zip(
                        self._state.filter_waves,
                        self._state.filter_trans,
                        strict=False,
                    )
                ]
            )

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


# ─────────────────────────────────────────────────────────────────────
# Lines group property registration (Phase 1B)
# ─────────────────────────────────────────────────────────────────────


def _line_ratio_floor() -> float:
    """The guard floor these line ratios divide by, made representable (#1568).

    ``1e-300`` is far below float32's smallest subnormal (1.4e-45), so in
    float32 every ``jnp.maximum(x, _line_ratio_floor())`` below was
    ``jnp.maximum(x, 0.0)`` — the floor did not clamp, and a dark line divided
    by another dark line gave ``0/0 = NaN`` rather than the finite ratio the
    clamp exists to produce.

    Evaluated at trace time, not import time: ``representable_floor`` resolves
    against the *working* dtype, so a module-level constant computed once at
    import would be pinned to whichever dtype happened to be active then.
    float64 keeps ``1e-300`` exactly.
    """
    from tengri.utils.scale import representable_floor

    return representable_floor(1e-300)


def _line_luminosity_helper(state, params, line_key):
    """Helper to extract a single line luminosity from the line catalog."""
    from tengri.utils.sed_quantities import KEY_LINES, extract_line_luminosity

    derived = state.derived
    nan_scalar = jnp.asarray(jnp.nan)

    if "line_waves" not in derived or "line_lums" not in derived:
        return nan_scalar

    line_waves = jnp.asarray(derived["line_waves"])
    line_lums = jnp.asarray(derived["line_lums"])
    return extract_line_luminosity(line_waves, line_lums, KEY_LINES[line_key])


def _lya_fn(state, params):
    """Lyman alpha line luminosity [erg/s]."""
    return _line_luminosity_helper(state, params, "lya")


def _civ_1549_fn(state, params):
    """CIV 1549 line luminosity [erg/s]."""
    return _line_luminosity_helper(state, params, "civ_1549")


def _oii_fn(state, params):
    """OII line luminosity [erg/s]."""
    return _line_luminosity_helper(state, params, "oii")


def _hbeta_fn(state, params):
    """Hβ line luminosity [erg/s]."""
    return _line_luminosity_helper(state, params, "hbeta")


def _oiii_4959_fn(state, params):
    """OIII 4959 line luminosity [erg/s]."""
    return _line_luminosity_helper(state, params, "oiii_4959")


def _oiii_5007_fn(state, params):
    """OIII 5007 line luminosity [erg/s]."""
    return _line_luminosity_helper(state, params, "oiii_5007")


def _nii_6548_fn(state, params):
    """NII 6548 line luminosity [erg/s]."""
    return _line_luminosity_helper(state, params, "nii_6548")


def _halpha_fn(state, params):
    """Hα line luminosity [erg/s]."""
    return _line_luminosity_helper(state, params, "halpha")


def _nii_6584_fn(state, params):
    """NII 6584 line luminosity [erg/s]."""
    return _line_luminosity_helper(state, params, "nii_6584")


def _sii_6717_fn(state, params):
    """SII 6717 line luminosity [erg/s]."""
    return _line_luminosity_helper(state, params, "sii_6717")


def _sii_6731_fn(state, params):
    """SII 6731 line luminosity [erg/s]."""
    return _line_luminosity_helper(state, params, "sii_6731")


def _bpt_nii_fn(state, params):
    """BPT-NII diagnostic: log10([NII]6584 / Hα) [dex]."""
    derived = state.derived
    nan_scalar = jnp.asarray(jnp.nan)

    if "line_waves" not in derived or "line_lums" not in derived:
        return nan_scalar

    from tengri.utils.sed_quantities import KEY_LINES, extract_line_luminosity

    line_waves = jnp.asarray(derived["line_waves"])
    line_lums = jnp.asarray(derived["line_lums"])
    nii_6584 = extract_line_luminosity(line_waves, line_lums, KEY_LINES["nii_6584"])
    halpha = extract_line_luminosity(line_waves, line_lums, KEY_LINES["halpha"])
    return jnp.log10(
        jnp.maximum(nii_6584, _line_ratio_floor()) / jnp.maximum(halpha, _line_ratio_floor())
    )


def _bpt_sii_fn(state, params):
    """BPT-SII diagnostic: log10(([SII]6717+6731) / Hα) [dex]."""
    derived = state.derived
    nan_scalar = jnp.asarray(jnp.nan)

    if "line_waves" not in derived or "line_lums" not in derived:
        return nan_scalar

    from tengri.utils.sed_quantities import KEY_LINES, extract_line_luminosity

    line_waves = jnp.asarray(derived["line_waves"])
    line_lums = jnp.asarray(derived["line_lums"])
    sii_6717 = extract_line_luminosity(line_waves, line_lums, KEY_LINES["sii_6717"])
    sii_6731 = extract_line_luminosity(line_waves, line_lums, KEY_LINES["sii_6731"])
    halpha = extract_line_luminosity(line_waves, line_lums, KEY_LINES["halpha"])
    sii_total = sii_6717 + sii_6731
    return jnp.log10(
        jnp.maximum(sii_total, _line_ratio_floor()) / jnp.maximum(halpha, _line_ratio_floor())
    )


def _o3hb_fn(state, params):
    """[OIII]5007/Hβ diagnostic: log10([OIII]5007 / Hβ) [dex]."""
    derived = state.derived
    nan_scalar = jnp.asarray(jnp.nan)

    if "line_waves" not in derived or "line_lums" not in derived:
        return nan_scalar

    from tengri.utils.sed_quantities import KEY_LINES, extract_line_luminosity

    line_waves = jnp.asarray(derived["line_waves"])
    line_lums = jnp.asarray(derived["line_lums"])
    oiii_5007 = extract_line_luminosity(line_waves, line_lums, KEY_LINES["oiii_5007"])
    hbeta = extract_line_luminosity(line_waves, line_lums, KEY_LINES["hbeta"])
    return jnp.log10(
        jnp.maximum(oiii_5007, _line_ratio_floor()) / jnp.maximum(hbeta, _line_ratio_floor())
    )


def _r23_fn(state, params):
    """R23 metallicity indicator: log10(([OII]+[OIII]4959+5007)/Hβ) [dex]."""
    derived = state.derived
    nan_scalar = jnp.asarray(jnp.nan)

    if "line_waves" not in derived or "line_lums" not in derived:
        return nan_scalar

    from tengri.utils.sed_quantities import KEY_LINES, extract_line_luminosity

    line_waves = jnp.asarray(derived["line_waves"])
    line_lums = jnp.asarray(derived["line_lums"])
    oii = extract_line_luminosity(line_waves, line_lums, KEY_LINES["oii"])
    oiii_4959 = extract_line_luminosity(line_waves, line_lums, KEY_LINES["oiii_4959"])
    oiii_5007 = extract_line_luminosity(line_waves, line_lums, KEY_LINES["oiii_5007"])
    hbeta = extract_line_luminosity(line_waves, line_lums, KEY_LINES["hbeta"])
    numerator = oii + oiii_4959 + oiii_5007
    return jnp.log10(
        jnp.maximum(numerator, _line_ratio_floor()) / jnp.maximum(hbeta, _line_ratio_floor())
    )


def _o32_fn(state, params):
    """O32 ionization parameter: log10([OIII]5007 / [OII]) [dex]."""
    derived = state.derived
    nan_scalar = jnp.asarray(jnp.nan)

    if "line_waves" not in derived or "line_lums" not in derived:
        return nan_scalar

    from tengri.utils.sed_quantities import KEY_LINES, extract_line_luminosity

    line_waves = jnp.asarray(derived["line_waves"])
    line_lums = jnp.asarray(derived["line_lums"])
    oiii_5007 = extract_line_luminosity(line_waves, line_lums, KEY_LINES["oiii_5007"])
    oii = extract_line_luminosity(line_waves, line_lums, KEY_LINES["oii"])
    return jnp.log10(
        jnp.maximum(oiii_5007, _line_ratio_floor()) / jnp.maximum(oii, _line_ratio_floor())
    )


def _balmer_decrement_fn(state, params):
    """Balmer decrement: Hα/Hβ [dimensionless]."""
    derived = state.derived
    nan_scalar = jnp.asarray(jnp.nan)

    if "line_waves" not in derived or "line_lums" not in derived:
        return nan_scalar

    from tengri.utils.sed_quantities import KEY_LINES, extract_line_luminosity

    line_waves = jnp.asarray(derived["line_waves"])
    line_lums = jnp.asarray(derived["line_lums"])
    halpha = extract_line_luminosity(line_waves, line_lums, KEY_LINES["halpha"])
    hbeta = extract_line_luminosity(line_waves, line_lums, KEY_LINES["hbeta"])
    return halpha / jnp.maximum(hbeta, _line_ratio_floor())


from tengri.forward.properties import Property, register_properties

_LINES_PROPERTIES = {
    "lya": Property(
        units="erg/s",
        group="lines",
        doc="Lyman alpha line luminosity",
        fn=_lya_fn,
    ),
    "civ_1549": Property(
        units="erg/s",
        group="lines",
        doc="CIV 1549 line luminosity",
        fn=_civ_1549_fn,
    ),
    "oii": Property(
        units="erg/s",
        group="lines",
        doc="OII line luminosity",
        fn=_oii_fn,
    ),
    "hbeta": Property(
        units="erg/s",
        group="lines",
        doc="Hβ line luminosity",
        fn=_hbeta_fn,
    ),
    "oiii_4959": Property(
        units="erg/s",
        group="lines",
        doc="OIII 4959 line luminosity",
        fn=_oiii_4959_fn,
    ),
    "oiii_5007": Property(
        units="erg/s",
        group="lines",
        doc="OIII 5007 line luminosity",
        fn=_oiii_5007_fn,
    ),
    "nii_6548": Property(
        units="erg/s",
        group="lines",
        doc="NII 6548 line luminosity",
        fn=_nii_6548_fn,
    ),
    "halpha": Property(
        units="erg/s",
        group="lines",
        doc="Hα line luminosity",
        fn=_halpha_fn,
    ),
    "nii_6584": Property(
        units="erg/s",
        group="lines",
        doc="NII 6584 line luminosity",
        fn=_nii_6584_fn,
    ),
    "sii_6717": Property(
        units="erg/s",
        group="lines",
        doc="SII 6717 line luminosity",
        fn=_sii_6717_fn,
    ),
    "sii_6731": Property(
        units="erg/s",
        group="lines",
        doc="SII 6731 line luminosity",
        fn=_sii_6731_fn,
    ),
    "bpt_nii": Property(
        units="dex",
        group="lines",
        doc="BPT-NII diagnostic: log10([NII]6584 / Hα)",
        fn=_bpt_nii_fn,
    ),
    "bpt_sii": Property(
        units="dex",
        group="lines",
        doc="BPT-SII diagnostic: log10(([SII]6717+6731) / Hα)",
        fn=_bpt_sii_fn,
    ),
    "o3hb": Property(
        units="dex",
        group="lines",
        doc="[OIII]5007/Hβ diagnostic: log10([OIII]5007 / Hβ)",
        fn=_o3hb_fn,
    ),
    "r23": Property(
        units="dex",
        group="lines",
        doc="R23 metallicity indicator: log10(([OII]+[OIII]4959+5007)/Hβ)",
        fn=_r23_fn,
    ),
    "o32": Property(
        units="dex",
        group="lines",
        doc="O32 ionization parameter: log10([OIII]5007 / [OII])",
        fn=_o32_fn,
    ),
    "balmer_decrement": Property(
        units="",
        group="lines",
        doc="Balmer decrement: Hα/Hβ",
        fn=_balmer_decrement_fn,
    ),
}

register_properties("nebular", _LINES_PROPERTIES)

del Property, register_properties, _LINES_PROPERTIES


# Register in the unified component dispatch table so build_components resolves
# the nebular component via _resolve_registry_component (single dispatch, #845)
# instead of importing the class directly.
from tengri.components.sed_model_component import _REGISTRY

_REGISTRY["nebular"] = NebularSEDComponent
