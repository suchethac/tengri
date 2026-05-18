"""SEDComponent protocol: the contract each physics module satisfies.

A component owns one block of the forward model — stellar emission,
dust attenuation+emission, nebular lines, AGN, IGM, radio, X-ray —
plus the parameters and precomputed tensors that go with it. The
:class:`tengri.SEDModel` orchestrator runs components in order, threading
a :class:`PipelineState` through them.

This module is part of the Part II-1 scaffold. Nothing in `tengri`
consumes these classes yet; they exist so future phases (II-2 onwards)
can migrate one component at a time onto a stable interface.

Notes
-----
**JIT-compatible:** :meth:`SEDComponent.apply` MUST be pure JAX so it
can be JIT/grad/vmap-compiled by the orchestrator. :meth:`precompute`
runs *before* tracing and may use eager numpy/file I/O.

**Immutability:** :class:`PipelineState` and :class:`SEDComponentState`
are frozen dataclasses. Components return *new* states rather than
mutating their inputs. This matches the global coding rule
(`~/.claude/rules/common/coding-style.md`) and is what makes
``jax.lax.scan`` over a list of components possible.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, NamedTuple, Protocol, runtime_checkable

import jax.numpy as jnp

__all__ = [
    "BARE_NAME_ALLOWLIST",
    "DerivedKey",
    "ParamDeclaration",
    "PipelineContractError",
    "PipelineState",
    "SEDComponent",
    "SEDComponentConfig",
    "SEDComponentState",
]


class PipelineContractError(ValueError):
    """Raised by :func:`tengri.forward.orchestrator.validate_pipeline` when
    publish/require constraints fail at component-list construction time.

    The contract is checked once at :class:`tengri.SEDModel` construction,
    before any JIT compile. A raised ``PipelineContractError`` always names
    the offending component class and the offending derived key, plus a
    "Did you mean: ..." suggestion when a missing-publisher likely indicates
    a typo. See :func:`validate_pipeline` for the full list of checks.
    """


# ─────────────────────────────────────────────────────────────────────
# Contract decisions resolved by the Phase II-1 first-adapter pass
# (RadioSEDComponent + IGMSEDComponent, 2026-05).
# ─────────────────────────────────────────────────────────────────────

#: Parameter names that the orchestrator passes to *every* component
#: regardless of ``parameter_prefix``. Today this is just ``redshift``,
#: which is read by IGM, radio, X-ray, and the observation model.
#: Extend with care — every entry here weakens the prefix discipline
#: enforced by ``tools/check_param_prefixes.py``.
BARE_NAME_ALLOWLIST: tuple[str, ...] = ("redshift",)


class ParamDeclaration(NamedTuple):
    """One free parameter a component declares it owns.

    Returned in a list from :meth:`SEDComponent.declared_parameters`. The
    orchestrator (or :class:`tengri.Parameters`) consumes this to build
    the prior dict; today the test suite asserts shape + prefix
    invariants only.

    Fields
    ------
    name : str
        Parameter name. Must start with the component's
        :attr:`SEDComponent.parameter_prefix` *or* be in
        :data:`BARE_NAME_ALLOWLIST`.
    prior : Any
        A :class:`tengri.parameters.priors.Distribution` (Uniform,
        Gaussian, Fixed, …). Typed loosely as ``Any`` here to avoid an
        import cycle between :mod:`tengri.core` and
        :mod:`tengri.parameters`.
    description : str
        One-line human-readable description, mirrored into the prior
        registry when a component's parameters are registered.
    bound_check : Any, optional
        Callable ``(lo, hi) -> bool`` invoked when a user narrows the
        prior at construction time. Returns ``True`` for an admissible
        bound. ``None`` (default) means no check. Typed loosely as
        ``Any`` for the same import-cycle reason as :attr:`prior`.
    bound_error : str
        Human-readable message attached to a :class:`ValueError` when
        :attr:`bound_check` rejects a bound. Empty string means the
        default generic message is used.
    """

    name: str
    prior: Any
    description: str = ""
    bound_check: Any = None
    bound_error: str = ""


class DerivedKey(NamedTuple):
    """One cross-component datum published into :attr:`PipelineState.derived`.

    Returned in a tuple from :meth:`SEDComponent.publishes` and
    :meth:`SEDComponent.requires`. Consumed once at pipeline construction
    by :func:`tengri.forward.orchestrator.validate_pipeline` to ensure
    every required key has an upstream publisher with matching units, and
    (when needed) to derive a topological ordering over components.

    Mirrors :class:`ParamDeclaration` line-for-line so future readers see
    the same shape on both sides of the cross-component contract.

    Fields
    ------
    name : str
        Stable key written to / read from :attr:`PipelineState.derived`.
        Compared by string equality, so renames are caught at construction.
    units : str
        Free-form units tag, e.g. ``"erg/s"``, ``"Msun/yr"``, ``"dex"``,
        ``"erg/s/Hz"``, ``"yr"``, ``"photons/s"``. The validator compares
        publisher and consumer units strings for equality; the
        ``_CANONICAL_UNITS`` table in
        :mod:`tengri.forward.orchestrator` pins the expected string for
        every well-known key. Use ``""`` for unitless quantities (e.g.
        transmission factors, mass weights, attenuation factors).
    description : str
        One-line human-readable description shown in error messages and
        by future introspection helpers. Optional.

    Notes
    -----
    No numeric unit conversion is attempted by the validator: the
    contract refuses to silently paper over a unit disagreement, but it
    will not convert ``"Lsun"`` to ``"erg/s"`` for the consumer. Each
    component is responsible for converting at its own boundary.

    The strings are deliberately *not* :mod:`astropy.units` quantities —
    that path is JIT-incompatible and would cost a 5–100× performance
    hit on the hot pipeline. Stringly-typed-with-a-canonical-table is
    enough to catch the realistic failure modes (`L_SUN` vs `L_SUN_CUE`
    confusion, ``"Lsun"`` vs ``"erg/s"`` cross-talk).
    """

    name: str
    units: str
    description: str = ""


# ─────────────────────────────────────────────────────────────────────
# Frozen state types
# ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SEDComponentConfig:
    """Frozen structural settings for a component.

    Marker base class. Concrete components subclass this with their own
    knobs (e.g. ``DustSEDComponentConfig`` carries
    ``attenuation_law: str``, ``emission_model: str``, …).

    Crucially, fields here are NOT JAX traced values — they configure
    the *shape* of the computation, not its inputs.
    """

    name: str = "component"


@dataclass(frozen=True)
class SEDComponentState:
    """Output of :meth:`SEDComponent.precompute`.

    Carries cached static tensors a component needs every call (template
    grids, age masks, filter convolution matrices). The orchestrator
    builds this once at compile time and reuses it across every
    likelihood evaluation.

    Concrete components subclass with typed fields::

        @dataclass(frozen=True)
        class DustEmissionState(SEDComponentState):
            template_grid: jnp.ndarray  # shape (n_T, n_wave)
            log_T_axis: jnp.ndarray  # shape (n_T,)
    """

    name: str = "component"


@dataclass(frozen=True)
class PipelineState:
    """Threaded state passed through a chain of :class:`SEDComponent`s.

    Each component reads what it needs and returns a new ``PipelineState``
    (immutable). Field semantics match :mod:`tengri.forward.prediction`:

    - ``wave``               : rest-frame wavelength grid in Å
    - ``sed_intrinsic``      : pre-attenuation rest-frame L_nu in erg/s/Hz
    - ``sed_attenuated``     : post-attenuation rest-frame L_nu in erg/s/Hz
    - ``sed_observed``       : observer-frame F_nu (after IGM, redshift,
                               and units conversion)
    - ``lines``              : optional dict of emission-line spectra
    - ``derived``            : free-form dict for diagnostics
                               (per-component intermediate L_nu, taus, etc.)

    All array fields use ``jnp.ndarray`` so the whole state can be a
    pytree leaf for ``jax.lax.scan`` / ``jax.tree_map``.

    Cross-component reads
    ---------------------
    A component MUST NOT read another component's free parameters
    directly. When a downstream component needs a quantity computed by
    an upstream component (e.g. radio reads ``L_ir`` produced by dust),
    the upstream component publishes it under a stable key in
    :attr:`derived` and the downstream component reads from there with
    a documented fallback. The first adapter pair (radio + IGM)
    establishes this convention; see ``RadioSEDComponent.apply`` for
    the canonical fallback pattern.
    """

    wave: jnp.ndarray
    sed_intrinsic: jnp.ndarray | None = None
    sed_attenuated: jnp.ndarray | None = None
    sed_observed: jnp.ndarray | None = None
    lines: Mapping[str, jnp.ndarray] | None = None
    derived: Mapping[str, Any] = field(default_factory=dict)

    def with_(self, **overrides: Any) -> PipelineState:
        """Return a copy of this state with selected fields replaced.

        Convenience for components::

            new_state = state.with_(sed_attenuated=tau_corrected)

        Equivalent to ``dataclasses.replace`` but reads better at call
        sites.
        """
        from dataclasses import replace

        return replace(self, **overrides)


# ─────────────────────────────────────────────────────────────────────
# JAX pytree registration
# ─────────────────────────────────────────────────────────────────────
#
# ``PipelineState`` and ``SEDComponentState`` are threaded through
# JIT-compiled component pipelines, so they must be JAX pytrees. All
# fields are dynamic (data, not static metadata) — even the ``Mapping``
# fields ``lines`` and ``derived`` are dicts of arrays that JAX's
# default dict-pytree handler unpacks recursively.
#
# Registration uses :func:`jax.tree_util.register_dataclass` (JAX ≥ 0.4),
# which preserves the frozen-dataclass nature and integrates with
# :func:`dataclasses.replace`-based mutation in :meth:`PipelineState.with_`.

from jax import tree_util as _tree_util

_tree_util.register_dataclass(
    PipelineState,
    data_fields=("wave", "sed_intrinsic", "sed_attenuated", "sed_observed", "lines", "derived"),
    meta_fields=(),
)
_tree_util.register_dataclass(
    SEDComponentState,
    data_fields=(),
    meta_fields=("name",),
)
_tree_util.register_dataclass(
    SEDComponentConfig,
    data_fields=(),
    meta_fields=("name",),
)

del _tree_util


# ─────────────────────────────────────────────────────────────────────
# The Protocol
# ─────────────────────────────────────────────────────────────────────


@runtime_checkable
class SEDComponent(Protocol):
    """Contract for one block of the SED forward model.

    Concrete subclasses (``StellarSEDComponent``, ``DustSEDComponent``,
    ``NebularSEDComponent``, ``AGNSEDComponent``, ``IGMSEDComponent``,
    ``RadioSEDComponent``, ``XRaySEDComponent``) live in
    :mod:`tengri.components.<domain>.component` (added in Phases II-2
    through II-4 of the migration).

    Required attributes
    -------------------
    name : str
        Stable identifier for diagnostics and registry lookup. Examples:
        ``"stellar"``, ``"dust"``, ``"nebular"``, ``"agn"``, ``"igm"``.

    parameter_prefix : str
        Domain prefix from NAMING_CONTRACT §3.2 — every parameter this
        component reads must start with this prefix. Used by the
        orchestrator to slice the global ``params`` dict before
        :meth:`apply`. Examples: ``"sfh_"`` (stellar), ``"dust_"``,
        ``"neb_"``, ``"agn_"``, ``"igm_"``, ``"radio_"``, ``"xray_"``.
        Must be non-empty. Bare names that several components share
        (e.g. ``redshift``) are passed via
        :data:`BARE_NAME_ALLOWLIST`, not via an empty prefix.

    config : SEDComponentConfig
        The frozen knobs that configure the *shape* of this
        component's computation. Held as a Python attribute, not a
        traced value.

    Required methods
    ----------------
    declared_parameters() -> list[ParamDeclaration]
        What free parameters this component owns. Consumed by
        :class:`tengri.Parameters` builder so users don't have to
        register them by hand.

    precompute(ssp_data=None, wave_grid=None) -> SEDComponentState
        Run once at compile time. Reads SSP grids and other static
        inputs; returns the cached static tensors :meth:`apply` will
        need on every call. Eager (not JIT'd). May use file I/O.
        Both arguments are optional — components that do not need an
        SSP grid (radio, IGM, X-ray) leave them defaulted.

    apply(state, params) -> PipelineState
        Pure JAX. Reads only parameters whose name starts with
        :attr:`parameter_prefix`, plus any cached tensors held by
        ``self`` (typically a frozen :class:`SEDComponentState`).
        Returns a *new* ``PipelineState`` (do not mutate the input).

    Notes
    -----
    The protocol is :func:`runtime_checkable` so tests can assert
    ``isinstance(MyComp, SEDComponent)`` without importing the abstract
    base.

    This is a Protocol, not an ABC. Components do not have to subclass
    anything — duck-typing is enough. We provide
    :class:`SEDComponentConfig`/:class:`SEDComponentState` as
    convenience frozen dataclasses but components may use their own
    immutable types as long as the shape matches.
    """

    name: str
    parameter_prefix: str
    config: SEDComponentConfig

    def declared_parameters(self) -> list[ParamDeclaration]:
        """Free parameters this component owns.

        Returns
        -------
        list of :class:`ParamDeclaration`
            One entry per free parameter (name, prior, description).
            The orchestrator hands these to :class:`tengri.Parameters`
            so the user doesn't have to register them by hand.

        Notes
        -----
        Every ``name`` MUST start with :attr:`parameter_prefix` *or* be
        listed in :data:`BARE_NAME_ALLOWLIST` (currently just
        ``redshift``). The param-prefix CI guard at
        ``tools/check_param_prefixes.py`` enforces this.
        """
        ...

    # ``publishes()`` and ``requires()`` are intentionally NOT part of the
    # runtime-checkable Protocol surface. The validator at
    # :func:`tengri.forward.orchestrator.validate_pipeline` consults them
    # via ``getattr(c, "publishes", lambda: ())()`` so components that
    # have not yet been annotated still satisfy ``isinstance(c, SEDComponent)``.
    # Concrete components that DO declare cross-component data should add
    # methods with the signature::
    #
    #     def publishes(self) -> tuple[DerivedKey, ...]: ...
    #     def requires(self) -> tuple[DerivedKey, ...]: ...
    #     def requires_optional(self) -> tuple[DerivedKey, ...]: ...
    #
    # ``requires`` declares HARD dependencies — a missing publisher is a
    # construction error. ``requires_optional`` (Phase B of issue #21)
    # declares opportunistic reads that have a documented fallback —
    # the validator still checks units on the optional read if an
    # upstream publishes the key, but a missing publisher is not an error.
    # See ``RadioSEDComponent`` and ``XRaySEDComponent`` for the
    # canonical optional-read pattern, and ADR-0004 for the full rationale.

    def precompute(
        self,
        ssp_data: Any | None = None,
        wave_grid: jnp.ndarray | None = None,
    ) -> SEDComponentState:
        """Cache static tensors. Run once before any JIT compile.

        ``ssp_data`` and ``wave_grid`` are both optional: not every
        component needs them (radio, IGM, X-ray ignore SSP grids; IGM
        builds its transmission curve on the observed-frame grid at
        :meth:`apply` time). Components that *do* need them will fail
        their own validation if either is ``None``.
        """
        ...

    def apply(
        self,
        state: PipelineState,
        params: Mapping[str, jnp.ndarray],
    ) -> PipelineState:
        """Pure JAX step.

        Reads ``params`` (already sliced to this component's prefix by
        the orchestrator) plus any frozen tensors held by ``self``,
        then returns a *new* :class:`PipelineState` with this
        component's contribution applied.

        Parameters
        ----------
        state : PipelineState
            Current state from upstream components (e.g. stellar emits
            ``sed_intrinsic``; dust then reads it and writes
            ``sed_attenuated``).
        params : mapping of str -> array
            Free parameters whose name starts with
            :attr:`parameter_prefix`. The orchestrator does the
            slicing.

        Returns
        -------
        PipelineState
            New state. MUST NOT mutate the input.
        """
        ...
