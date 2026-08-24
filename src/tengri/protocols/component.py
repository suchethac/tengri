# SPDX-License-Identifier: BSD-3-Clause
"""SEDComponent protocol: the shape each physics block satisfies.

A component owns one block of the forward model: stellar emission,
dust attenuation+emission, nebular lines, AGN, IGM, radio, X-ray: plus the parameters and
precomputed tensors that go with it. The
:class:`tengri.SEDModel` orchestrator runs components in order, threading
a :class:`ForwardState` through them. Each component declares the
derived keys it reads (`inputs()`) and the ones it writes (`outputs()`),
so the orchestrator can check the chain of physics before any JIT compile.

Notes
-----
**JIT-compatible:** :meth:`SEDComponent.apply` MUST be pure JAX so it
can be JIT/grad/vmap-compiled by the orchestrator. :meth:`precompute`
runs *before* tracing and may use eager numpy/file I/O.

**Immutability:** :class:`ForwardState` and :class:`SEDComponentState`
are frozen dataclasses. Components return *new* states rather than
mutating their inputs. This matches the global coding rule
(`~/.claude/rules/common/coding-style.md`) and is what makes
``jax.lax.scan`` over a list of components possible.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, NamedTuple, Protocol, runtime_checkable

import jax.numpy as jnp

from tengri.protocols.derived_state import DerivedState

# Deprecated alias kept on tengri.protocols.component for one release: # imports of
# ``DerivedBundle`` from this module continue to work via the
# renamed canonical type ``DerivedState``. The walker in
# ``tengri.citations.collect`` and a few legacy test fixtures still
# read this attribute; remove in v1.0.
DerivedBundle = DerivedState

__all__ = [
    "BARE_NAME_ALLOWLIST",
    "ComponentIOError",
    "DerivedBundle",
    "DerivedKey",
    "DerivedState",
    "ForwardState",
    "ParamDeclaration",
    "SEDComponent",
    "SEDComponentConfig",
    "SEDComponentState",
    "declared_default",
]


class ComponentIOError(ValueError):
    """Raised when one component's declared inputs/outputs disagree with the chain.

    Checked once at :class:`tengri.SEDModel` construction by
    :func:`tengri.forward.orchestrator.validate_pipeline`, before any
    JIT compile. The error message always names the offending component
    class and the offending derived key, with a "Did you mean: ..." hint
    when a missing producer looks like a typo. See
    :func:`validate_pipeline` for the full list of checks.
    """


# Deprecated alias: old name kept for one release.
PipelineContractError = ComponentIOError


# ─────────────────────────────────────────────────────────────────────
# Contract decisions resolved by the first-adapter pass
# (RadioSEDComponent + IGMSEDComponent, 2026-05).
# ─────────────────────────────────────────────────────────────────────

#: Parameter names that the orchestrator passes to *every* component
#: regardless of ``parameter_prefix``. Today this is just ``redshift``,
#: which is read by IGM, radio, X-ray, and the observation model.
#: Extend with care: every entry here weakens the prefix discipline
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
    name: str
        Parameter name. Must start with the component's
        :attr:`SEDComponent.parameter_prefix` *or* be in
        :data:`BARE_NAME_ALLOWLIST`.
    prior: Any
        A :class:`tengri.parameters.priors.Distribution` (Uniform,
        Gaussian, Fixed, …). Typed loosely as ``Any`` here to avoid an
        import cycle between :mod:`tengri.protocols` and
        :mod:`tengri.parameters`.
    description: str
        One-line human-readable description, mirrored into the prior
        registry when a component's parameters are registered.
    bound_check: Any, optional
        Callable ``(lo, hi) -> bool`` invoked when a user narrows the
        prior at construction time. Returns ``True`` for an admissible
        bound. ``None`` (default) means no check. Typed loosely as
        ``Any`` for the same import-cycle reason as :attr:`prior`.
    bound_error: str
        Human-readable message attached to a :class:`ValueError` when
        :attr:`bound_check` rejects a bound. Empty string means the
        default generic message is used.
    units: str
        Free-form units string (e.g. ``"Myr"``, ``"erg/s/Hz"``,
        ``"Msun/yr"``, ``"dex"``). Used by translation and introspection
        code to document parameter semantics. Empty string means unitless
        or not-yet-documented. Placed last in the NamedTuple so positional
        callsites: which historically use up to 5 args ending with
        ``bound_error``; remain backwards-compatible.
    free_prior: Any, optional
        The distribution to use when the user asks for this parameter to be
        **free**: i.e. ``all_params: FREE`` or a bare ``FREE`` sentinel.

        :attr:`prior` is the *registry default*: what you get when you do not
        mention the parameter, and for most parameters that is a ``Fixed``
        scalar. Before this field existed, ``FREE`` resolved to :attr:`prior`,
        so asking to free a Fixed-default parameter silently left it pinned
        and the fit ran with that physics frozen (#1264).

        Declare ``free_prior`` whenever the parameter has a defensible
        admissible range; normally the same range :attr:`bound_check`
        enforces, so the two cannot disagree (a contract test asserts
        ``bound_check(free_prior.lo, free_prior.hi)`` holds). ``None`` means
        "no defensible range declared"; ``FREE`` then refuses loudly rather
        than pretending, and the caller passes an explicit prior instead.
    """

    name: str
    prior: Any
    description: str = ""
    bound_check: Any = None
    bound_error: str = ""
    units: str = ""
    free_prior: Any = None


def declared_default(params: Sequence[ParamDeclaration], name: str) -> float:
    """Read a parameter's default out of its declaration.

    Use this to write a standalone model function's signature default::

        def skirtor_grid(wavelength, agn_log_lbol=DEFAULT_AGN_LOG_LBOL, ...):

    instead of repeating the number as a literal. A literal is a second copy
    of a value the declaration already owns, and the two drift: nine AGN
    entry points shipped ``agn_log_lbol=45.0``; the ``log10(erg/s)``
    magnitude; against a declaration reading ``log10(L/L_sun)``, so a bare
    call was ~1e33 too luminous and sat 31 dex outside the prior a fit can
    reach (#1200, #1560).

    Parameters
    ----------
    params: sequence of ParamDeclaration
        The declaring component's ``PARAMS`` tuple.
    name: str
        Parameter name to look up, e.g. ``"agn_log_lbol"``.

    Returns
    -------
    float
        The declared default.

    Raises
    ------
    KeyError
        If ``name`` is not declared in ``params``.
    ValueError
        If the declaration carries no ``default`` to read.

    Notes
    -----
    **JIT-compatible**: not applicable; import-time lookup over a static
    tuple, never traced.

    Enforces the ADR-0011 rule that the prior object is a parameter's single
    source of truth. ``tools/check_param_defaults.py`` guards the converse:
    no signature default may fall outside its declared prior's support.
    """
    for declaration in params:
        if declaration.name != name:
            continue
        default = getattr(declaration.prior, "default", None)
        if default is None:
            raise ValueError(
                f"{name!r} is declared but its prior carries no default to read; "
                f"give the declaration a `default=` or state the value explicitly."
            )
        return float(default)
    raise KeyError(f"{name!r} is not declared in the supplied PARAMS tuple.")


class DerivedKey(NamedTuple):
    """One cross-component datum published into :attr:`ForwardState.derived`.

    Returned in a tuple from :meth:`SEDComponent.outputs` and
    :meth:`SEDComponent.inputs`. Consumed once at pipeline construction
    by :func:`tengri.forward.orchestrator.validate_pipeline` to ensure
    every required key has an upstream producer with matching units, and
    (when needed) to derive a topological ordering over components.

    Mirrors :class:`ParamDeclaration` line-for-line so future readers see
    the same shape on both sides of the cross-component contract.

    Fields
    ------
    name: str
        Stable key written to / read from :attr:`ForwardState.derived`.
        Compared by string equality, so renames are caught at construction.
    units: str
        Free-form units tag, e.g. ``"erg/s"``, ``"Msun/yr"``, ``"dex"``,
        ``"erg/s/Hz"``, ``"yr"``, ``"photons/s"``. The validator compares
        publisher and consumer units strings for equality; the
        ``_CANONICAL_UNITS`` table in
        :mod:`tengri.forward.orchestrator` pins the expected string for
        every well-known key. Use ``""`` for unitless quantities (e.g.
        transmission factors, mass weights, attenuation factors).
    description: str
        One-line human-readable description shown in error messages and
        by future introspection helpers. Optional.

    Notes
    -----
    No numeric unit conversion is attempted by the validator: the
    contract refuses to silently paper over a unit disagreement, but it
    will not convert ``"Lsun"`` to ``"erg/s"`` for the consumer. Each
    component is responsible for converting at its own boundary.

    The strings are deliberately *not* :mod:`astropy.units` quantities; that path is
    JIT-incompatible and would cost a 5–100× performance
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

    Crucially, fields here are NOT JAX traced values: they configure
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
class ForwardState:
    """Threaded state passed through a chain of :class:`SEDComponent` objects.

    The carrier object for the forward model: holds the SED-in-progress
    plus a typed bag of derived quantities that downstream components
    read from upstream components. Renamed from ``PipelineState``
    (2026-05-18) for astronomer-friendly user-facing names;
    ``PipelineState`` remains as a soft alias for one minor version.

    Each component reads what it needs and returns a new ``ForwardState``
    (immutable). Field semantics match :mod:`tengri.forward.prediction`:

    - ``wave``              : rest-frame wavelength grid in Å
    - ``sed_intrinsic``     : pre-attenuation rest-frame L_nu in erg/s/Hz
    - ``sed_attenuated``    : post-attenuation rest-frame L_nu in erg/s/Hz
    - ``sed_observed``      : observer-frame F_nu (after IGM, redshift,
                               and units conversion)
    - ``lines``             : optional dict of emission-line spectra
    - ``derived``           : free-form dict for diagnostics
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
    # ``derived`` is a typed :class:`DerivedState`. Dict-shaped writes
    # at construction or via ``with_(derived={...})`` are auto-coerced
    # for backward compatibility (see ``__post_init__`` and ``with_``).
    derived: DerivedState = field(default_factory=lambda: DerivedState())

    def __post_init__(self) -> None:
        """Coerce dict-shaped ``derived`` input to a :class:`DerivedState`."""
        if not isinstance(self.derived, DerivedState):
            # Frozen dataclass: bypass the guard with object.__setattr__.
            object.__setattr__(self, "derived", DerivedState.from_dict(dict(self.derived)))

    def with_(self, **overrides: Any) -> ForwardState:
        """Return a copy of this state with selected fields replaced.

        Convenience for components::

            new_state = state.with_(sed_attenuated=tau_corrected)

        Equivalent to ``dataclasses.replace`` but reads better at call
        sites. A dict-shaped ``derived=`` is auto-coerced to
        :class:`DerivedState` so existing dict-style write patterns
        keep working unchanged.
        """
        from dataclasses import replace

        if "derived" in overrides and not isinstance(overrides["derived"], DerivedState):
            overrides["derived"] = DerivedState.from_dict(dict(overrides["derived"]))
        return replace(self, **overrides)

    def add_intrinsic(self, L_component: jnp.ndarray) -> ForwardState:
        """Accumulate a component's contribution to the intrinsic SED.

        Factorizes the common pattern::

            if state.sed_intrinsic is None:
                new_sed = L_component
            else:
                new_sed = state.sed_intrinsic + L_component
            state = state.with_(sed_intrinsic=new_sed)

        into a single call, improving readability and reducing
        copy-paste across components.

        Parameters
        ----------
        L_component: ndarray
            Component's contribution to the intrinsic SED in erg/s/Hz.
            Broadcasts against ``self.wave`` via JAX's standard rules.

        Returns
        -------
        ForwardState
            New state with ``sed_intrinsic`` = ``L_component`` (if
            ``self.sed_intrinsic is None``) or
            ``self.sed_intrinsic + L_component``.

        Notes
        -----
        JIT/grad/vmap-compatible: the branching on ``self.sed_intrinsic is None``
        is resolved at compile time (frozen field).
        """
        if self.sed_intrinsic is None:
            new_sed = L_component
        else:
            new_sed = self.sed_intrinsic + L_component
        return self.with_(sed_intrinsic=new_sed)


# ─────────────────────────────────────────────────────────────────────
# JAX pytree registration
# ─────────────────────────────────────────────────────────────────────
#
# ``ForwardState`` and ``SEDComponentState`` are threaded through
# JIT-compiled component pipelines, so they must be JAX pytrees. All
# fields are dynamic (data, not static metadata): even the ``Mapping``
# fields ``lines`` and ``derived`` are dicts of arrays that JAX's
# default dict-pytree handler unpacks recursively.
#
# Registration uses :func:`jax.tree_util.register_dataclass` (JAX ≥ 0.4),
# which preserves the frozen-dataclass nature and integrates with
# :func:`dataclasses.replace`-based mutation in :meth:`ForwardState.with_`.

from jax import tree_util as _tree_util

_tree_util.register_dataclass(
    ForwardState,
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


# Soft alias for backwards compatibility. ``ForwardState`` is the canonical
# name as of 2026-05-18; ``PipelineState`` will be removed in
# tengri v1.0. Aliasing the class object (rather than wrapping it) preserves
# isinstance checks and type-annotation equivalence in downstream code.
PipelineState = ForwardState


# ─────────────────────────────────────────────────────────────────────
# The Protocol
# ─────────────────────────────────────────────────────────────────────


@runtime_checkable
class SEDComponent(Protocol):
    """Contract for one block of the SED forward model.

    Concrete subclasses (``StellarSEDComponent``, ``DustSEDComponent``,
    ``NebularSEDComponent``, ``AGNSEDComponent``, ``IGMSEDComponent``,
    ``RadioSEDComponent``, ``XRaySEDComponent``) live in
    :mod:`tengri.components.<domain>.component`.

    Required attributes
    -------------------
    name: str
        Stable identifier for diagnostics and registry lookup. Examples:
        ``"stellar"``, ``"dust"``, ``"nebular"``, ``"agn"``, ``"igm"``.

    parameter_prefix: str
        Domain prefix from NAMING_CONTRACT §3.2: every parameter this
        component reads must start with this prefix. Used by the
        orchestrator to slice the global ``params`` dict before
        :meth:`apply`. Examples: ``"sfh_"`` (stellar), ``"dust_"``,
        ``"neb_"``, ``"agn_"``, ``"igm_"``, ``"radio_"``, ``"xray_"``.
        Must be non-empty. Bare names that several components share
        (e.g. ``redshift``) are passed via
        :data:`BARE_NAME_ALLOWLIST`, not via an empty prefix.

    config: SEDComponentConfig
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
        Both arguments are optional; components that do not need an
        SSP grid (radio, IGM, X-ray) leave them defaulted.

    apply(state, params) -> ForwardState
        Pure JAX. Reads only parameters whose name starts with
        :attr:`parameter_prefix`, plus any cached tensors held by
        ``self`` (typically a frozen :class:`SEDComponentState`).
        Returns a *new* ``ForwardState`` (do not mutate the input).

    Notes
    -----
    The protocol is :func:`runtime_checkable` so tests can assert
    ``isinstance(MyComp, SEDComponent)`` without importing the abstract
    base.

    This is a Protocol, not an ABC. Components do not have to subclass
    anything; duck-typing is enough. We provide
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

    # ``outputs()`` and ``inputs()`` are intentionally NOT part of the
    # runtime-checkable Protocol surface. The validator at
    # :func:`tengri.forward.orchestrator.validate_pipeline` consults them
    # via ``getattr(c, "outputs", lambda: ())()`` so components that
    # have not yet been annotated still satisfy ``isinstance(c, SEDComponent)``.
    # Concrete components that DO declare cross-component data should add
    # methods with the signature::
    #
    #     def outputs(self) -> tuple[DerivedKey, ...]: ...
    #     def inputs(self) -> tuple[DerivedKey, ...]: ...
    #     def optional_inputs(self) -> tuple[DerivedKey, ...]: ...
    #
    # ``inputs`` declares HARD dependencies: a missing producer is a
    # construction error. ``optional_inputs`` (Phase B of issue #21)
    # declares opportunistic reads that have a documented fallback: # the validator still checks
    # units on the optional read if an
    # upstream component outputs the key, but a missing producer is not
    # an error. See ``RadioSEDComponent`` and ``XRaySEDComponent`` for
    # the canonical optional-read pattern, and ADR-0009 for the full
    # rationale.
    #
    # Renamed from ``publishes`` / ``requires`` / ``requires_optional``
    # (2026-05-18). The old method names are still accepted by
    # the orchestrator as a backwards-compatible fallback for one minor
    # version; they emit a ``DeprecationWarning`` at validate_pipeline time.

    def precompute(
        self,
        ssp_data: Any | None = None,
        wave_grid: jnp.ndarray | None = None,
        approx: Mapping[str, bool] | None = None,
        filters: tuple[tuple[jnp.ndarray, jnp.ndarray], ...] | None = None,
    ) -> SEDComponentState:
        """Cache static tensors. Run once before any JIT compile.

        ``ssp_data`` and ``wave_grid`` are both optional: not every
        component needs them (radio, IGM, X-ray ignore SSP grids; IGM
        builds its transmission curve on the observed-frame grid at
        :meth:`apply` time). Components that *do* need them will fail
        their own validation if either is ``None``.

        ``approx`` is the build-time approximation dict (the resolved
        ``SEDModel._approx``). Each component reads the flags it owns
        (e.g. :class:`StellarSEDComponent` reads
        ``approx.get("wave_precomp")``) and ignores the rest. ``None``
        is equivalent to no approximations: all exact paths.

        ``filters`` is a tuple of (filter_wave_obs, filter_trans) pairs,
        where each pair contains 1-D arrays. Used only by components that
        build photometric lookup tables. ``None`` means no photometric
        precomputation is available.
        """
        ...

    def apply(
        self,
        state: ForwardState,
        params: Mapping[str, jnp.ndarray],
        ssp_data: Any | None = None,
        template_data: Any | None = None,
    ) -> ForwardState:
        """Pure JAX step.

        Reads ``params`` (already sliced to this component's prefix by
        the orchestrator) plus any frozen tensors held by ``self``,
        then returns a *new* :class:`ForwardState` with this
        component's contribution applied.

        Parameters
        ----------
        state: ForwardState
            Current state from upstream components (e.g. stellar emits
            ``sed_intrinsic``; dust then reads it and writes
            ``sed_attenuated``).
        params: mapping of str -> array
            Free parameters whose name starts with
            :attr:`parameter_prefix`. The orchestrator does the
            slicing.
        ssp_data: Any | None, optional
            SSP stellar population synthesis grid. Passed by the
            orchestrator for components that need it (typically stellar).
            Components that do not use it should ignore this argument.
            When provided, should override any SSP data held in ``self``
            for JIT purposes (threading as a runtime parameter instead of
            closure-capturing).
        template_data: Any | None, optional
            Nebular backend grids and weights (Cue, CloudyGrid, etc.).
            Passed by the orchestrator for components that need it
            (typically nebular). Components that do not use it should
            ignore this argument. When provided, should override any
            template data held in ``self`` for JIT purposes.

        Returns
        -------
        ForwardState
            New state. MUST NOT mutate the input.
        """
        ...

    def citations(self) -> tuple[str, ...]:
        """Bib keys for papers this component implements or solves for.

        Returns
        -------
        tuple of str
            Zero or more keys into
            :data:`tengri.citations.registry.REGISTRY`. Components with
            no associated papers return ``()``; components that ship
            real physics return the keys for every paper whose
            equation, atlas, NN weights, or algorithm they use. The
            walker in :mod:`tengri.citations.collect` unions these with
            the static association tables in
            :mod:`tengri.citations.associations` to assemble the full
            bibliography for a :class:`SEDModel`.

        Notes
        -----
        **JIT-compatible:** no. Called once, eagerly, when the walker
        assembles a model's bibliography. Never inside a JAX trace.

        **Required, not optional.** Concrete components MUST implement
        this; empty tuple is a valid return for boilerplate components
        with no physics paper, but the method itself is part of the
        contract. The earlier optional form (commit 8c06e142) silently
        dropped provenance from any component that forgot to annotate.

        Examples
        --------
        A dust component implementing Calzetti (2000) attenuation and
        Draine & Li (2007) thermal emission:

        >>> def citations(self) -> tuple[str, ...]:
        ...     return ("calzetti2000", "draine_li2007")
        """
        ...
