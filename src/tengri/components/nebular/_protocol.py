# SPDX-License-Identifier: BSD-3-Clause
"""Protocol definition for nebular emission backends.

All backends must satisfy the NebularBackend Protocol to be usable in
the tengri forward model. The Protocol defines the minimal interface:

- has_continuum: class-level bool, True if backend provides nebular continuum
- predict_nebular_sed: SED on the SSP wavelength grid (existing interface)

Note: The unified line_luminosities / continuum_luminosity API is a future
step. For now, has_continuum is the only required addition.
"""

from typing import TYPE_CHECKING, ClassVar, Protocol, runtime_checkable

if TYPE_CHECKING:
    from tengri.protocols.component import ParamDeclaration


class NebularContinuumUnavailableError(Exception):
    """Raised when a nebular backend cannot provide continuum and no fallback exists.

    This exception is raised by NebularContinuumFallback (with fallback_mode="error")
    when a line-only backend (CB19, MAPPINGS, Shock) is called without either
    a secondary continuum-capable backend or required keyword arguments for analytic
    continuum.

    Notes
    -----
    Resolution strategies:
    (1) Pass a continuum-capable backend as ``fallback=`` to NebularContinuumFallback.
    (2) Provide ``ssp_wave`` and ``gas_logqion`` at prediction time for analytic continuum.
    (3) Switch to CloudyGridBackend or CueBackend which provide continuum natively.

    """


@runtime_checkable
class NebularBackend(Protocol):
    """Protocol definition for pluggable nebular emission backends.

    Defines the minimal interface that all nebular backends must satisfy to be
    usable in the tengri forward model, enabling runtime swapping of
    implementations (CB19, Cue, MAPPINGS, Cloudy grids, etc.) without changing
    the forward model or inference code.

    Attributes
    ----------
    has_continuum : bool
        Whether the backend provides nebular continuum emission. True for
        physics-based grids (Cue, CloudyGrid); False for line-only backends
        (CB19, MAPPINGS, Shock). NebularContinuumFallback uses this flag to
        add continuum via fallback mechanisms.
    has_free_params : bool
        Whether the backend has differentiable (JAX-traced) free parameters
        that can be optimized during inference. True for data-driven emulators
        (Cue); False for tabular grids. Used by inference to decide whether
        the backend participates in gradient-based optimization.
    name : str
        Short human-readable identifier (e.g., "Cue", "CB19", "Cloudy_CB19").
        Used in logging, configuration, and error messages.

    config_name : ClassVar[str]
        The ``config.backend`` string that selects this backend in
        :class:`NebularSEDComponentConfig` (e.g. ``"cb19"``). Differs from
        :attr:`name` for some backends (``CB19Backend.name == "cb19_grid"``).
    supports_split_sed : ClassVar[bool]
        Whether ``NebularSEDComponent.split_sed`` accepts this backend. False
        for the baked-in and shock backends, which have no continuum + line
        catalog split.
    line_fraction_policy : ClassVar[bool]
        Whether ``publish_line_fraction=True`` is allowed with this backend.
        True only for the Cue backend.
    reads : ClassVar[tuple[str, ...]]
        Names of the ``state.derived`` keys this backend adds to the
        component's ``inputs()``, in the order ``inputs()`` lists them.
    publishes_shock : ClassVar[bool]
        Whether the component publishes ``sed_shock`` in ``outputs()``. True
        only for the shock backend.
    carries_freefree : ClassVar[bool]
        Whether the backend's continuum carries thermal free-free when no instance
        overrides it. An instance attribute ``carries_freefree_continuum`` (set by
        Gutkin16 from ``continuum=``) takes precedence. Feeds
        :func:`nebular_backends_with_freefree_continuum`.

    additive : ClassVar[bool]
        Whether the backend adds its emission to ``sed_intrinsic``. False only
        for the baked-in backend, whose emission is already in the SSP grid.
    requires_instance : ClassVar[bool]
        Whether the component needs a constructed backend object at precompute.
        False only for the baked-in backend.
    caches_filters : ClassVar[bool]
        Whether precompute caches the filter passbands for the nebular photometry
        (``approx["wave_precomp"]``). False only for the baked-in backend.

    The photoionization and line backends also inherit
    ``NebularBackendDefaults`` (in ``_shared.py``), which
    supplies the hooks the component calls on every backend: ``call_kwargs``,
    ``split_nebular``, ``line_catalog_kwargs``, ``extra_derived`` and
    ``precompute_check``. A backend overrides only the hooks it differs on.

    Class-level members are readable without an instance, so the component can
    answer for a configuration string before any backend object exists.

    Methods
    -------
    predict_nebular_sed
        Returns nebular emission SED on the SSP wavelength grid [erg/s/Hz].
        Signature varies by backend; see individual backend docstrings.

    parameters
        Class method returning the free-parameter declarations the component
        owns for this backend (``NebularSEDComponent.declared_parameters``).

    Notes
    -----
    **JIT-compatible**: Protocol methods may or may not be JIT-compatible
    depending on backend implementation. Check individual backend docstrings
    for JIT compatibility statements.

    **Composition**: Any callable with these three attributes satisfies the
    protocol at runtime (duck typing via @runtime_checkable). This enables
    rapid prototyping of new backends without modifying the Protocol.

    **Line-only vs. continuum-capable**: Backends are not required to provide
    continuum. Line-only backends can be wrapped with NebularContinuumFallback
    to add analytic or physics-based continuum automatically.

    References
    ----------
    .. [1] This protocol is used internally by tengri to enable swappable
       nebular emission backends. Backend implementations should document
       their JIT compatibility and gradient behavior in their own docstrings.

    """

    has_continuum: bool
    has_free_params: bool
    name: str
    config_name: ClassVar[str]
    supports_split_sed: ClassVar[bool]
    line_fraction_policy: ClassVar[bool]
    reads: ClassVar[tuple[str, ...]]
    publishes_shock: ClassVar[bool]
    carries_freefree: ClassVar[bool]
    additive: ClassVar[bool]
    requires_instance: ClassVar[bool]
    caches_filters: ClassVar[bool]

    @classmethod
    def parameters(cls) -> "list[ParamDeclaration]":
        """Free-parameter declarations this backend contributes [see class docs]."""
        ...
