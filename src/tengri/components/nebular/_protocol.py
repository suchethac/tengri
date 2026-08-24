# SPDX-License-Identifier: BSD-3-Clause
"""Protocol definition for nebular emission backends.

All backends must satisfy the NebularBackend Protocol to be usable in
the tengri forward model. The Protocol defines the minimal interface:

- has_continuum: class-level bool, True if backend provides nebular continuum
- predict_nebular_sed: SED on the SSP wavelength grid (existing interface)

Note: The unified line_luminosities / continuum_luminosity API is a future
step. For now, has_continuum is the only required addition.
"""

from typing import Protocol, runtime_checkable


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

    Methods
    -------
    predict_nebular_sed
        Returns nebular emission SED on the SSP wavelength grid [erg/s/Hz].
        Signature varies by backend; see individual backend docstrings.

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
