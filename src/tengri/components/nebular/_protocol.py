"""Protocol definition for nebular emission backends.

All backends must satisfy the NebularBackend Protocol to be usable in
the tengri forward model. The Protocol defines the minimal interface:
- has_continuum: class-level bool, True if backend provides nebular continuum
- predict_nebular_sed: SED on the SSP wavelength grid (existing interface)

Note: The unified line_luminosities / continuum_luminosity API is a future
step (Phase N-2b). For now, has_continuum is the only required addition.
"""

from typing import Protocol, runtime_checkable


class NebularContinuumUnavailableError(Exception):
    """Raised when a nebular backend cannot provide continuum and no fallback exists.

    This exception is raised by NebularContinuumFallback (with fallback_mode="error")
    when a line-only backend (CB19, MAPPINGS, Shock) is called without either:
    - a secondary continuum-capable backend (fallback=CueBackend or CloudyGridBackend), or
    - the required keyword arguments (ssp_wave and gas_logqion) for analytic continuum.

    Resolution: either (1) pass a continuum-capable backend as fallback=,
    (2) provide ssp_wave and gas_logqion at prediction time, or
    (3) switch to CloudyGridBackend or CueBackend which include continuum natively.
    """


@runtime_checkable
class NebularBackend(Protocol):
    """Protocol definition for pluggable nebular emission backends.

    All nebular backends in the tengri forward model must satisfy this minimal
    interface, enabling runtime swapping of implementations (CB19, Cue, MAPPINGS,
    Cloudy grids, etc.) without changing the forward model or inference code.

    Attributes
    ----------
    has_continuum : bool
        Whether the backend provides nebular continuum emission in addition to
        (or in place of) emission lines. True for physics-based grids (Cue,
        CloudyGrid); False for line-only backends (CB19, MAPPINGS). Tooling
        like NebularContinuumFallback uses this flag to add continuum via
        fallback mechanisms when needed.
    has_free_params : bool
        Whether the backend has differentiable (JAX-traced) free parameters
        that can be optimized during inference. True for data-driven emulators
        (Cue); False for tabular grids. Used by the inference layer to decide
        whether the backend participates in gradient-based inference.
    name : str
        Short human-readable identifier (e.g., "Cue", "CB19", "Cloudy_CB19").
        Used in logging, configuration, and error messages.

    Methods
    -------
    predict_nebular_sed(...)
        Returns nebular emission SED on the SSP wavelength grid [erg/s/Hz].
        Signature varies by backend; see individual backend docstrings.

    Notes
    -----
    **Composition**: Any callable with these three attributes is considered a valid
    backend at runtime (duck typing via @runtime_checkable). This enables rapid
    prototyping of new backends without modifying the Protocol definition.

    **Line-only vs. continuum-capable**: A backend is NOT required to provide
    continuum. Line-only backends can be wrapped with NebularContinuumFallback
    to add analytic or physics-based continuum automatically.
    """

    has_continuum: bool
    has_free_params: bool
    name: str
