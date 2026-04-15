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
    """Raised when a backend has no nebular continuum and no fallback is configured.

    Use NebularContinuumFallback wrapper from _shared.py to add a fallback,
    or switch to CloudyGridBackend or CueBackend which provide continuum natively.
    """


@runtime_checkable
class NebularBackend(Protocol):
    """Minimal Protocol all nebular backends must satisfy.

    Attributes
    ----------
    has_continuum : bool
        True if the backend can return nebular continuum via predict_nebular_sed.
        False for line-only backends (CB19, MappingsPhotoStellar, MappingsPhotoAGN,
        ShockEmission).
    has_free_params : bool
        True if the backend has JAX-traced (differentiable) parameters.
    name : str
        Short identifier string for the backend.
    """

    has_continuum: bool
    has_free_params: bool
    name: str
