"""Kernel selection protocol.

Defines the uniform shape that every forward-kernel adapter satisfies, so that
:class:`tengri.forward._kernels.strategy.KernelStrategy` can pick among them
without knowing which underlying ``build_*`` function it wraps.

The Protocol lives one level above JIT — never traced, never closed over by
a JAX transform. It is pure Python orchestration.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping
from typing import Any, Literal, Protocol, runtime_checkable

Product = Literal["rest_sed", "photometry", "spectrum"]
"""What a kernel produces. Used to filter adapters at select time."""


@runtime_checkable
class Kernel(Protocol):
    """Uniform interface every forward-kernel adapter satisfies.

    Adapters wrap the seven existing ``build_*`` builders (exact_sed,
    fused_rest_sed, fused_tier2_photometry, fused_tier2_spectrum,
    hybrid_photometry, hybrid_photometry_ztable, hybrid_spectrum) without
    moving any math. The Protocol exists so the strategy module can iterate
    them by name and product.

    Attributes
    ----------
    name : str
        Stable identifier, e.g. ``"hybrid_photometry"`` or ``"exact_rest_sed"``.
    product : {"rest_sed", "photometry", "spectrum"}
        What the built closure returns.
    """

    name: str
    product: Product

    def is_compatible(self, state: Any, model: Any = None) -> bool:
        """Return True iff the kernel's static preconditions are satisfied.

        State-only check; does **not** look at user params. The param-time
        check lives in :meth:`is_compatible_with_params`.
        """
        ...

    def is_compatible_with_params(self, params: Mapping[str, Any]) -> bool:
        """Return True iff this kernel can be invoked with these params.

        Default implementation returns True; adapters override when a
        param-shape constraint (e.g. tabulated SFH blocks hybrid) applies.
        """
        ...

    def build(self, state: Any, model: Any = None) -> Callable[..., Any] | None:
        """Build the JIT-compiled closure, or return None if prerequisites are
        not present in ``state`` at build time.

        Implementations should not raise for ordinary preconditions —
        compose ``is_compatible`` to gate the call. Raising signals an
        unexpected build failure (XLA OOM, unsupported component combo)
        which the caller will record in the build log.
        """
        ...


@dataclasses.dataclass(frozen=True)
class KernelBuildFailure:
    """Captures why a kernel failed to build, for the build log.

    Attributes
    ----------
    kernel_name : str
        Identifier of the adapter that failed.
    exception_type : str
        Class name of the exception (e.g. ``"XlaRuntimeError"``).
    message : str
        First line of the exception message.
    """

    kernel_name: str
    exception_type: str
    message: str


class NoCompatibleKernelError(RuntimeError):
    """Raised when a :class:`KernelStrategy` cannot find any compatible kernel.

    The message includes the strategy's preferred order and the model's
    build log so the user can see why each option was rejected.
    """
