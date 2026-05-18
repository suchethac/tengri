"""Inference context — the Python-level seam between Fitter and backends.

This module defines :class:`InferenceContext`, the bundle of state that
inference backends receive in lieu of the full :class:`~tengri.Fitter`.
It mirrors the role of :class:`~tengri.forward._kernels.strategy.KernelStrategy`
on the forward-model side (ADR-0004): a frozen Python-level object that
adapters consume to avoid coupling to the orchestrator's private internals.

Design rules — these are non-negotiable:

1. **Python-level only.** ``InferenceContext`` must never be hashed into a
   JIT key, passed through ``jax.jit`` / ``jax.vmap`` / ``jax.lax.scan`` as a
   traced argument, or stored inside a pytree leaf seen by JAX. Pull
   primitives (``loss_fn``, ``data_args``) out of context *before* entering
   any JAX transform.
2. **The dataclass is frozen, the engine handle is not.** ``frozen=True``
   protects the *wiring* of the context (which fitter, which engine), not
   the contents of mutable objects it references. The in-process sampler
   cache (``Fitter._jit_sampler``) remains mutable and shared across
   ``run()`` calls.
3. **The Fitter reference is the legacy escape hatch.** Backends that
   still call ``fitter._unbounded_from_posterior(...)`` etc. read it via
   ``context.fitter``. Subsequent PRs migrate those calls onto explicit
   context methods, after which the escape hatch is removed.

See ``docs/adr/0004-kernel-strategy-module.md`` for the forward-model
analogue this design copies.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tengri.inference.fitter import Fitter
    from tengri.inference.posterior import Posterior

__all__ = ["InferenceContext"]


@dataclass(frozen=True)
class InferenceContext:
    """Frozen bundle of state that inference backends consume.

    Backends receive an ``InferenceContext`` from :meth:`Fitter.run` and
    use it to obtain the loss function, parameter spec, data, and
    initialization helpers without reaching into Fitter internals.

    Parameters
    ----------
    fitter : Fitter
        Source-of-truth orchestrator. Currently the canonical place where
        the loss function, JIT engine cache, and parameter spec live.
        Backends are encouraged to read through the explicit accessors
        below; direct ``context.fitter`` access is the legacy escape
        hatch and will be progressively narrowed.

    Notes
    -----
    The accessors are intentionally thin properties — they exist so that
    a future PR can replace each one's implementation (e.g. promote
    ``loss_fn`` to a value held directly on the context) without touching
    backend code.

    JIT contract: never hash or trace this object. See module docstring.
    """

    fitter: Fitter

    # ── Forward-model state ──────────────────────────────────────────────
    @property
    def model(self):
        """The SEDModel being fit."""
        return self.fitter.model

    @property
    def spec(self):
        """The Parameters spec (free + fixed parameters)."""
        return self.fitter.spec

    # ── Loss and gradient (JIT-cached on the Fitter) ─────────────────────
    @property
    def loss_fn(self) -> Callable:
        """JIT-compiled negative-log-posterior callable.

        Cached on the Fitter; safe to call repeatedly. The compiled
        callable closes over ``data_args`` and the parameter spec.
        """
        return self.fitter._get_or_build_loss_fn()

    @property
    def grad_fn(self) -> Callable:
        """JIT-compiled gradient of the loss function."""
        return self.fitter._get_or_build_grad_fn()

    # ── Data and run-time controls ───────────────────────────────────────
    @property
    def data_args(self) -> dict:
        """Dict of arrays the loss function closes over (fluxes, masks, ...)."""
        return self.fitter._data_args

    @property
    def memory_mode(self) -> str:
        """ "fast" or "low" — controls jax.checkpoint wrapping inside CG."""
        return getattr(self.fitter, "_memory_mode", "fast")

    @property
    def posterior_chunk_size(self) -> int | None:
        """Chunk size for posterior sampling, or ``None`` for unchunked."""
        return getattr(self.fitter, "_posterior_chunk_size", None)

    # ── Initialization helpers ───────────────────────────────────────────
    def initial_params(self, key, init_from: Posterior | None = None) -> dict:
        """Build a starting point in unbounded parameter space.

        Delegates to ``Fitter._unbounded_from_posterior`` when warm-starting
        from a previous result, otherwise to ``Fitter._initialize_unbounded``.
        """
        if init_from is not None:
            return self.fitter._unbounded_from_posterior(init_from)
        return self.fitter._initialize_unbounded(key)

    def to_physical(self, params: dict) -> dict:
        """Map unbounded params back to physical (bounded) space."""
        return self.fitter._to_physical(params)

    # ── Free-name list (handy for diagnostics) ───────────────────────────
    @property
    def free_names(self) -> list[str]:
        """List of free parameter names in dispatch order."""
        return list(self.fitter._free_names)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"InferenceContext(n_free={self.spec.n_free}, "
            f"memory_mode={self.memory_mode!r}, "
            f"model={type(self.model).__name__})"
        )

    # Explicitly forbid pickling/hashing into JAX-traceable state.
    def __jax_array__(self) -> Any:  # pragma: no cover - guard rail
        raise TypeError(
            "InferenceContext is a Python-level orchestration object and "
            "must not be traced by JAX. Pull primitives "
            "(context.loss_fn, context.data_args, ...) out of the context "
            "before entering jax.jit / jax.vmap / jax.lax.scan."
        )
