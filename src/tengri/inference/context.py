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

    # ── Constructors ─────────────────────────────────────────────────────
    @classmethod
    def from_target(cls, target) -> InferenceContext:
        """Normalize ``target`` (Fitter or InferenceContext) to a context.

        Backends use this at their entry point during the multi-PR
        migration window — ``Fitter.run`` may pass either an
        ``InferenceContext`` (migrated backends) or a raw ``Fitter``
        (legacy path). Returns ``target`` unchanged if it is already
        a context.
        """
        if isinstance(target, cls):
            return target
        return cls(fitter=target)

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

    @property
    def cache(self):
        """The :class:`~tengri.inference.jit_engine.CompileCache` this fit uses.

        ADR-0009-Step-C exposes the cache as a per-fit object owned by the
        Fitter; this property surfaces it on the context so backends and
        the Likelihood module don't have to know to reach across
        ``context.fitter`` for it.
        """
        return self.fitter.cache

    # ── Observed data and likelihood-construction config ─────────────────
    #
    # These properties surface the data + noise + likelihood-shape
    # configuration that today lives on the Fitter as init kwargs. The
    # ``Likelihood`` module (``inference/likelihood.py``) reads through
    # this context rather than reaching into ``fitter.*`` private state,
    # so a backend that wants to build a likelihood without a full Fitter
    # only needs an ``InferenceContext`` and the data + noise arrays.
    #
    # Properties intentionally mirror the Fitter's attribute names without
    # the leading underscore — they are read-only from the backend's
    # perspective, but the storage stays on the Fitter (single source of
    # truth) until a separate refactor lifts the config into its own type.

    @property
    def data(self):
        """Observed flux array (photometry, spectroscopy, or joint)."""
        return self.fitter.data

    @property
    def noise(self):
        """1-σ noise array matching :attr:`data`."""
        return self.fitter.noise

    @property
    def data_mask(self):
        """Optional boolean / float mask over :attr:`data` (None when absent)."""
        return self.fitter.data_mask

    @property
    def data_type(self) -> str:
        """One of ``"photometry"``, ``"spectroscopy"``, ``"joint"``."""
        return self.fitter.data_type

    @property
    def has_spectroscopy(self) -> bool:
        """True iff :attr:`data_type` includes a spectroscopy channel."""
        return self.fitter._has_spectroscopy

    @property
    def fixed_values(self) -> dict:
        """Fixed-parameter values from the spec (frozen at Fitter init)."""
        return self.fitter._fixed_values

    # ── Likelihood-shape configuration ───────────────────────────────────
    @property
    def calibration_marginalize(self) -> bool:
        """Whether to marginalise an additive calibration polynomial."""
        return self.fitter._calibration_marginalize

    @property
    def cal_n_poly(self) -> int:
        """Polynomial order of the calibration model."""
        return self.fitter._cal_n_poly

    @property
    def cal_prior_sigma(self) -> float:
        """Prior width on calibration-polynomial coefficients."""
        return self.fitter._cal_prior_sigma

    @property
    def eline_marginalize(self) -> bool:
        """Whether to analytically marginalise emission-line amplitudes."""
        return self.fitter._eline_marginalize

    @property
    def eline_fitted(self) -> bool:
        """Whether to fit emission-line amplitudes as free parameters."""
        return self.fitter._eline_fitted

    @property
    def eline_prior_type(self) -> str | None:
        """``"flat"`` or ``"cloudy"`` (None when eline path inactive)."""
        return self.fitter._eline_prior_type

    @property
    def eline_prior_sigma(self) -> float | None:
        """Prior width on emission-line amplitudes (flat prior)."""
        return self.fitter._eline_prior_sigma

    @property
    def eline_prior_width_dex(self):
        """Prior width in dex for log-amplitude priors (Cloudy prior)."""
        return self.fitter._eline_prior_width_dex

    @property
    def eline_independent_wavelengths(self):
        """Wavelengths of independent (non-tied) emission lines."""
        return self.fitter._eline_independent_wavelengths

    @property
    def eline_amplitude_names(self) -> list[str]:
        """Names of the fitted emission-line amplitude parameters."""
        return self.fitter._eline_amplitude_names

    @property
    def eline_wavelengths(self):
        """All emission-line wavelengths (independent + tied)."""
        return self.fitter._eline_wavelengths

    @property
    def eline_constraint_matrix(self):
        """Matrix tying tied line amplitudes to independent ones."""
        return self.fitter._eline_constraint_matrix

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

    def unbounded_from_posterior(self, posterior: Posterior) -> dict:
        """Extract a warm-start point in unbounded space from a posterior."""
        return self.fitter._unbounded_from_posterior(posterior)

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
