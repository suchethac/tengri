# SPDX-License-Identifier: BSD-3-Clause
"""Inference context — the Python-level seam between Fitter and backends.

This module defines :class:`InferenceContext`, the bundle of state that
inference backends receive in lieu of the full :class:`~tengri.Fitter`.
It mirrors the role ``KernelStrategy`` played on the forward-model side
(ADR-0004): a frozen Python-level object that adapters consume to avoid
coupling to the orchestrator's private internals. That class has since been
removed, so it is named here in plain markup rather than cross-referenced.

Design rules — these are non-negotiable:

1. **Python-level only.** ``InferenceContext`` must never be hashed into a
   JIT key, passed through ``jax.jit`` / ``jax.vmap`` / ``jax.lax.scan`` as a
   traced argument, or stored inside a pytree leaf seen by JAX. Pull
   primitives (``neg_log_posterior_fn``, ``data_args``) out of context
   *before* entering any JAX transform.
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
analog this design copies.
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

    # ── Objective and gradient (JIT-cached on the Fitter) ────────────────
    @property
    def neg_log_posterior_fn(self) -> Callable:
        """JIT-compiled negative-log-posterior callable.

        This is the quantity samplers / optimizers minimize: it is the
        negative of the (unnormalized) Bayesian posterior

        .. math::

            -\\log p(\\theta \\mid d) =
                -\\log p(d \\mid \\theta) - \\log p(\\theta) + \\text{const.}

        i.e. the sum of the negative log-likelihood and the negative log-prior.
        Astronomers familiar with χ²-minimization can think of this as a
        prior-regularized χ² (up to a constant). Cached on the Fitter; safe
        to call repeatedly. The compiled callable takes parameters in
        **unbounded** space and closes over ``data_args`` and the
        parameter spec.

        See Also
        --------
        tengri.inference.loss_functions.build_loglikelihood_fn
            Builds the **pure** log-likelihood ``log p(d | params)`` with
            no prior contribution, in **physical** parameter space.
            Suitable when you want to inspect data fit quality without
            the prior penalty.

        Notes
        -----
        Renamed from ``loss_fn`` (2026-05-18) for astronomer-friendly
        naming. ``loss_fn`` remains as a deprecated
        property; both return the same callable. ``loss_fn`` is removed
        in tengri v1.0.
        """
        return self.fitter._get_or_build_loss_fn()

    @property
    def loss_fn(self) -> Callable:
        """Deprecated alias for :attr:`neg_log_posterior_fn`.

        .. deprecated:: 0.x
            ``loss_fn`` is ML-jargon for the negative log posterior;
            use :attr:`neg_log_posterior_fn` instead. The old name is
            removed in tengri v1.0.
        """
        import warnings

        warnings.warn(
            "`InferenceContext.loss_fn` is deprecated and will be removed "
            "in tengri v1.0; use `InferenceContext.neg_log_posterior_fn` "
            "instead (same callable, astronomer-friendly name).",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.fitter._get_or_build_loss_fn()

    @property
    def grad_fn(self) -> Callable:
        """JIT-compiled gradient of :attr:`neg_log_posterior_fn`."""
        return self.fitter._get_or_build_grad_fn()

    @property
    def log_likelihood_fn(self) -> Callable:
        """JIT-compiled log-likelihood ``log p(d | xi)`` in standardized space.

        The pure data term of the information Hamiltonian, with the
        prior contribution stripped off. Useful for inference methods
        that consume likelihood and prior separately — most notably
        Nested Sampling, which draws from the prior under a
        likelihood-threshold constraint.

        Sign convention: returns the **positive** log-likelihood
        (i.e. ``-0.5 * chi^2 + const``); negate to get the χ² term
        of :math:`\\mathcal{H}`.

        Signature: ``log_likelihood_fn(params_unbounded, data_args) -> scalar``.

        See Also
        --------
        log_prior_fn
            The pure prior term ``log p(xi) = -0.5 * xi^T xi``.
        neg_log_posterior_fn
            The combined :math:`\\mathcal{H} = -\\log p(d|\\xi) - \\log p(\\xi)`.
        """
        return self.fitter._get_or_build_loglikelihood_unbounded_fn()

    @property
    def log_prior_fn(self) -> Callable:
        """Log-prior ``log p(xi)`` in standardized space.

        In the standardized latent space (paper §2 'Standardized
        Inference'), the prior is :math:`\\mathcal{N}(0, I)` for every
        free parameter — see ``_unstandardize_parameters`` in
        :mod:`tengri.inference.loss_functions`. The (unnormalized)
        log-prior is therefore:

        .. math::

            \\log p(\\boldsymbol{\\xi}) =
                -\\tfrac{1}{2}\\,\\boldsymbol{\\xi}^{\\!\\top}\\boldsymbol{\\xi} + \\text{const}

        i.e. the prior penalty term of the information Hamiltonian
        with the opposite sign.

        Signature: ``log_prior_fn(params_unbounded) -> scalar``.

        Stochastic-SFH free fields contribute their ``psd_xi`` array
        to the same quadratic form (paper §2.2 and Appendix A justify
        the same isotropic-normal cancellation across prior types).

        See Also
        --------
        log_likelihood_fn
            The pure data term.
        neg_log_posterior_fn
            The combined :math:`\\mathcal{H}`.
        """

        spec = self.fitter.spec
        free_names = self.fitter._free_names
        stochastic = spec.stochastic
        field_centering = float(getattr(spec, "field_centering", 1.0))

        from tengri.inference.loss_functions import standardized_neg_log_prior

        def log_prior(params_unbounded):
            """log p(xi) = -0.5 * sum(xi^2) for the standardized N(0,I) prior.

            Shares the objective's implementation rather than restating it. The
            two disagreed on batched inputs: this one omitted the per-galaxy
            reduction and returned shape ``(n_gal,)`` against the scalar its
            docstring promised.

            At ``field_centering < 1`` the field term is no longer standardized
            and depends on sigma, so the amplitude is read from the same dict —
            see :func:`standardized_neg_log_prior` (#1355).
            """
            psd_sigma_dex = None
            if field_centering != 1.0:
                psd_sigma_dex = params_unbounded.get("sfh_field_psd_sigma")
            return -standardized_neg_log_prior(
                params_unbounded,
                free_names,
                stochastic=stochastic,
                centering=field_centering,
                psd_sigma_dex=psd_sigma_dex,
            )

        return log_prior

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
        """Whether to marginalize an additive calibration polynomial."""
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
        """Whether to analytically marginalize emission-line amplitudes."""
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
            "(context.neg_log_posterior_fn, context.data_args, ...) out of "
            "the context before entering jax.jit / jax.vmap / jax.lax.scan."
        )
