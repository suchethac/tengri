"""Kernel selection strategy.

Replaces the smeared try/except/fallback chain in ``forward/sed_model.py``
with one small, testable, frozen dataclass: :class:`KernelStrategy`. Given a
preferred order of adapter names and a target product (rest_sed / photometry
/ spectrum), iterate compatible adapters so the caller can pick the first
that builds (or has been pre-built).

The strategy lives one level above JIT — never traced. It only inspects
state and params with plain Python.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator, Mapping
from typing import Any, Literal

from tengri.forward._kernels._adapters import ALL_ADAPTERS, adapters_by_name
from tengri.forward._kernels._protocol import (
    Kernel,
    NoCompatibleKernelError,
    Product,
)

OnFailure = Literal["warn", "raise", "silent"]
"""Policy for build-time failures (used by ``sed_model.py`` in PR2, declared
here so the strategy can carry the intent without depending on the model)."""


# Preferred names, in the order auto-mode should consult them. The default
# matches the historical cascade in ``_predict_photometry_auto``
# (sed_model.py:3697-3713) and ``_predict_spectrum_auto`` (3877-3886).
_DEFAULT_ORDER: tuple[str, ...] = (
    "compositional_photometry",
    "compositional_spectrum",
    "compositional_rest_sed",
    "hybrid_photometry",
    "hybrid_photometry_ztable",
    "hybrid_spectrum",
    "exact_rest_sed",
)


@dataclasses.dataclass(frozen=True)
class KernelStrategy:
    """Policy object that picks among compatible kernel adapters.

    Parameters
    ----------
    preferred : tuple of str
        Adapter names in priority order. The first compatible adapter for
        the requested product wins.
    on_failure : {"warn", "raise", "silent"}, default "warn"
        How build-time exceptions should be surfaced. Read by the model's
        build loop in PR2; the strategy itself does not raise.

    Notes
    -----
    Frozen and hashable so it can be passed safely between threads, stored
    on ``SEDModel``, or compared in tests. Never enters JIT.
    """

    preferred: tuple[str, ...] = _DEFAULT_ORDER
    on_failure: OnFailure = "warn"

    def select(
        self,
        state: Any,
        model: Any = None,
        product: Product = "photometry",
        requested_mode: str = "auto",
        params: Mapping[str, Any] | None = None,
    ) -> Iterator[Kernel]:
        """Yield adapters in preference order that match ``product`` and pass
        ``is_compatible`` (and ``is_compatible_with_params`` if ``params`` is
        given).

        Parameters
        ----------
        state : SEDModelState-like
            Frozen runtime bundle. Adapters read attributes like
            ``filter_waves``, ``precomputed.photometry``, ``z_fixed``.
        model : SEDModel or None
            Legacy escape hatch passed through to the underlying builders.
        product : {"rest_sed", "photometry", "spectrum"}
            Filters adapters to those producing the requested output type.
        requested_mode : str, default "auto"
            ``"auto"`` defers entirely to ``self.preferred``. Any other value
            is treated as a single-adapter shortcut and yields just that one
            (still gated by compatibility). Recognised shortcuts: ``"exact"``,
            ``"hybrid"``, ``"compositional"``; or a full adapter name.
        params : mapping or None
            User parameter dict, consulted by ``is_compatible_with_params``.

        Yields
        ------
        Kernel
            Adapters whose state-level and param-level predicates pass.
        """
        by_name = adapters_by_name()

        if requested_mode != "auto":
            names = _resolve_mode_shortcut(requested_mode, product)
            order = tuple(n for n in names if n in by_name)
        else:
            order = self.preferred

        for name in order:
            adapter = by_name.get(name)
            if adapter is None or adapter.product != product:
                continue
            if not adapter.is_compatible(state, model):
                continue
            if params is not None and not adapter.is_compatible_with_params(params):
                continue
            yield adapter

    def first_or_raise(
        self,
        state: Any,
        model: Any = None,
        product: Product = "photometry",
        requested_mode: str = "auto",
        params: Mapping[str, Any] | None = None,
    ) -> Kernel:
        """Return the first compatible adapter, or raise
        :class:`NoCompatibleKernelError`.

        Convenience for callers that want a single adapter, not an iterator.
        """
        for adapter in self.select(state, model, product, requested_mode, params):
            return adapter
        raise NoCompatibleKernelError(
            f"No compatible kernel for product={product!r}, "
            f"requested_mode={requested_mode!r}, preferred={self.preferred!r}. "
            "Call model.list_available_kernels() to see why each candidate "
            "was rejected."
        )


def _resolve_mode_shortcut(mode: str, product: Product) -> tuple[str, ...]:
    """Translate user-facing ``mode`` values to adapter-name tuples.

    Recognised shortcuts: ``exact``, ``hybrid``, ``compositional`` (each
    expands to the matching adapter for the requested product). Any unknown
    string is treated as a literal adapter name.
    """
    if mode == "exact":
        return ("exact_rest_sed",) if product == "rest_sed" else ()
    if mode == "hybrid":
        return {
            "rest_sed": (),
            "photometry": ("hybrid_photometry", "hybrid_photometry_ztable"),
            "spectrum": ("hybrid_spectrum",),
        }[product]
    if mode == "compositional":
        return {
            "rest_sed": ("compositional_rest_sed",),
            "photometry": ("compositional_photometry",),
            "spectrum": ("compositional_spectrum",),
        }[product]
    return (mode,)


# ── built-in policies ───────────────────────────────────────────────


DEFAULT: KernelStrategy = KernelStrategy()
"""Current production behaviour: compositional first, hybrid second, exact last."""

LOW_MEMORY: KernelStrategy = KernelStrategy(
    preferred=(
        "compositional_photometry",
        "compositional_spectrum",
        "compositional_rest_sed",
        "exact_rest_sed",
    )
)
"""Skip hybrid (largest closure / biggest XLA HLO). Prefer compositional + exact."""

EXACT_ONLY: KernelStrategy = KernelStrategy(preferred=("exact_rest_sed",))
"""Force the slow path. Useful for regression testing and debugging."""

COMPOSITIONAL_ONLY: KernelStrategy = KernelStrategy(
    preferred=(
        "compositional_photometry",
        "compositional_spectrum",
        "compositional_rest_sed",
    )
)
"""Force the Tier 2 compositional path."""


# Sanity check: every adapter referenced by a built-in policy is registered.
def _check_builtins() -> None:
    known = {a.name for a in ALL_ADAPTERS}
    for policy in (DEFAULT, LOW_MEMORY, EXACT_ONLY, COMPOSITIONAL_ONLY):
        missing = set(policy.preferred) - known
        if missing:  # pragma: no cover — defensive; means a typo above
            raise RuntimeError(f"KernelStrategy {policy!r} references unknown adapters: {missing}")


_check_builtins()
