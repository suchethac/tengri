"""Concrete kernel adapters.

Each adapter is a thin wrapper around one of the seven existing ``build_*``
factories. No math is moved here — the adapters exist to expose a uniform
:class:`Kernel`-shaped surface for selection. Predicates are lifted from
the equivalent inline checks in ``sed_model.py``; line references are kept
in docstrings so future readers can find the original site.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping
from typing import Any

from tengri.forward._kernels.compositional import (
    build_fused_tier2_photometry,
    build_fused_tier2_spectrum,
    build_hybrid_spectrum,
)
from tengri.forward._kernels.exact import build_exact_sed, build_fused_rest_sed
from tengri.forward._kernels.hybrid import (
    build_hybrid_photometry,
    build_hybrid_photometry_ztable,
)

# ── helpers ─────────────────────────────────────────────────────────


def _precomputed(state: Any) -> Any:
    """Return ``state.precomputed`` or ``None`` (state may be a bare mock in tests)."""
    return getattr(state, "precomputed", None)


def _has(state: Any, dotted: str) -> bool:
    """``state.<a>.<b>`` is not None. Returns False if any segment is missing."""
    obj: Any = state
    for part in dotted.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return False
    return True


# ── adapters ────────────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class ExactRestSEDKernel:
    """Adapter for :func:`build_exact_sed` — always available rest-SED builder.

    Original site: ``sed_model.py:1549`` (``_build_compositional_kernels``).
    """

    name: str = "exact_rest_sed"
    product: str = "rest_sed"

    def is_compatible(self, state: Any, model: Any = None) -> bool:
        return True

    def is_compatible_with_params(self, params: Mapping[str, Any]) -> bool:
        return True

    def build(self, state: Any, model: Any = None) -> Callable[..., Any] | None:
        return build_exact_sed(state)


@dataclasses.dataclass(frozen=True)
class CompositionalRestSEDKernel:
    """Adapter for :func:`build_fused_rest_sed` — Tier 2 full-wavelength rest SED.

    Original site: ``sed_model.py:1553``.
    """

    name: str = "compositional_rest_sed"
    product: str = "rest_sed"

    def is_compatible(self, state: Any, model: Any = None) -> bool:
        return getattr(state, "rest_wavelength", None) is not None

    def is_compatible_with_params(self, params: Mapping[str, Any]) -> bool:
        return True

    def build(self, state: Any, model: Any = None) -> Callable[..., Any] | None:
        return build_fused_rest_sed(state, model)


@dataclasses.dataclass(frozen=True)
class CompositionalPhotometryKernel:
    """Adapter for :func:`build_fused_tier2_photometry`.

    Original site: ``sed_model.py:1574``. Param-time block on
    ``met_mode in {"ramp", "chem_evol"}`` is lifted from
    ``sed_model.py:1815-1820``.
    """

    name: str = "compositional_photometry"
    product: str = "photometry"

    def is_compatible(self, state: Any, model: Any = None) -> bool:
        return (
            getattr(state, "filter_waves", None) is not None
            and getattr(state, "rest_wavelength", None) is not None
        )

    def is_compatible_with_params(self, params: Mapping[str, Any]) -> bool:
        # The met_mode predicate is state-level (lives on the model), but
        # historically the inline check was triggered by params reaching the
        # predict path. We keep it accessible here for future param-aware
        # routing; today it always returns True.
        return True

    def build(self, state: Any, model: Any = None) -> Callable[..., Any] | None:
        return build_fused_tier2_photometry(state, model)


@dataclasses.dataclass(frozen=True)
class CompositionalSpectrumKernel:
    """Adapter for :func:`build_fused_tier2_spectrum`.

    Original sites: ``sed_model.py:1583`` and ``sed_model.py:5075``
    (``precompute_spectroscopy`` rebuild).
    """

    name: str = "compositional_spectrum"
    product: str = "spectrum"

    def is_compatible(self, state: Any, model: Any = None) -> bool:
        if getattr(state, "rest_wavelength", None) is None:
            return False
        return (
            _has(state, "precomputed.spectroscopy") or getattr(state, "wave_obs", None) is not None
        )

    def is_compatible_with_params(self, params: Mapping[str, Any]) -> bool:
        return True

    def build(self, state: Any, model: Any = None) -> Callable[..., Any] | None:
        return build_fused_tier2_spectrum(state, model)


@dataclasses.dataclass(frozen=True)
class HybridPhotometryKernel:
    """Adapter for :func:`build_hybrid_photometry` (fixed-redshift).

    Original site: ``sed_model.py:1613``. Param-time block on tabulated SFH
    (``"sfh_t_gyr" in params``) is lifted from
    ``sed_model.py:3699`` (``_predict_photometry_auto``).
    """

    name: str = "hybrid_photometry"
    product: str = "photometry"

    def is_compatible(self, state: Any, model: Any = None) -> bool:
        return (
            _has(state, "precomputed.photometry") and getattr(state, "z_fixed", None) is not None
        )

    def is_compatible_with_params(self, params: Mapping[str, Any]) -> bool:
        # Tabulated SFH uses variable-size arrays; incompatible with
        # precomputed hybrid grids.
        return "sfh_t_gyr" not in params

    def build(self, state: Any, model: Any = None) -> Callable[..., Any] | None:
        return build_hybrid_photometry(state, model)


@dataclasses.dataclass(frozen=True)
class HybridPhotometryZTableKernel:
    """Adapter for :func:`build_hybrid_photometry_ztable` (free-redshift).

    Original site: ``sed_model.py:5196`` (``precompute_ztable``). Unlike
    the other adapters, the underlying builder **raises** ``ValueError``
    when the ztable is missing — the ``is_compatible`` gate avoids that.
    """

    name: str = "hybrid_photometry_ztable"
    product: str = "photometry"

    def is_compatible(self, state: Any, model: Any = None) -> bool:
        return _has(state, "precomputed.photometry_ztable")

    def is_compatible_with_params(self, params: Mapping[str, Any]) -> bool:
        return "sfh_t_gyr" not in params

    def build(self, state: Any, model: Any = None) -> Callable[..., Any] | None:
        return build_hybrid_photometry_ztable(state, model)


@dataclasses.dataclass(frozen=True)
class HybridSpectrumKernel:
    """Adapter for :func:`build_hybrid_spectrum`.

    Original site: ``sed_model.py:5081`` (``precompute_spectroscopy``).
    """

    name: str = "hybrid_spectrum"
    product: str = "spectrum"

    def is_compatible(self, state: Any, model: Any = None) -> bool:
        return (
            _has(state, "precomputed.spectroscopy") and getattr(state, "z_fixed", None) is not None
        )

    def is_compatible_with_params(self, params: Mapping[str, Any]) -> bool:
        return "sfh_t_gyr" not in params

    def build(self, state: Any, model: Any = None) -> Callable[..., Any] | None:
        return build_hybrid_spectrum(state, model)


# ── default adapter registry ────────────────────────────────────────


ALL_ADAPTERS: tuple[Any, ...] = (
    ExactRestSEDKernel(),
    CompositionalRestSEDKernel(),
    CompositionalPhotometryKernel(),
    CompositionalSpectrumKernel(),
    HybridPhotometryKernel(),
    HybridPhotometryZTableKernel(),
    HybridSpectrumKernel(),
)
"""Canonical adapter registry. Used by :class:`KernelStrategy` to resolve names
to adapters and to enumerate everything that could be built."""


def adapters_by_name() -> dict[str, Any]:
    """Return a ``{name: adapter}`` dict over :data:`ALL_ADAPTERS`."""
    return {a.name: a for a in ALL_ADAPTERS}
