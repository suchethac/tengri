# SPDX-License-Identifier: BSD-3-Clause
"""Typed bundle for cross-component derived data on :attr:`ForwardState.derived`.

``DerivedBundle`` is a frozen dataclass with one field per documented
inter-component datum (``L_ir``, ``lnu_age``, …). Writers go through
``bundle.with_(L_ir=...)``; a typo (``L_ie``) becomes a ``TypeError``
at trace time instead of a silent missing key.

Read sites keep using mapping syntax (``state.derived.get("L_ir", 0.0)``,
``state.derived["lnu_age"]``, ``"L_ir" in state.derived``) — the bundle
implements ``__getitem__``, ``.get()``, ``__contains__``, ``keys()``,
``items()``, ``values()``, ``__iter__``, and ``__len__`` for that. A
field whose value is ``None`` reads as "not present" so missing data
behaves like a missing dict key.

Unknown-key errors include a Levenshtein-2 ``Did you mean: ...`` hint.

Adding a new derived key: add a field here, then add the same key to
``_CANONICAL_UNITS`` in :mod:`tengri.forward.orchestrator`. The field
set is hand-maintained on purpose — same friction model as the
publish/require contract (ADR-0004), and a deliberate guard against
``derived`` becoming a junk drawer.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field, fields
from typing import Any

import jax.numpy as jnp

__all__ = ["DerivedBundle"]


_MISSING = object()  # sentinel for .get(name) without default — match dict.get


@dataclass(frozen=True)
class DerivedBundle:
    r"""Typed container for cross-component derived data.

    Fields mirror :data:`tengri.forward.orchestrator._CANONICAL_UNITS`
    one-for-one — every canonical derived key has a field with the same
    name. ``None`` means "not populated by any upstream component".

    Mutation via :meth:`with_` (the same pattern
    :class:`tengri.protocols.ForwardState` uses); never assign in place
    because the dataclass is frozen.

    Dict compatibility
    ------------------

    All of these work, returning ``True`` / the value / iterating
    non-None fields exactly as a plain ``dict`` would::

        "L_ir" in bundle
        bundle["L_ir"]
        bundle.get("L_ir", 0.0)
        bundle.keys()
        bundle.items()
        bundle.values()
        list(bundle)
        len(bundle)

    JAX pytree
    ----------

    Registered with :func:`jax.tree_util.register_dataclass` so the
    bundle can ride through ``jax.lax.scan`` / ``jax.tree_map`` along
    with the rest of :class:`ForwardState`. Every field is a *data*
    field — JAX traces each one. ``None`` defaults are static at trace
    time; switching which fields are populated invalidates the JIT
    cache, which is acceptable because a given :class:`tengri.SEDModel`
    pins its component list at construction and never reshuffles.
    """

    # Stellar — formed/surviving mass + SFR variants
    log_mstar: jnp.ndarray | None = None
    log_mstar_formed: jnp.ndarray | None = None
    sfr: jnp.ndarray | None = None
    sfr_10myr: jnp.ndarray | None = None
    sfr_100myr: jnp.ndarray | None = None

    # Stellar — age-resolved tensors and ionising rate
    L_age: jnp.ndarray | None = None
    lnu_age: jnp.ndarray | None = None
    ssp_ages_yr: jnp.ndarray | None = None
    age_weights: jnp.ndarray | None = None
    nion: jnp.ndarray | None = None

    # Stellar — SFH grid + chemistry history (diagnostic)
    sfh_grid_lbt_yr: jnp.ndarray | None = None
    sfr_history: jnp.ndarray | None = None
    log_metallicity_history: jnp.ndarray | None = None

    # Stellar — photometry LUT (Phase 3b, published only when
    # ``approx={'wave_precomp': True}`` is set on SEDModel).
    # Rest-frame F_nu through the configured filters, in erg/s/Hz at
    # the source (no redshift / luminosity distance applied).
    stellar_phot_lnu_precomp: jnp.ndarray | None = None
    # Stellar — Taylor moment for filter-level dust attenuation
    # (Phase 3c-3c, published alongside stellar_phot_lnu_precomp). First
    # spectral moment of the CSP within each filter:
    # Ψ_b = ∫ L_ν(λ) (λ - λ_eff_b) T_b(λ) λ dλ / ∫ T_b(λ) λ dλ.
    # Used by Phase 3c-3c-ii dust integration via the expansion
    # f_b ≈ A(λ_eff)·Φ_b + A'(λ_eff)·Ψ_b (Zacharegkas+2025).
    # Units: erg/s/Hz × Å.
    stellar_phot_moment_precomp: jnp.ndarray | None = None
    # Stellar — age-resolved per-filter LUT (Phase 3c-3c-iv-a). Shape
    # ``(n_age, n_filter)``, units erg/s/Hz. The age axis is NOT
    # marginalised. Sum over the age axis equals
    # ``stellar_phot_lnu_precomp``. Published only when
    # ``approx={'wave_precomp': True}`` is set. Consumed by the two-
    # component dust LUT path (Phase 3c-3c-iv-c) to apply per-age
    # attenuation ``T(a, λ) = T_diff(λ) × T_bc(λ)^y(a)``.
    stellar_phot_lnu_per_age_precomp: jnp.ndarray | None = None
    stellar_phot_moment_per_age_precomp: jnp.ndarray | None = None

    # Dust attenuation / emission
    L_ir: jnp.ndarray | None = None
    L_absorbed: jnp.ndarray | None = None
    dust_attenuation_factor: jnp.ndarray | None = None
    sed_dust_attenuated: jnp.ndarray | None = None
    sed_dust_ir: jnp.ndarray | None = None
    # Dust attenuation per filter (Phase 3c-3c-ii). A(λ_eff) and its
    # wavelength derivative A'(λ_eff) at each filter pivot, used by
    # Phase 3c-3c-iii to apply Taylor expansion attenuation in
    # ``Observation.predict_via_precomp``:
    # f_b ≈ A(λ_eff)·Φ_b + A'(λ_eff)·Ψ_b
    # Shape ``(n_filters,)``. Published only when
    # ``approx={'wave_precomp': True}`` is set.
    dust_attenuation_precomp: jnp.ndarray | None = None
    dust_attenuation_slope_precomp: jnp.ndarray | None = None
    # Two-component dust (Phase 3c-3c-iv-b). Birth-cloud and diffuse
    # attenuation factors per filter pivot, plus their wavelength
    # slopes. Two-component dust factorises as
    # ``T(a, λ) = T_diff(λ) × T_bc(λ)^y(a)`` — the per-age dependence
    # comes from the young indicator ``y(a)`` below. Used by
    # Phase 3c-3c-iv-c to apply per-age expansion in predict_via_precomp.
    dust_bc_attenuation_precomp: jnp.ndarray | None = None
    dust_bc_attenuation_slope_precomp: jnp.ndarray | None = None
    dust_diff_attenuation_precomp: jnp.ndarray | None = None
    dust_diff_attenuation_slope_precomp: jnp.ndarray | None = None
    # Young-star indicator on the SSP age grid (Phase 3c-3c-iv-b),
    # shape ``(n_age,)``. Smooth sigmoid transition around
    # ``DustSEDComponent.config.t_birth_yr``. ``y(a)`` = 1 for fully
    # young, 0 for fully old, with a logistic transition controlled by
    # ``transition_width_dex``.
    dust_young_indicator: jnp.ndarray | None = None
    # Filter pivot wavelengths in the rest frame (published by stellar
    # when wave_precomp is on; shared by downstream filter-level
    # consumers like the dust attenuation LUT). Shape ``(n_filters,)``,
    # units Å.
    filter_eff_waves: jnp.ndarray | None = None

    # AGN (incl. GRAHSP alternates)
    L_agn_bol: jnp.ndarray | None = None
    L_agn_torus: jnp.ndarray | None = None
    L_agn_absorbed: jnp.ndarray | None = None
    sed_agn: jnp.ndarray | None = None
    sed_grahsp: jnp.ndarray | None = None

    # Nebular
    sed_nebular: jnp.ndarray | None = None
    sed_shock: jnp.ndarray | None = None
    line_waves: jnp.ndarray | None = None
    line_lums: jnp.ndarray | None = None
    # Nebular — photometry LUT (Phase 3c-3b, published only when
    # ``approx={'wave_precomp': True}`` is set on SEDModel and the nebular
    # backend supports filter-level precomputation (Cue / CloudyGrid).
    # For BakedIn nebular this is None — the nebular emission is already
    # baked into the SSP grid and therefore included in stellar_phot_lnu_precomp.
    nebular_phot_lnu_precomp: jnp.ndarray | None = None

    # Radio / X-ray / IGM / shock
    sed_radio: jnp.ndarray | None = None
    sed_xray: jnp.ndarray | None = None
    igm_transmission: jnp.ndarray | None = None
    shock_log_lhalpha: jnp.ndarray | None = None

    # Free-form spillover dict for keys not yet promoted to fields.
    # Empty in steady state; provides a graceful path when an
    # in-flight migration declares a new key in _CANONICAL_UNITS
    # before adding the matching field here. Always-non-None
    # (default empty dict) so JAX pytree registration sees a stable
    # structure across instances.
    _extras: dict[str, Any] = field(default_factory=dict)

    # ── helpers ────────────────────────────────────────────────

    @classmethod
    def field_names(cls) -> tuple[str, ...]:
        """Tuple of every typed-field name (excludes ``_extras``)."""
        return tuple(f.name for f in fields(cls) if f.name != "_extras")

    def with_(self, **overrides: Any) -> DerivedBundle:
        """Return a copy with selected fields replaced.

        Unknown keys raise :class:`TypeError` (the dataclass-frozen
        default) with a Levenshtein-2 ``Did you mean: ...`` hint if
        the typo is close to a known field.
        """
        from dataclasses import replace

        # Pre-check unknown keys to produce a friendly hint message.
        known = set(self.field_names())
        unknown = [k for k in overrides if k not in known]
        if unknown:
            offender = unknown[0]
            hint = _did_you_mean(offender, known)
            suffix = f" (Did you mean: {hint!r}?)" if hint else ""
            raise TypeError(
                f"DerivedBundle has no field {offender!r}.{suffix} "
                f"Use ``with_(_extras={{...}})`` to attach values that "
                f"haven't been promoted to typed fields yet."
            )
        return replace(self, **overrides)

    # ── dict compatibility ────────────────────────────────────

    def __getitem__(self, name: str) -> Any:
        if name in self.field_names():
            value = getattr(self, name)
            if value is None:
                raise KeyError(name)
            return value
        if name in self._extras:
            return self._extras[name]
        raise KeyError(name)

    def get(self, name: str, default: Any = _MISSING) -> Any:
        try:
            return self[name]
        except KeyError:
            if default is _MISSING:
                return None
            return default

    def __contains__(self, name: object) -> bool:
        if not isinstance(name, str):
            return False
        if name in self.field_names():
            return getattr(self, name) is not None
        return name in self._extras

    def keys(self) -> list[str]:
        """Names of fields with non-None values + any keys in _extras."""
        live = [n for n in self.field_names() if getattr(self, n) is not None]
        live.extend(self._extras.keys())
        return live

    def values(self) -> list[Any]:
        return [
            getattr(self, n) for n in self.field_names() if getattr(self, n) is not None
        ] + list(self._extras.values())

    def items(self) -> list[tuple[str, Any]]:
        out: list[tuple[str, Any]] = []
        for n in self.field_names():
            v = getattr(self, n)
            if v is not None:
                out.append((n, v))
        out.extend(self._extras.items())
        return out

    def __iter__(self) -> Iterator[str]:
        return iter(self.keys())

    def __len__(self) -> int:
        return len(self.keys())

    # ── migration helpers ─────────────────────────────────────

    @classmethod
    def from_dict(
        cls,
        d: dict[str, Any],
        *,
        allow_extras: bool = False,
    ) -> DerivedBundle:
        """Build a bundle from a plain dict.

        Strict by default: when ``allow_extras=False``, keys outside
        the typed set raise :class:`TypeError` with a Levenshtein-2
        "Did you mean: ..." hint. Production write paths
        (:meth:`ForwardState.__post_init__`, :meth:`ForwardState.with_`)
        rely on this strictness to catch typos at trace time.

        Pass ``allow_extras=True`` to spill unknown keys into
        ``_extras`` (still readable via mapping syntax). Used by the
        legacy-compat tests; not used by production code.
        """
        known = set(cls.field_names())
        typed = {k: v for k, v in d.items() if k in known}
        extras = {k: v for k, v in d.items() if k not in known}
        if extras and not allow_extras:
            # Pick the first offender so the error message is short
            # and stable. The hint pinpoints likely typos at distance
            # ≤ 2 — same UX as ``with_(unknown=…)``.
            offender = next(iter(extras))
            hint = _did_you_mean(offender, known)
            suffix = f" (Did you mean: {hint!r}?)" if hint else ""
            raise TypeError(
                f"DerivedBundle.from_dict received {len(extras)} unknown "
                f"key(s): first is {offender!r}.{suffix} The dict-style "
                f"write path is closed as of ADR-0007 Phase 4 — use "
                f"``state.derived.with_(...)`` to set typed fields, or "
                f"pass ``allow_extras=True`` to opt into the legacy "
                f"spillover-to-_extras shim for migration / debugging."
            )
        return cls(**typed, _extras=extras)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dict — the inverse of :meth:`from_dict`.

        Non-None typed fields are included; ``_extras`` is merged in.
        Useful for serialisation, debugging, and gradual migration.
        """
        out: dict[str, Any] = {
            n: getattr(self, n) for n in self.field_names() if getattr(self, n) is not None
        }
        out.update(self._extras)
        return out


def _did_you_mean(target: str, options) -> str | None:
    """Levenshtein-2 nearest match, or None if no close match."""
    from tengri.utils.strings import closest

    return closest(target, options, max_distance=2)


# ── JAX pytree registration ──────────────────────────────────
#
# Each typed field is a data field. ``_extras`` is also a data field
# (a dict; JAX's default-dict-pytree handler will recurse into it).
# The frozen-dataclass + register_dataclass combo is identical to how
# ForwardState is registered in tengri.protocols.component.

from jax import tree_util as _tree_util

_DATA_FIELDS = tuple(f.name for f in fields(DerivedBundle))

_tree_util.register_dataclass(
    DerivedBundle,
    data_fields=_DATA_FIELDS,
    meta_fields=(),
)

del _tree_util
