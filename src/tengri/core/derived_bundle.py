# SPDX-License-Identifier: BSD-3-Clause
"""Typed bundle for cross-component derived data.

Phase 1 of ADR-0007: introduce :class:`DerivedBundle` as a *drop-in*
replacement for the free-form ``Mapping[str, Any]`` currently stored
on :attr:`PipelineState.derived`. This module ships the type and the
dict-compat semantics; it does **not** change ``PipelineState.derived``
yet. That migration happens in a follow-up PR once every component's
write site has been updated.

Motivation
----------

After ADR-0004 (the publish/require contract), the *names* of
cross-component data are declared by ``publishes()`` and the *units*
are pinned by ``_CANONICAL_UNITS``. But the runtime container is still
``Mapping[str, Any]`` — so a publisher can typo its own write site
(``new_derived["L_ie"] = ...`` instead of ``"L_ir"``) and the contract
catches nothing. The validator agrees with the declaration; the
declaration agrees with the table; only the *code* disagrees, and the
consumer fails at runtime when reading the missing key.

A typed bundle closes that loop: the writer says
``bundle.with_(L_ir=value)``; a typo says ``bundle.with_(L_ie=...)``
and Python raises ``TypeError`` at trace time. Plus the static type
checker can flag the typo at edit time, no runtime needed.

Migration ergonomics
--------------------

Today's read sites use ``state.derived.get("L_ir", 0.0)`` and
``state.derived["lnu_age"]``. The bundle implements ``__getitem__``,
``.get()``, ``__contains__``, ``keys()``, ``items()``, ``values()``,
``__iter__``, and ``__len__`` so existing code keeps working without
edits. A field with value ``None`` is treated as "not present" — i.e.
``"L_ir" in bundle`` is ``True`` only if some upstream component
populated it. This matches the dict semantics where a missing key
means "no value".

The unknown-key error message includes a Levenshtein-2 ``Did you
mean: ...`` hint, mirroring the validator's style.

Adding a new derived key
------------------------

Add a field to :class:`DerivedBundle`, then add the same key to
``_CANONICAL_UNITS`` in ``tengri.forward.orchestrator``. Both edits
live in the same PR that introduces the publisher. The field set is
hand-maintained on purpose — same friction model as the contract
itself (ADR-0004), and a deliberate guard against ``derived`` becoming
the junk drawer it was before this typing.
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
    :class:`tengri.core.PipelineState` uses); never assign in place
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
    with the rest of :class:`PipelineState`. Every field is a *data*
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

    # Dust attenuation / emission
    L_ir: jnp.ndarray | None = None
    L_absorbed: jnp.ndarray | None = None
    dust_attenuation_factor: jnp.ndarray | None = None
    sed_dust_attenuated: jnp.ndarray | None = None
    sed_dust_ir: jnp.ndarray | None = None

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
    def from_dict(cls, d: dict[str, Any]) -> DerivedBundle:
        """Build a bundle from a plain dict.

        Keys matching typed fields populate those fields directly.
        Keys NOT in the typed set land in ``_extras`` rather than
        raising — this is the migration shim's whole point: existing
        write sites can spray dict keys, and the resulting bundle
        still answers ``__getitem__`` for them. Once every write site
        is migrated, the validator should additionally assert that
        ``_extras`` is empty for production builds.
        """
        known = set(cls.field_names())
        typed = {k: v for k, v in d.items() if k in known}
        extras = {k: v for k, v in d.items() if k not in known}
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

    def lev(a: str, b: str) -> int:
        if a == b:
            return 0
        if not a:
            return len(b)
        if not b:
            return len(a)
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            curr = [i]
            for j, cb in enumerate(b, 1):
                curr.append(min(curr[-1] + 1, prev[j] + 1, prev[j - 1] + (0 if ca == cb else 1)))
            prev = curr
        return prev[-1]

    best_name: str | None = None
    best_dist = 3
    for k in options:
        d = lev(target, k)
        if d < best_dist:
            best_dist = d
            best_name = k
    return best_name


# ── JAX pytree registration ──────────────────────────────────
#
# Each typed field is a data field. ``_extras`` is also a data field
# (a dict; JAX's default-dict-pytree handler will recurse into it).
# The frozen-dataclass + register_dataclass combo is identical to how
# PipelineState is registered in tengri.core.component.

from jax import tree_util as _tree_util

_DATA_FIELDS = tuple(f.name for f in fields(DerivedBundle))

_tree_util.register_dataclass(
    DerivedBundle,
    data_fields=_DATA_FIELDS,
    meta_fields=(),
)

del _tree_util
