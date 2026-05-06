"""Lock the deprecation contract for the `_sfh`-suffix removal (Phase 3).

The 19 SFH function names with a redundant ``_sfh`` suffix were renamed to
their canonical short names in 2026-05. The old names remain importable as
``deprecated_alias``-wrapped callables that emit a one-shot
:class:`DeprecationWarning` on first call. They will be removed in tengri v1.0.

Plan: docs/dev/api_migration_v0.x.md, ~/.claude/plans/i-want-you-to-lazy-acorn.md
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import pytest

# (old_name, canonical_name) — every renamed alias.
RENAMED_PAIRS = [
    ("constant_sfh", "constant"),
    ("exponential_sfh", "exponential"),
    ("delayed_exponential_sfh", "delayed_exponential"),
    ("gaussian_sfh", "gaussian"),
    ("lognormal_sfh", "lognormal"),
    ("powerlaw_sfh", "powerlaw"),
    ("skewnormal_sfh", "skewnormal"),
    ("snorm_burst_sfh", "snorm_burst"),
    ("snorm_trunc_burst_sfh", "snorm_trunc_burst"),
    ("spline_sfh", "spline"),
    ("truncated_skewnormal_sfh", "truncated_skewnormal"),
    ("dense_basis_sfh", "dense_basis"),
    ("dense_basis_pure_sfh", "dense_basis_pure"),
    ("dirichlet_sfh", "dirichlet"),
    ("continuity_sfh", "continuity"),
    ("continuity_flex_sfh", "continuity_flex"),
    ("psb_continuity_sfh", "psb_continuity"),
    ("declining_exponential_sfh", "declining_exponential"),
    ("constant_then_exponential_sfh", "constant_then_exponential"),
]


@pytest.mark.parametrize(("old_name", "new_name"), RENAMED_PAIRS)
def test_old_and_new_resolve_from_canonical_namespace(old_name: str, new_name: str) -> None:
    """Both names must resolve from `tengri.components.stellar.sfh`."""
    import tengri.components.stellar.sfh as sfh_mod

    assert hasattr(sfh_mod, new_name), f"canonical name {new_name!r} missing"
    assert hasattr(sfh_mod, old_name), f"deprecated name {old_name!r} missing"


@pytest.mark.parametrize(("old_name", "new_name"), RENAMED_PAIRS)
def test_canonical_names_emit_no_deprecation_warning(old_name: str, new_name: str) -> None:
    """Looking up the canonical name must not warn."""
    import tengri.components.stellar.sfh as sfh_mod

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        getattr(sfh_mod, new_name)
    deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert not deprecations, (
        f"{new_name} emitted DeprecationWarning: {[str(w.message) for w in deprecations]}"
    )


# Subset that we can call cheaply with simple positional arguments.
# Keeps the parametrize matrix small while still verifying the wrapper
# fires on actual invocation, not just attribute access.
CALLABLE_PROBES = [
    # (old_name, args)
    ("constant_sfh", (jnp.linspace(0.0, 1e10, 16), 0.0, 1e10)),
    ("declining_exponential_sfh", (jnp.linspace(0.0, 1e10, 16), 0.0, 1e9, 1e10)),
    ("constant_then_exponential_sfh", (jnp.linspace(0.0, 1e10, 16), 0.0, 1e9, 5e9, 1e10)),
]


@pytest.mark.parametrize(("old_name", "args"), CALLABLE_PROBES)
def test_deprecated_call_emits_warning(old_name: str, args: tuple) -> None:
    """Calling the deprecated alias must emit a DeprecationWarning naming it."""
    import tengri.components.stellar.sfh as sfh_mod

    fn = getattr(sfh_mod, old_name)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fn(*args)
    deprecations = [
        w
        for w in caught
        if issubclass(w.category, DeprecationWarning) and old_name in str(w.message)
    ]
    assert deprecations, f"calling {old_name} did not emit DeprecationWarning"


def test_registry_uses_canonical_names_internally() -> None:
    """SFH_REGISTRY must reference the canonical (short) functions, not aliases.

    If the registry held a deprecated_alias wrap, every fit using these
    models would emit a DeprecationWarning per call — which is not what we
    want. The registry stores callables internally; only user-facing names
    should go through the deprecation shim.
    """
    from tengri.components.stellar.sfh.registry import SFH_REGISTRY

    tau = SFH_REGISTRY["tau"]
    const_exp = SFH_REGISTRY["const_exp"]
    assert tau.fn.__name__ == "declining_exponential"
    assert const_exp.fn.__name__ == "constant_then_exponential"
