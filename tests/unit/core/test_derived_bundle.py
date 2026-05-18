# SPDX-License-Identifier: BSD-3-Clause
"""Tests for :class:`tengri.core.DerivedBundle`.

Phase 1 of ADR-0007. Covers field shape, dict-compat semantics, the
``with_`` typo hint, ``from_dict`` / ``to_dict`` migration helpers,
the ``_extras`` spillover path, and JAX pytree registration.

These tests assume :class:`DerivedBundle` is a *drop-in* replacement
for the current ``Mapping[str, Any]`` shape on ``PipelineState.derived``
— every existing dict-style read should keep working.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest
from jax import tree_util

from tengri.core import DerivedBundle


class TestFieldShape:
    def test_all_fields_default_to_none(self):
        b = DerivedBundle()
        for name in b.field_names():
            assert getattr(b, name) is None, f"{name} must default to None"

    def test_field_names_include_canonical_keys(self):
        # A handful of canonical keys must be present as typed fields.
        # Anything missing here means the bundle was created without
        # matching the contract — caught at construction in this test.
        canonical = {
            "log_mstar",
            "sfr",
            "L_age",
            "lnu_age",
            "ssp_ages_yr",
            "nion",
            "L_ir",
            "L_agn_bol",
            "sed_nebular",
            "sed_radio",
            "igm_transmission",
        }
        names = set(DerivedBundle.field_names())
        missing = canonical - names
        assert not missing, f"Missing canonical fields: {missing}"

    def test_frozen(self):
        b = DerivedBundle(L_ir=jnp.asarray(1.0))
        with pytest.raises(AttributeError):
            b.L_ir = jnp.asarray(2.0)


class TestWith:
    def test_with_replaces_one_field(self):
        b = DerivedBundle()
        b2 = b.with_(L_ir=jnp.asarray(1.0))
        assert b.L_ir is None  # original unchanged
        assert b2.L_ir is not None

    def test_with_unknown_field_raises_with_hint(self):
        b = DerivedBundle()
        with pytest.raises(TypeError, match=r"L_ie.*Did you mean.*L_ir"):
            b.with_(L_ie=jnp.asarray(1.0))

    def test_with_unknown_field_without_close_match(self):
        b = DerivedBundle()
        with pytest.raises(TypeError, match="totally_unrelated_name"):
            b.with_(totally_unrelated_name=jnp.asarray(1.0))


class TestDictCompat:
    def test_getitem_present(self):
        b = DerivedBundle(L_ir=jnp.asarray(3.0))
        assert float(b["L_ir"]) == 3.0

    def test_getitem_unset_raises_keyerror(self):
        b = DerivedBundle()
        with pytest.raises(KeyError, match="L_ir"):
            _ = b["L_ir"]

    def test_get_with_default(self):
        b = DerivedBundle()
        assert b.get("L_ir", 0.0) == 0.0
        b = DerivedBundle(L_ir=jnp.asarray(2.0))
        assert float(b.get("L_ir", 0.0)) == 2.0

    def test_get_no_default_returns_none(self):
        b = DerivedBundle()
        assert b.get("L_ir") is None

    def test_contains_only_when_populated(self):
        b = DerivedBundle()
        assert "L_ir" not in b
        b = b.with_(L_ir=jnp.asarray(1.0))
        assert "L_ir" in b

    def test_contains_non_string_returns_false(self):
        b = DerivedBundle(L_ir=jnp.asarray(1.0))
        assert 42 not in b

    def test_keys_only_populated_fields(self):
        b = DerivedBundle(L_ir=jnp.asarray(1.0), sfr=jnp.asarray(2.0))
        assert set(b.keys()) == {"L_ir", "sfr"}

    def test_items_and_values(self):
        b = DerivedBundle(L_ir=jnp.asarray(1.0), sfr=jnp.asarray(2.0))
        keys = [k for k, _ in b.items()]
        assert set(keys) == {"L_ir", "sfr"}
        assert len(b.values()) == 2

    def test_iter_and_len(self):
        b = DerivedBundle()
        assert list(b) == []
        assert len(b) == 0
        b = b.with_(L_ir=jnp.asarray(1.0))
        assert list(b) == ["L_ir"]
        assert len(b) == 1


class TestExtrasSpillover:
    """Phase 4 (ADR-0007): the spillover-to-_extras path is opt-in only.

    Default ``from_dict`` raises on unknown keys; the legacy shim is
    available behind ``allow_extras=True`` for migration / debugging.
    """

    def test_unknown_key_via_from_dict_lands_in_extras_when_opted_in(self):
        b = DerivedBundle.from_dict({"L_ir": 1.0, "future_key": "anything"}, allow_extras=True)
        assert b["L_ir"] == 1.0
        assert b["future_key"] == "anything"
        # The typed field gets the value; the unknown key lands in _extras.
        assert b.L_ir == 1.0
        assert b._extras == {"future_key": "anything"}

    def test_extras_show_up_in_keys_items_when_opted_in(self):
        b = DerivedBundle.from_dict({"L_ir": 1.0, "x": 7}, allow_extras=True)
        assert "x" in list(b.keys())
        assert dict(b.items())["x"] == 7

    def test_from_dict_empty(self):
        # Empty dict: no unknown keys to spill, strict mode is happy.
        b = DerivedBundle.from_dict({})
        assert b == DerivedBundle()
        assert b._extras == {}

    def test_to_dict_roundtrip_with_extras(self):
        d = {"L_ir": 1.0, "sfr": 2.0, "x": 99}
        b = DerivedBundle.from_dict(d, allow_extras=True)
        back = b.to_dict()
        # to_dict orders typed fields first then extras; compare by content.
        assert back == d

    def test_from_dict_strict_raises_on_unknown_key(self):
        """Default ``allow_extras=False`` rejects unknown keys with a hint."""
        with pytest.raises(TypeError, match="unknown key"):
            DerivedBundle.from_dict({"L_ir": 1.0, "future_key": "anything"})

    def test_from_dict_strict_hint_for_typo(self):
        """Typo close to a known field is surfaced via Did-you-mean."""
        with pytest.raises(TypeError, match=r"L_IR.*Did you mean.*L_ir"):
            DerivedBundle.from_dict({"L_IR": 1.0})

    def test_from_dict_strict_only_typed_keys_ok(self):
        """Dict with only typed keys passes strict mode."""
        b = DerivedBundle.from_dict({"L_ir": 1.0, "sfr": 2.0})
        assert b.L_ir == 1.0
        assert b.sfr == 2.0
        assert b._extras == {}


class TestPytreeRegistration:
    def test_tree_flatten_unflatten_roundtrip(self):
        b = DerivedBundle(
            L_ir=jnp.asarray(1.0),
            sfr=jnp.asarray(2.0),
        )
        leaves, treedef = tree_util.tree_flatten(b)
        b2 = tree_util.tree_unflatten(treedef, leaves)
        assert isinstance(b2, DerivedBundle)
        assert float(b2.L_ir) == 1.0
        assert float(b2.sfr) == 2.0

    def test_tree_map_doubles_arrays(self):
        b = DerivedBundle(L_ir=jnp.asarray(1.0), sfr=jnp.asarray(2.0))
        # None leaves are filtered out by tree_map by default (in modern JAX,
        # None is a pytree leaf-like value that's handled identically across
        # the tree). Use is_leaf=lambda x: x is None to ride over None safely.
        b2 = tree_util.tree_map(
            lambda x: x * 2 if x is not None else x,
            b,
            is_leaf=lambda x: x is None,
        )
        assert float(b2.L_ir) == 2.0
        assert float(b2.sfr) == 4.0
        # Unset fields stay None.
        assert b2.log_mstar is None


class TestPackageRootExport:
    def test_at_tengri_core(self):
        import tengri.core as core

        assert core.DerivedBundle is DerivedBundle


class TestPhase2Flip:
    """PipelineState.derived is now DerivedBundle-typed (ADR-0007 Phase 2).

    These tests verify the type flip + auto-coercion shim. They live
    in the bundle test file (not pipeline_state's) because the bundle
    is the type that matters; PipelineState is just its host.
    """

    def test_default_derived_is_a_bundle(self):
        from tengri.core import PipelineState

        s = PipelineState(wave=jnp.linspace(1000.0, 10000.0, 8))
        assert isinstance(s.derived, DerivedBundle)
        # All fields unset on a fresh state.
        assert len(s.derived) == 0

    def test_construct_with_dict_coerces_to_bundle(self):
        # Legacy callers that pass derived={"L_ir": ...} at construction
        # get a DerivedBundle automatically via __post_init__.
        from tengri.core import PipelineState

        s = PipelineState(
            wave=jnp.linspace(1000.0, 10000.0, 8),
            derived={"L_ir": jnp.asarray(3.0)},
        )
        assert isinstance(s.derived, DerivedBundle)
        assert float(s.derived["L_ir"]) == 3.0

    def test_with_dict_coerces_to_bundle(self):
        # The legacy write pattern — dict(state.derived) → mutate →
        # state.with_(derived=new_dict) — must still work.
        from tengri.core import PipelineState

        s = PipelineState(wave=jnp.linspace(1000.0, 10000.0, 8))
        new_derived = dict(s.derived)
        new_derived["L_ir"] = jnp.asarray(7.0)
        s2 = s.with_(derived=new_derived)
        assert isinstance(s2.derived, DerivedBundle)
        assert float(s2.derived["L_ir"]) == 7.0
        # Original unchanged (immutability invariant preserved).
        assert "L_ir" not in s.derived

    def test_with_bundle_passes_through(self):
        # New-style write: pass a DerivedBundle directly. No coercion
        # needed, identity preserved by ``replace``.
        from tengri.core import PipelineState

        s = PipelineState(wave=jnp.linspace(1000.0, 10000.0, 8))
        b = DerivedBundle(L_ir=jnp.asarray(9.0))
        s2 = s.with_(derived=b)
        assert s2.derived is b

    def test_unknown_key_in_dict_raises_after_phase4(self):
        # Phase 4 (ADR-0007): the spillover-to-_extras path is closed
        # for production code. PipelineState.__post_init__ now calls
        # DerivedBundle.from_dict(..., allow_extras=False), so passing
        # a stale dict-style write with an unknown key fails loudly.
        from tengri.core import PipelineState

        with pytest.raises(TypeError, match="unknown key"):
            PipelineState(
                wave=jnp.linspace(1000.0, 10000.0, 8),
                derived={"L_ir": jnp.asarray(1.0), "future_key": "anything"},
            )

    def test_typed_dict_still_coerces_after_phase4(self):
        """The happy dict-style path — all keys are typed — still works."""
        from tengri.core import PipelineState

        s = PipelineState(
            wave=jnp.linspace(1000.0, 10000.0, 8),
            derived={"L_ir": jnp.asarray(1.0), "sfr": jnp.asarray(2.0)},
        )
        assert s.derived["L_ir"] == 1.0
        assert s.derived["sfr"] == 2.0
        assert s.derived._extras == {}
