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
    def test_unknown_key_via_from_dict_lands_in_extras(self):
        b = DerivedBundle.from_dict({"L_ir": 1.0, "future_key": "anything"})
        assert b["L_ir"] == 1.0
        assert b["future_key"] == "anything"
        # The typed field gets the value; the unknown key lands in _extras.
        assert b.L_ir == 1.0
        assert b._extras == {"future_key": "anything"}

    def test_extras_show_up_in_keys_items(self):
        b = DerivedBundle.from_dict({"L_ir": 1.0, "x": 7})
        assert "x" in list(b.keys())
        assert dict(b.items())["x"] == 7

    def test_from_dict_empty(self):
        b = DerivedBundle.from_dict({})
        assert b == DerivedBundle()
        assert b._extras == {}

    def test_to_dict_roundtrip(self):
        d = {"L_ir": 1.0, "sfr": 2.0, "x": 99}
        b = DerivedBundle.from_dict(d)
        back = b.to_dict()
        # to_dict orders typed fields first then extras; compare by content.
        assert back == d


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
