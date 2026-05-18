# SPDX-License-Identifier: BSD-3-Clause
"""Tests for the introspection registry over per-component ``_params.py``.

Covers the shape of :class:`ParameterRecord`, the cache behaviour of
:func:`registry`, the filtering of :func:`list_parameters`, and the
"Did you mean" hint of :func:`describe_parameter`.

Phase 1 of ADR-0005.
"""

from __future__ import annotations

import pytest

from tengri.core.component import ParamDeclaration
from tengri.parameters.registry import (
    ParameterRecord,
    _clear_cache,
    describe_parameter,
    list_parameters,
    registry,
)


class TestRegistryShape:
    def test_named_tuple_fields(self):
        rec = ParameterRecord(name="x", prior=None, description="", owner="m", group="PARAMS")
        assert rec.name == "x"
        assert rec.owner == "m"

    def test_registry_returns_dict_of_records(self):
        reg = registry()
        assert isinstance(reg, dict)
        for k, v in list(reg.items())[:5]:
            assert isinstance(k, str)
            assert isinstance(v, ParameterRecord)
            assert v.name == k

    def test_registry_is_nonempty(self):
        # At minimum, the canonical parameters declared by every
        # currently-shipping physics block must be present.
        names = set(registry().keys())
        assert "redshift" in names or any(n == "redshift" for n in names)
        # Domain parameters from at least three different components:
        assert any(n.startswith("dust_") for n in names)
        assert any(n.startswith("radio_") for n in names)
        assert any(n.startswith("agn_") for n in names)


class TestListParameters:
    def test_no_filter_returns_all_sorted(self):
        ls = list_parameters()
        assert ls == sorted(ls)
        assert set(ls) == set(registry().keys())

    def test_prefix_filter(self):
        ls = list_parameters(prefix="radio_")
        assert all(n.startswith("radio_") for n in ls)
        assert len(ls) >= 3  # radio has at least 3 declared parameters

    def test_unknown_prefix_returns_empty(self):
        assert list_parameters(prefix="nonexistent_") == []


class TestDescribeParameter:
    def test_known_param_returns_record(self):
        # ``redshift`` is the canonical bare-name allowlist entry; any
        # registry that doesn't include it is broken.
        rec = describe_parameter("redshift")
        assert isinstance(rec, ParameterRecord)
        assert rec.name == "redshift"
        # ``owner`` must be a real importable module — either a
        # component _params.py (after migration) or the legacy
        # _param_defs.py (current state, gap noted in ADR-0005).
        assert rec.owner.startswith("tengri.")

    def test_unknown_param_raises_key_error(self):
        with pytest.raises(KeyError, match="No parameter named"):
            describe_parameter("not_a_real_parameter_xyz")

    def test_typo_gets_did_you_mean(self):
        # 'log_mstar' isn't a free parameter, but it's a derived key
        # name. The closest free parameter name should still be
        # suggested if within edit-distance 2. Use a deliberately close
        # typo of a known parameter to exercise the hint path.
        known = list_parameters(prefix="dust_")
        if not known:
            pytest.skip("no dust_ parameters in this build")
        target = known[0]
        # Mutate one character so the typo is edit-distance 1.
        typo = target[:-1] + ("z" if target[-1] != "z" else "y")
        with pytest.raises(KeyError, match="Did you mean"):
            describe_parameter(typo)


class TestCacheBehaviour:
    def test_repeated_calls_return_same_object(self):
        a = registry()
        b = registry()
        assert a is b  # cached, not rebuilt

    def test_clear_cache_forces_rebuild(self):
        a = registry()
        _clear_cache()
        b = registry()
        assert a is not b
        assert a == b  # same content though


class TestPackageRootExports:
    def test_list_parameters_at_package_root(self):
        import tengri

        assert tengri.list_parameters is list_parameters
        assert tengri.describe_parameter is describe_parameter
        assert tengri.ParameterRecord is ParameterRecord


class TestRegistryConsistency:
    """Sanity checks: every declared parameter agrees with its source."""

    def test_every_record_resolves_to_a_real_declaration(self):
        """For every component-_params record, importing the owner
        module and grabbing ``getattr(mod, group)`` must yield a tuple
        containing a ``ParamDeclaration`` with the same name and
        description.

        Legacy ``_NON_SFH_PARAMS`` records use a dict-of-4-tuples shape
        and are checked separately by the consistency below.
        """
        import importlib

        # Sample 30 records; only consistency-check component _params.
        records = list(registry().items())[:30]
        for name, rec in records:
            if rec.group == "_NON_SFH_PARAMS":
                # Legacy bucket: dict[name, (description, ...)]
                mod = importlib.import_module(rec.owner)
                bucket = getattr(mod, rec.group)
                assert name in bucket
                assert bucket[name][0] == rec.description
                continue
            mod = importlib.import_module(rec.owner)
            group = getattr(mod, rec.group)
            assert isinstance(group, tuple)
            match = [d for d in group if isinstance(d, ParamDeclaration) and d.name == name]
            assert len(match) == 1, f"{name} not found in {rec.owner}.{rec.group}"
            assert match[0].description == rec.description
