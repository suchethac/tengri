# SPDX-License-Identifier: BSD-3-Clause
"""Tests for the introspection registry over per-component ``_params.py``.

Covers the shape of :class:`ParameterRecord`, the cache behavior of
:func:`registry`, the filtering of :func:`list_parameters`, and the
"Did you mean" hint of :func:`describe_parameter`.

Phase 1 of ADR-0005.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.contract

from tengri.parameters.registry import (
    ParameterRecord,
    _clear_cache,
    describe_parameter,
    list_parameters,
    registry,
)
from tengri.protocols.component import ParamDeclaration


class TestRegistryShape:
    def test_named_tuple_fields(self):
        rec = ParameterRecord(
            name="x", prior=None, description="", units="", owner="m", group="PARAMS"
        )
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
        # Returns a _RegistryTable since #1285; ``.names()`` is the old shape.
        names = list_parameters().names()
        assert names == sorted(names)
        assert set(names) == set(registry().keys())

    def test_prefix_filter(self):
        names = list_parameters(prefix="radio_").names()
        assert all(n.startswith("radio_") for n in names)
        assert len(names) >= 3  # radio has at least 3 declared parameters

    def test_unknown_prefix_returns_empty(self):
        assert list_parameters(prefix="nonexistent_").names() == []

    def test_rows_carry_description_and_units(self):
        """The point of returning a table: the metadata is no longer thrown away."""
        rows = list_parameters(prefix="dust_")
        assert rows, "no dust_ parameters in this build"
        assert set(rows[0]) >= {"name", "description", "units", "owner"}


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
        known = list_parameters(prefix="dust_").names()
        if not known:
            pytest.skip("no dust_ parameters in this build")
        target = known[0]
        # Mutate one character so the typo is edit-distance 1.
        typo = target[:-1] + ("z" if target[-1] != "z" else "y")
        with pytest.raises(KeyError, match="Did you mean"):
            describe_parameter(typo)


class TestCacheBehavior:
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


class TestSharedParamsMigration:
    """Verify that shared parameters (redshift, met_logzsol, noise_*, etc.)
    are cleanly declared in tengri.parameters._shared.PARAMS and properly
    registered.

    Tests the ADR-0005 follow-up #1 migration: moving _NON_SFH_PARAMS from
    a literal dict in _param_defs.py to a derived view sourced from
    _shared.py.
    """

    def test_redshift_owner_is_shared(self):
        """``redshift`` should be owned by tengri.parameters._shared."""
        rec = describe_parameter("redshift")
        assert rec.owner == "tengri.parameters._shared"
        assert rec.group == "PARAMS"
        assert rec.name == "redshift"

    def test_met_logzsol_owner_is_shared(self):
        """``met_logzsol`` should be owned by tengri.parameters._shared."""
        rec = describe_parameter("met_logzsol")
        assert rec.owner == "tengri.parameters._shared"
        assert rec.group == "PARAMS"
        assert rec.name == "met_logzsol"

    def test_noise_frac_cal_is_registered(self):
        """``noise_frac_cal`` must be in the registry; owner is implementation detail."""
        rec = describe_parameter("noise_frac_cal")
        assert rec.name == "noise_frac_cal"

    def test_all_shared_params_present(self):
        """All five canonical shared parameters are registered (owner location may vary)."""
        shared_names = {"redshift", "met_logzsol", "noise_frac_cal", "noise_dof", "sigma_v_kms"}
        registry_names = set(registry().keys())
        assert shared_names.issubset(registry_names)

    def test_shared_params_have_correct_descriptions(self):
        """Descriptions for the canonical shared parameters."""
        expected = {
            "redshift": "Source redshift",
            "met_logzsol": "log10(Z/Zsun)",
            "sigma_v_kms": (
                "Stellar velocity dispersion sigma_v [km/s], added in quadrature "
                "to the instrumental LSF when computing spectra"
            ),
        }
        for name, desc in expected.items():
            rec = describe_parameter(name)
            assert rec.description == desc
