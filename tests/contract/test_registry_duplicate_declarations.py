# SPDX-License-Identifier: BSD-3-Clause
"""First-wins duplicate handling in the parameter introspection registry (ADR-0008).

The registry keeps the first declaration it meets for a name, but a later
declaration that disagrees on ``prior`` or ``free_prior`` is a real conflict
and must be announced with a ``RuntimeWarning`` naming both owners. An
identical re-declaration stays silent.
"""

from __future__ import annotations

import warnings

import pytest

pytestmark = pytest.mark.contract

from tengri.parameters import registry as registry_module
from tengri.parameters.priors import Fixed, LogNormal, Uniform
from tengri.protocols.component import ParamDeclaration


def _decl(prior, free_prior=None, name="synthetic_knob"):
    return ParamDeclaration(
        name,
        prior,
        "synthetic knob for duplicate-handling tests",
        None,
        "",
        free_prior=free_prior,
    )


def _fresh_out():
    return {}


class TestConflictingDuplicatesWarn:
    def test_differing_prior_warns_and_names_both_owners(self):
        out = _fresh_out()
        registry_module._add_declaration(
            out, _decl(Uniform(0.0, 1.0, "k", units="")), owner="mod.a", group="PARAMS"
        )
        with pytest.warns(RuntimeWarning) as caught:
            registry_module._add_declaration(
                out, _decl(Uniform(0.0, 2.0, "k", units="")), owner="mod.b", group="EXTRA"
            )
        message = str(caught[0].message)
        assert "synthetic_knob" in message
        assert "mod.a.PARAMS" in message
        assert "mod.b.EXTRA" in message

    def test_first_declaration_is_kept(self):
        out = _fresh_out()
        first = Uniform(0.0, 1.0, "k", units="")
        registry_module._add_declaration(out, _decl(first), owner="mod.a", group="PARAMS")
        with pytest.warns(RuntimeWarning):
            registry_module._add_declaration(out, _decl(Fixed(5.0)), owner="mod.b", group="PARAMS")
        assert out["synthetic_knob"].prior is first
        assert out["synthetic_knob"].owner == "mod.a"

    def test_differing_free_prior_warns(self):
        out = _fresh_out()
        registry_module._add_declaration(
            out,
            _decl(Fixed(1.0), free_prior=LogNormal(mu=0.0, sigma=0.05)),
            owner="mod.a",
            group="PARAMS",
        )
        with pytest.warns(RuntimeWarning, match="free_prior"):
            registry_module._add_declaration(
                out,
                _decl(Fixed(1.0), free_prior=Uniform(0.5, 2.0, "k", units="")),
                owner="mod.b",
                group="PARAMS",
            )


class TestIdenticalDuplicatesAreSilent:
    def test_identical_redeclaration_emits_nothing(self):
        out = _fresh_out()
        decl = _decl(Uniform(0.0, 1.0, "k", units=""), free_prior=Uniform(0.0, 3.0, "k", units=""))
        registry_module._add_declaration(out, decl, owner="mod.a", group="PARAMS")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            registry_module._add_declaration(out, decl, owner="mod.a", group="PARAMS")
            registry_module._add_declaration(
                out,
                _decl(
                    Uniform(0.0, 1.0, "k", units=""), free_prior=Uniform(0.0, 3.0, "k", units="")
                ),
                owner="mod.b",
                group="PARAMS",
            )
        assert [w for w in caught if issubclass(w.category, RuntimeWarning)] == []


class TestShippedRegistryIsConflictFree:
    def test_building_the_shipped_registry_emits_no_conflict_warning(self):
        registry_module._clear_cache()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", RuntimeWarning)
                reg = registry_module.registry()
        finally:
            registry_module._clear_cache()
        assert "redshift" in reg
