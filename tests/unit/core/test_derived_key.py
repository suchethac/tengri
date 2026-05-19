"""Shape, equality, and hashability tests for :class:`DerivedKey`.

These tests cover the new typed cross-component contract introduced
alongside ADR-0004. ``DerivedKey`` is a ``NamedTuple``; most of these
properties come for free but we assert them anyway so future refactors
(e.g. switching to a frozen dataclass) cannot silently break consumers
that rely on tuple semantics.
"""

from __future__ import annotations

import pytest

from tengri.protocols.component import (
    DerivedKey,
    PipelineContractError,
)


class TestDerivedKeyShape:
    def test_named_tuple_fields(self):
        k = DerivedKey("L_ir", "erg/s", "Integrated dust-absorbed luminosity")
        assert k.name == "L_ir"
        assert k.units == "erg/s"
        assert k.description == "Integrated dust-absorbed luminosity"

    def test_description_optional(self):
        k = DerivedKey("L_ir", "erg/s")
        assert k.description == ""

    def test_positional_access_matches_named(self):
        k = DerivedKey("L_ir", "erg/s", "luminosity")
        # NamedTuple supports both positional and named access.
        assert k[0] == k.name
        assert k[1] == k.units
        assert k[2] == k.description

    def test_equality(self):
        a = DerivedKey("L_ir", "erg/s", "lum")
        b = DerivedKey("L_ir", "erg/s", "lum")
        assert a == b

    def test_inequality_on_units(self):
        a = DerivedKey("L_ir", "erg/s")
        b = DerivedKey("L_ir", "Lsun")
        assert a != b

    def test_hashable(self):
        # Required to use DerivedKey in sets / as dict keys (the validator
        # builds {name: (idx, comp, key)} maps; the keys themselves must
        # be hashable in case they ever land in a set).
        s = {DerivedKey("L_ir", "erg/s"), DerivedKey("L_ir", "erg/s")}
        assert len(s) == 1


class TestPipelineContractError:
    def test_is_value_error(self):
        # ValueError is the canonical base for "this argument is bad"
        # errors in tengri. Subclassing it means callers can catch
        # ValueError to handle both classical and contract-level
        # validation failures uniformly.
        assert issubclass(PipelineContractError, ValueError)

    def test_can_raise_and_catch(self):
        with pytest.raises(PipelineContractError, match="some explanation"):
            raise PipelineContractError("some explanation")
