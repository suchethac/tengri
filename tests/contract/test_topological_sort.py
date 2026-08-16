# SPDX-License-Identifier: BSD-3-Clause
"""Stable topological sort over the publish/require dependency graph.

Covers ADR-0006: ordering is *derived* from declared dependencies, not
hand-coded. The sort is stable — components with no ordering constraint
preserve their input order, which is how the canonical pipeline retains
its byte-for-byte ordering and the existing snapshot baselines.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

pytestmark = pytest.mark.contract

from tengri.forward.orchestrator import topological_sort
from tengri.protocols.component import (
    ComponentIOError,
    DerivedKey,
    SEDComponentConfig,
)


@dataclass(frozen=True)
class FakeComponent:
    """Minimal stand-in mirroring the validator test fixture."""

    name: str = "fake"
    parameter_prefix: str = "fake_"
    config: SEDComponentConfig = field(default_factory=SEDComponentConfig)
    _publishes: tuple[DerivedKey, ...] = ()
    _requires: tuple[DerivedKey, ...] = ()
    _requires_optional: tuple[DerivedKey, ...] = ()

    def declared_parameters(self):
        return []

    def outputs(self):
        return self._publishes

    def inputs(self):
        return self._requires

    def optional_inputs(self):
        return self._requires_optional


def _mk(class_name: str, *, publishes=(), requires=(), requires_optional=()):
    cls = type(class_name, (FakeComponent,), {})
    return cls(
        _publishes=publishes,
        _requires=requires,
        _requires_optional=requires_optional,
    )


def _names(components) -> list[str]:
    return [type(c).__name__ for c in components]


class TestStability:
    def test_empty_list(self):
        assert topological_sort([]) == []

    def test_no_deps_preserves_input_order(self):
        a = _mk("FakeA")
        b = _mk("FakeB")
        c = _mk("FakeC")
        assert _names(topological_sort([a, b, c])) == ["FakeA", "FakeB", "FakeC"]
        # Reordering input → reordering output (stable, not "smart"):
        assert _names(topological_sort([c, b, a])) == ["FakeC", "FakeB", "FakeA"]


class TestHardRequires:
    def test_consumer_moves_after_publisher_when_input_is_reversed(self):
        # Consumer before publisher on input → sort moves publisher first.
        pub = _mk("FakeDust", publishes=(DerivedKey("L_ir", "erg/s"),))
        con = _mk("FakeNeb", requires=(DerivedKey("L_ir", "erg/s"),))
        assert _names(topological_sort([con, pub])) == ["FakeDust", "FakeNeb"]
        # And preserves correct input order:
        assert _names(topological_sort([pub, con])) == ["FakeDust", "FakeNeb"]

    def test_chain_of_three(self):
        a = _mk("FakeStellar", publishes=(DerivedKey("lnu_age", "erg/s/Hz"),))
        b = _mk(
            "FakeNebular",
            publishes=(DerivedKey("sed_nebular", "erg/s/Hz"),),
            requires=(DerivedKey("lnu_age", "erg/s/Hz"),),
        )
        c = _mk(
            "FakeRadio",
            publishes=(DerivedKey("sed_radio", "erg/s/Hz"),),
            requires=(DerivedKey("sed_nebular", "erg/s/Hz"),),
        )
        # Any input order → same output: a, b, c.
        for inp in [[a, b, c], [c, b, a], [b, a, c]]:
            assert _names(topological_sort(inp)) == [
                "FakeStellar",
                "FakeNebular",
                "FakeRadio",
            ]


class TestOptionalRequires:
    def test_optional_with_publisher_present_orders_correctly(self):
        # Optional reads also establish ordering — consumer must come
        # after publisher so the read returns the actual value.
        pub = _mk("FakeDust", publishes=(DerivedKey("L_ir", "erg/s"),))
        con = _mk("FakeRadio", requires_optional=(DerivedKey("L_ir", "erg/s"),))
        assert _names(topological_sort([con, pub])) == ["FakeDust", "FakeRadio"]

    def test_optional_with_no_publisher_is_inert(self):
        # No publisher → optional declaration is purely metadata; no
        # ordering constraint. Stable order preserved.
        con = _mk("FakeRadio", requires_optional=(DerivedKey("L_ir", "erg/s"),))
        a = _mk("FakeA")
        assert _names(topological_sort([con, a])) == ["FakeRadio", "FakeA"]


class TestCycleDetection:
    def test_two_node_cycle_raises(self):
        a = _mk(
            "FakeA",
            publishes=(DerivedKey("k1", "dex"),),
            requires=(DerivedKey("k2", "dex"),),
        )
        b = _mk(
            "FakeB",
            publishes=(DerivedKey("k2", "dex"),),
            requires=(DerivedKey("k1", "dex"),),
        )
        with pytest.raises(ComponentIOError, match="cycle"):
            topological_sort([a, b])


class TestRealPipeline:
    """The canonical build_components pipeline must reproduce its
    hand-coded order under the sort.

    This was described as "the unit version", with
    tests/integration/test_derived_contract_snapshots.py as the end-to-end
    counterpart. That file was deleted in #1029, so there is no end-to-end
    version any more and this class is the whole of the check."""

    def test_stellar_nebular_dust_radio_xray_igm_order_preserved(self):
        from tengri.components.dust.component import DustAttenuationSEDComponent
        from tengri.components.igm.component import IGMSEDComponent
        from tengri.components.nebular.component import NebularSEDComponent
        from tengri.components.radio.component import RadioSEDComponent
        from tengri.components.stellar.component import StellarSEDComponent
        from tengri.components.xray.component import XRaySEDComponent

        # Approximate the hand-coded order in build_components.
        chain = [
            StellarSEDComponent(),
            NebularSEDComponent(),
            DustAttenuationSEDComponent(),
            RadioSEDComponent(),
            XRaySEDComponent(),
            IGMSEDComponent(),
        ]
        sorted_chain = topological_sort(chain)
        # Order must be preserved bit-identically.
        assert _names(sorted_chain) == _names(chain)
