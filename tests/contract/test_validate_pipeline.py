# SPDX-License-Identifier: BSD-3-Clause
"""Construction-time contract checks over the publish/require graph.

Covers every error path of
:func:`tengri.forward.orchestrator.validate_pipeline` plus the happy
path. The fixtures construct minimal fake components that satisfy the
:class:`SEDComponent` Protocol shape with just enough surface to be
inspected — they never get applied or precomputed, so no SSP / JAX
state is needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

pytestmark = pytest.mark.contract

from tengri.forward.orchestrator import validate_pipeline
from tengri.protocols.component import (
    DerivedKey,
    PipelineContractError,
    SEDComponentConfig,
)


@dataclass(frozen=True)
class FakeComponent:
    """Minimal :class:`SEDComponent` look-alike for validator tests.

    Carries just the attributes / methods the validator reads. Real
    components inherit from concrete physics modules; these fakes let us
    drive every error path in isolation.
    """

    name: str = "fake"
    parameter_prefix: str = "fake_"
    config: SEDComponentConfig = field(default_factory=SEDComponentConfig)
    _publishes: tuple[DerivedKey, ...] = ()
    _requires: tuple[DerivedKey, ...] = ()

    def declared_parameters(self) -> list[Any]:
        return []

    def outputs(self) -> tuple[DerivedKey, ...]:
        return self._publishes

    def inputs(self) -> tuple[DerivedKey, ...]:
        return self._requires


def _mk(class_name: str, *, publishes=(), requires=()):
    """Make a FakeComponent subclass whose __name__ is ``class_name``.

    The validator's error messages quote ``type(component).__name__``, so
    we need each component in a test to look like a distinct class.
    """
    NewCls = type(class_name, (FakeComponent,), {})
    return NewCls(_publishes=publishes, _requires=requires)


class TestHappyPath:
    def test_empty_pipeline(self):
        validate_pipeline([])

    def test_pipeline_with_only_publishers(self):
        validate_pipeline([_mk("FakePublisher", publishes=(DerivedKey("L_ir", "erg/s"),))])

    def test_publisher_then_consumer(self):
        publisher = _mk("FakeDust", publishes=(DerivedKey("L_ir", "erg/s"),))
        consumer = _mk("FakeRadio", requires=(DerivedKey("L_ir", "erg/s"),))
        validate_pipeline([publisher, consumer])


class TestMissingPublisher:
    def test_missing_key_raises(self):
        consumer = _mk("FakeRadio", requires=(DerivedKey("L_ir", "erg/s"),))
        with pytest.raises(PipelineContractError, match="L_ir"):
            validate_pipeline([consumer])

    def test_did_you_mean_for_likely_typo(self):
        """Levenshtein-based hint catches the silent-rename hazard.

        If dust renames ``L_ir`` to ``L_dust_total`` and radio's
        ``requires`` still says ``L_ir``, the error message must point at
        ``L_dust_total`` so the developer doesn't waste time guessing.
        """
        # Publisher uses 'L_IR' (case-typo, edit distance 2 from 'L_ir').
        # Use the canonical key 'L_ir' for consumer so the canonical-units
        # check doesn't fire first.
        publisher = _mk("FakeDust", publishes=(DerivedKey("L_IR", "erg/s"),))
        consumer = _mk("FakeRadio", requires=(DerivedKey("L_ir", "erg/s"),))
        with pytest.raises(PipelineContractError, match="Did you mean"):
            validate_pipeline([publisher, consumer])


class TestOutOfOrder:
    def test_consumer_before_publisher_raises(self):
        consumer = _mk("FakeRadio", requires=(DerivedKey("L_ir", "erg/s"),))
        publisher = _mk("FakeDust", publishes=(DerivedKey("L_ir", "erg/s"),))
        with pytest.raises(PipelineContractError, match="strictly before"):
            validate_pipeline([consumer, publisher])

    def test_self_publish_is_out_of_order(self):
        """A component reading its own published key in the same step is
        ill-defined — the publisher must come *strictly* before the
        consumer."""
        comp = _mk(
            "FakeSelfRef",
            publishes=(DerivedKey("L_ir", "erg/s"),),
            requires=(DerivedKey("L_ir", "erg/s"),),
        )
        with pytest.raises(PipelineContractError, match="strictly before"):
            validate_pipeline([comp])


class TestDuplicatePublisher:
    def test_two_publishers_of_same_key_raises(self):
        a = _mk("FakeA", publishes=(DerivedKey("L_ir", "erg/s"),))
        b = _mk("FakeB", publishes=(DerivedKey("L_ir", "erg/s"),))
        with pytest.raises(PipelineContractError, match="published by both"):
            validate_pipeline([a, b])

    def test_alternate_publishers_allowed(self):
        """DustSEDComponent and DustAttenuationSEDComponent are registered
        as alternates in ``_ALTERNATE_PUBLISHERS`` — both may declare
        ``L_ir``; in practice the factory picks one at a time."""
        a = _mk("DustSEDComponent", publishes=(DerivedKey("L_ir", "erg/s"),))
        b = _mk("DustAttenuationSEDComponent", publishes=(DerivedKey("L_ir", "erg/s"),))
        # The validator only cares about the pair being in the alternates
        # set, not about whether both are *actually* in the pipeline.
        validate_pipeline([a, b])


class TestUnitsMismatch:
    def test_publisher_consumer_units_disagree(self):
        publisher = _mk("FakeDust", publishes=(DerivedKey("L_ir", "erg/s"),))
        # The consumer says Lsun. Validator must refuse to convert.
        # NOTE: The canonical-units check fires first (Lsun for L_ir
        # violates the table), so we match on either error message.
        consumer = _mk("FakeRadio", requires=(DerivedKey("L_ir", "Lsun"),))
        with pytest.raises(PipelineContractError, match=r"erg/s|Lsun"):
            validate_pipeline([publisher, consumer])

    def test_publisher_violates_canonical_units(self):
        # L_ir is pinned to "erg/s" in _CANONICAL_UNITS. A publisher
        # declaring "Lsun" must fail loudly — that's the half of the
        # contract that prevents a new component from inventing its own
        # units convention.
        publisher = _mk("BadDust", publishes=(DerivedKey("L_ir", "Lsun"),))
        with pytest.raises(PipelineContractError, match="canonical"):
            validate_pipeline([publisher])


class TestNebularRequiresStellar:
    """Phase A of issue #21: Nebular's hard dependency on Stellar.

    Before this contract, a pipeline with Nebular but no Stellar would
    KeyError at JIT trace time on ``state.derived["lnu_age"]``. The hard
    ``requires`` declaration on NebularSEDComponent promotes that
    failure to construction time with a named-key error message.
    """

    def test_real_nebular_without_stellar_raises(self):
        """Build a Cue-backend Nebular with no upstream Stellar publisher."""
        from tengri.components.nebular.component import (
            NebularSEDComponent,
            NebularSEDComponentConfig,
        )

        neb = NebularSEDComponent(config=NebularSEDComponentConfig(backend="cue"))
        with pytest.raises(PipelineContractError, match=r"lnu_age|ssp_ages_yr"):
            validate_pipeline([neb])

    def test_real_baked_in_nebular_alone_is_ok(self):
        """The BakedIn backend reads nothing from ``state.derived``, so
        it should be valid as a single-component pipeline — confirms
        the backend-dependent ``requires()`` branch."""
        from tengri.components.nebular.component import (
            NebularSEDComponent,
            NebularSEDComponentConfig,
        )

        neb = NebularSEDComponent(config=NebularSEDComponentConfig(backend="baked_in"))
        validate_pipeline([neb])

    def test_real_cloudy_grid_also_requires_age_weights(self):
        """CloudyGrid path additionally reads ``age_weights`` to sum
        per-bin grid lookups."""
        from tengri.components.nebular.component import (
            NebularSEDComponent,
            NebularSEDComponentConfig,
        )

        neb = NebularSEDComponent(config=NebularSEDComponentConfig(backend="cloudy_grid"))
        # No publisher → missing-publisher error fires on the first
        # required key (lnu_age) before the validator reaches age_weights,
        # so we match on any of the three to be robust.
        with pytest.raises(PipelineContractError, match=r"lnu_age|ssp_ages_yr|age_weights"):
            validate_pipeline([neb])


class TestRequiresOptional:
    """Phase B of issue #21: opportunistic reads with documented fallbacks.

    The validator does NOT require an upstream publisher for an
    ``requires_optional`` key (consumer falls back), but DOES check
    units if a publisher is present. Catches a future publisher rename
    or unit drift without forcing every pipeline to instantiate the
    optional upstream component.
    """

    def test_missing_publisher_is_ok(self):
        """No upstream publisher → consumer's fallback applies; no error."""
        consumer = type(
            "FakeRadioOptional",
            (FakeComponent,),
            {"requires_optional": lambda self: (DerivedKey("L_ir", "erg/s"),)},
        )()
        validate_pipeline([consumer])

    def test_units_mismatch_with_publisher_raises(self):
        """If a publisher IS present and its units disagree, the
        validator must refuse to silently paper over the drift —
        identical semantics to the hard-required path, just gated on
        publisher presence."""
        publisher = _mk("FakeDust", publishes=(DerivedKey("L_ir", "erg/s"),))
        consumer = type(
            "FakeRadioOptional",
            (FakeComponent,),
            # Same key, different units — only possible if a future
            # contributor edits the consumer's requires_optional in
            # isolation, or if a publisher's units string drifts away
            # from the canonical table. Both are real failure modes.
            {"requires_optional": lambda self: (DerivedKey("L_ir", "Lsun"),)},
        )()
        with pytest.raises(PipelineContractError, match=r"erg/s|Lsun|canonical"):
            validate_pipeline([publisher, consumer])

    def test_publisher_present_matching_units_passes(self):
        """The happy path: publisher exists, units match → silent OK."""
        publisher = _mk("FakeDust", publishes=(DerivedKey("L_ir", "erg/s"),))
        consumer = type(
            "FakeRadioOptional",
            (FakeComponent,),
            {"requires_optional": lambda self: (DerivedKey("L_ir", "erg/s"),)},
        )()
        validate_pipeline([publisher, consumer])

    def test_out_of_order_optional_raises(self):
        """Same strictly-before rule applies to optional reads."""
        consumer = type(
            "FakeRadioOptional",
            (FakeComponent,),
            {"requires_optional": lambda self: (DerivedKey("L_ir", "erg/s"),)},
        )()
        publisher = _mk("FakeDust", publishes=(DerivedKey("L_ir", "erg/s"),))
        with pytest.raises(PipelineContractError, match=r"strictly before"):
            validate_pipeline([consumer, publisher])

    def test_real_radio_alone_validates(self):
        """The real RadioSEDComponent declares L_ir, L_agn_bol, log_mstar
        as optional reads. Constructing radio without upstream publishers
        must still validate cleanly — that's the whole point of the
        optional-read pattern."""
        from tengri.components.radio.component import RadioSEDComponent

        validate_pipeline([RadioSEDComponent()])

    def test_real_xray_alone_validates(self):
        """Same check for XRaySEDComponent (declares sfr, log_mstar,
        L_agn_bol as optional reads)."""
        from tengri.components.xray.component import XRaySEDComponent

        validate_pipeline([XRaySEDComponent()])
