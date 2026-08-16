# SPDX-License-Identifier: BSD-3-Clause
"""Guards on the derived dust-law sweep in ``tests/_dust_laws.py``.

Every property test over dust laws now parametrizes off one derived list. That
removes seven hand-written enumerations, but it concentrates the risk: a
derivation that quietly returned nothing would make every one of those property
tests pass vacuously, and pytest reports zero collected parameters as success,
not as an error.

So the derivation is checked here directly — that it matches the registry, that
it is not empty, and that the two laws the old hand-written lists missed are
present by name, so a regression says which failure recurred.
"""

from __future__ import annotations

import pytest

from tengri.components.dust.attenuation import DUST_LAWS
from tests._dust_laws import (
    GRAIN_MODEL_LAWS,
    every_dust_law,
    law_names,
    swept_names,
)

pytestmark = [pytest.mark.contract, pytest.mark.regression_bug]


class TestTheSweepMatchesTheRegistry:
    def test_it_covers_every_registered_law(self):
        swept = swept_names(every_dust_law())
        assert swept == set(DUST_LAWS), (
            f"sweep and registry disagree — missing "
            f"{sorted(set(DUST_LAWS) - swept)}, unknown {sorted(swept - set(DUST_LAWS))}"
        )

    def test_it_is_not_empty(self):
        """A sweep of nothing passes every property test it feeds."""
        assert len(every_dust_law()) > 15, (
            f"only {len(every_dust_law())} laws discovered — DUST_LAWS stopped "
            f"being enumerable, so every property parametrized off this list "
            f"now proves nothing."
        )

    @pytest.mark.parametrize("name", ["reddy15", "prevot_smc"])
    def test_the_laws_the_hand_written_lists_missed(self, name):
        """``reddy15`` was in neither file's list; ``prevot_smc`` was in only
        one. Named explicitly so a regression says which one came loose."""
        assert name in swept_names(every_dust_law())

    def test_excluding_a_law_removes_exactly_that_law(self):
        """The exclusion hatch must not silently drop more than asked."""
        full = swept_names(every_dust_law())
        trimmed = swept_names(every_dust_law(exclude={"prevot_smc"}))
        assert full - trimmed == {"prevot_smc"}

    def test_law_names_and_the_parametrize_list_agree(self):
        """The two entry points must not drift apart."""
        assert set(law_names()) == swept_names(every_dust_law())


class TestTheGrainModelMarking:
    def test_the_named_grain_models_are_registered(self):
        """A renamed grain law would silently lose its skipif and start
        failing with ImportError again."""
        assert set(DUST_LAWS) >= GRAIN_MODEL_LAWS, (
            f"marked as grain models but not registered: "
            f"{sorted(GRAIN_MODEL_LAWS - set(DUST_LAWS))}"
        )

    def test_grain_models_carry_a_marker_and_others_do_not(self):
        """Marking everything would skip the whole sweep when the optional
        package is absent; marking nothing brings the ImportErrors back."""
        marked = {p.values[0] for p in every_dust_law() if hasattr(p, "values")}
        assert marked == GRAIN_MODEL_LAWS, (
            f"marker applied to the wrong set: expected {sorted(GRAIN_MODEL_LAWS)}, "
            f"got {sorted(marked)}"
        )
