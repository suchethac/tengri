# SPDX-License-Identifier: BSD-3-Clause
"""``dust_model="single_component"`` silently discarded ``dust_law_diff`` (#2224).

``_init_dust_config`` honored only ``dust_law_bc`` on a single-component
screen: a flat ``Parameters(dust_model="single_component",
dust_law_diff="narayanan_z", ...)`` built a model whose one screen used
``power_law`` (the flat-kwarg default), silently dropping the caller's law and
building physics nobody asked for -- the same silent-discard shape as #2185
(a per-screen key the law never reads), but at the screen-selection layer
instead of inside a law's own kwargs.

The single owner of the (dust_law_bc, dust_law_diff) pair for a given
``dust_model`` is now :func:`tengri.parameters._dust_laws.resolve_dust_screen_laws`,
called from both :meth:`Parameters._init_dust_config` (flat-kwarg path) and
:func:`tengri.parameters.groups._translate_dust_attenuation` (grammar path,
via the equal pair it already writes for ``single_component``). This file
pins:

1. a lone ``dust_law_diff`` on ``single_component`` now raises, naming
   ``dust_law_bc`` as the fix (was: silently built ``power_law``/``power_law``);
2. a disagreeing ``(dust_law_bc, dust_law_diff)`` pair on ``single_component``
   now raises, naming both laws (was: silently discarded ``dust_law_diff`` and
   built ``dust_law_bc``/``dust_law_bc``);
3. the working shapes are unaffected: ``dust_law_bc`` alone still inherits
   into ``dust_law_diff`` on the flat path, and the grammar's own
   already-equal ``(law, law)`` pair still builds;
4. ``two_component`` inheritance (#1989) is unchanged in both directions.
"""

from __future__ import annotations

import pytest

from tengri.parameters.groups import parse_groups
from tengri.parameters.parameters import Parameters
from tengri.parameters.priors import Fixed

pytestmark = pytest.mark.regression_bug


class TestSingleComponentFlatKwargDiscardRaises:
    """Flat ``Parameters(dust_model="single_component", ...)`` law resolution."""

    def test_diff_only_raises_naming_dust_law_bc(self):
        """A lone ``dust_law_diff`` used to be silently discarded (#2224)."""
        with pytest.raises(ValueError) as exc_info:
            Parameters(
                mean_sfh_type="dpl",
                dust_model="single_component",
                dust_law_diff="narayanan_z",
                dust_tau_v=Fixed(0.3),
                apply_igm=False,
            )
        message = str(exc_info.value)
        assert "dust_law_bc" in message
        assert "narayanan_z" in message

    def test_disagreeing_pair_raises_naming_both(self):
        """``dust_law_bc`` != ``dust_law_diff`` used to silently keep ``dust_law_bc``."""
        with pytest.raises(ValueError) as exc_info:
            Parameters(
                mean_sfh_type="dpl",
                dust_model="single_component",
                dust_law_bc="calzetti",
                dust_law_diff="narayanan_z",
                dust_tau_v=Fixed(0.3),
                apply_igm=False,
            )
        message = str(exc_info.value)
        assert "dust_law_bc" in message
        assert "calzetti" in message
        assert "narayanan_z" in message

    def test_bc_only_still_inherits(self):
        """``dust_law_bc`` alone is still the working single-screen spelling."""
        spec = Parameters(
            mean_sfh_type="dpl",
            dust_model="single_component",
            dust_law_bc="calzetti",
            dust_tau_v=Fixed(0.3),
            apply_igm=False,
        )
        assert spec.dust_law_bc == "calzetti"
        assert spec.dust_law_diff == "calzetti"

    def test_no_law_kwargs_still_defaults_to_power_law(self):
        """No law kwarg at all keeps the long-standing flat-kwarg default."""
        spec = Parameters(
            mean_sfh_type="dpl",
            dust_model="single_component",
            dust_tau_v=Fixed(0.3),
            apply_igm=False,
        )
        assert spec.dust_law_bc == "power_law"
        assert spec.dust_law_diff == "power_law"


class TestSingleComponentGrammarEqualPairStillBuilds:
    """The grammar already writes an equal (law, law) pair -- must stay green."""

    def test_grammar_single_component_law_builds_with_equal_pair(self):
        spec = parse_groups(
            dust_attenuation={"type": "single_component", "law": "calzetti", "tau_v": Fixed(0.3)},
            ssp_data=None,
            redshift=Fixed(0.1),
        )
        assert spec.dust_model == "single_component"
        assert spec.dust_law_bc == "calzetti"
        assert spec.dust_law_diff == "calzetti"


class TestTwoComponentInheritanceUnchanged:
    """#1989: the low-level two_component/off/wg00 inheritance is untouched."""

    def test_bc_only_inherits_into_diff(self):
        spec = Parameters(
            mean_sfh_type="dpl",
            dust_model="two_component",
            dust_law_bc="calzetti",
            dust_tau_bc=Fixed(0.3),
            dust_tau_diff=Fixed(0.2),
            apply_igm=False,
        )
        assert spec.dust_law_bc == "calzetti"
        assert spec.dust_law_diff == "calzetti"

    def test_diff_only_inherits_into_bc(self):
        spec = Parameters(
            mean_sfh_type="dpl",
            dust_model="two_component",
            dust_law_diff="narayanan_z",
            dust_tau_bc=Fixed(0.3),
            dust_tau_diff=Fixed(0.2),
            apply_igm=False,
        )
        assert spec.dust_law_bc == "narayanan_z"
        assert spec.dust_law_diff == "narayanan_z"

    def test_both_given_pass_through_unmodified(self):
        spec = Parameters(
            mean_sfh_type="dpl",
            dust_model="two_component",
            dust_law_bc="calzetti",
            dust_law_diff="power_law",
            dust_tau_bc=Fixed(0.3),
            dust_tau_diff=Fixed(0.2),
            apply_igm=False,
        )
        assert spec.dust_law_bc == "calzetti"
        assert spec.dust_law_diff == "power_law"

    def test_neither_given_defaults_to_power_law_both(self):
        spec = Parameters(
            mean_sfh_type="dpl",
            dust_model="two_component",
            dust_tau_bc=Fixed(0.3),
            dust_tau_diff=Fixed(0.2),
            apply_igm=False,
        )
        assert spec.dust_law_bc == "power_law"
        assert spec.dust_law_diff == "power_law"
