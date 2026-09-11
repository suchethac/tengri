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
    """The flat-Parameters path for two_component: inheritance rules from #1989."""

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


class TestResolverInheritanceAllTwoScreenModels:
    """Direct tests on resolve_dust_screen_laws for two_component/wg00/off models.

    Covers the inheritance rules from #1989 across all two-screen dust models:
    ``dust_model`` in ``("two_component", "wg00", "off")``.
    """

    @pytest.mark.parametrize(
        ("dust_model", "law_bc", "law_diff", "expected_bc", "expected_diff"),
        [
            # Both None -> default to power_law for both
            ("two_component", None, None, "power_law", "power_law"),
            ("wg00", None, None, "power_law", "power_law"),
            ("off", None, None, "power_law", "power_law"),
            # bc only -> inherit into diff
            ("two_component", "calzetti", None, "calzetti", "calzetti"),
            ("wg00", "calzetti", None, "calzetti", "calzetti"),
            ("off", "calzetti", None, "calzetti", "calzetti"),
            # diff only -> inherit into bc
            ("two_component", None, "smc", "smc", "smc"),
            ("wg00", None, "smc", "smc", "smc"),
            ("off", None, "smc", "smc", "smc"),
            # Both given -> pass through unmodified
            ("two_component", "calzetti", "smc", "calzetti", "smc"),
            ("wg00", "calzetti", "smc", "calzetti", "smc"),
            ("off", "calzetti", "smc", "calzetti", "smc"),
        ],
    )
    def test_resolver_inheritance_all_models(
        self, dust_model, law_bc, law_diff, expected_bc, expected_diff
    ):
        """resolve_dust_screen_laws applies inheritance rules to two-screen models."""
        from tengri.parameters._dust_laws import resolve_dust_screen_laws

        resolved_bc, resolved_diff = resolve_dust_screen_laws(dust_model, law_bc, law_diff)
        assert resolved_bc == expected_bc, (
            f"{dust_model}: law_bc={law_bc}, law_diff={law_diff} "
            f"-> got ({resolved_bc!r}, {resolved_diff!r}), "
            f"expected ({expected_bc!r}, {expected_diff!r})"
        )
        assert resolved_diff == expected_diff

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
