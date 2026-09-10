# SPDX-License-Identifier: BSD-3-Clause
"""``suggest_parameters``'s dust-law defaults mirror ``Parameters()`` (#2224).

``suggest_parameters(dust_law_bc=...)`` defaulted to ``"power_law"``
regardless of ``dust_model``, so a single-component cheatsheet request that
also passed a disagreeing ``dust_law_diff`` printed a table for a
configuration ``Parameters()`` itself would refuse. The default is now
``None`` (mirroring ``Parameters()``'s own flat-kwarg default of "unset"),
and the resolved pair is run through the same
:func:`tengri.parameters._dust_laws.resolve_dust_screen_laws` rule that
``Parameters._init_dust_config`` uses, so the cheatsheet and the constructor
cannot disagree about what a given ``(dust_model, dust_law_bc,
dust_law_diff)`` triple means.
"""

from __future__ import annotations

import inspect

import pytest

from tengri.registry import suggest_parameters

pytestmark = pytest.mark.contract


class TestSuggestParametersDustLawDefault:
    def test_dust_law_bc_default_is_none(self):
        default = inspect.signature(suggest_parameters).parameters["dust_law_bc"].default
        assert default is None

    def test_single_component_disagreeing_pair_raises(self, capsys):
        with pytest.raises(ValueError) as exc_info:
            suggest_parameters(
                dust_model="single_component",
                dust_law_bc="calzetti",
                dust_law_diff="narayanan_z",
            )
        message = str(exc_info.value)
        assert "dust_law_bc" in message
        assert "calzetti" in message
        assert "narayanan_z" in message

    def test_single_component_bc_only_still_resolves(self, capsys):
        table = suggest_parameters(dust_model="single_component", dust_law_bc="calzetti")
        capsys.readouterr()
        assert table is not None

    def test_two_component_defaults_unchanged(self, capsys):
        """No dust_law_bc/diff at all: two_component still defaults power_law/power_law."""
        table = suggest_parameters(dust_model="two_component")
        capsys.readouterr()
        assert table is not None


class TestSuggestParametersDustLawAppliesToBothScreens:
    """``dust_law=`` names ONE law for BOTH screens (regression, same session).

    ``dust_law`` is documented as "use 'law' for single_component" -- i.e. it
    is the single-law spelling, not a diffuse-only one. ``suggest_parameters``
    used to fold a lone ``dust_law`` into ``dust_law_diff`` only ("Treat
    single dust_law= as the diffuse component"), so
    ``suggest_parameters(dust_model="single_component", dust_law="calzetti")``
    reached the resolver as ``(dust_law_bc=None, dust_law_diff="calzetti")``,
    which #2224 correctly refuses for a single-screen model -- a regression
    introduced by that same fix. ``dust_law`` must fill whichever screen the
    caller left unnamed, on both ``dust_model`` variants.
    """

    def test_single_component_dust_law_does_not_raise(self, capsys):
        table = suggest_parameters(dust_model="single_component", dust_law="calzetti")
        banner = capsys.readouterr().out
        assert table is not None
        assert "dust_law_bc='calzetti'" in banner
        assert "dust_law_diff=" not in banner

    def test_two_component_dust_law_sets_equal_pair(self, capsys):
        table = suggest_parameters(dust_law="calzetti")
        banner = capsys.readouterr().out
        assert table is not None
        assert "dust_law_bc='calzetti'" in banner
        assert "dust_law_diff=" not in banner

    def test_two_component_dust_law_with_explicit_diff_overrides_partner_only(self, capsys):
        table = suggest_parameters(dust_law="calzetti", dust_law_diff="smc")
        banner = capsys.readouterr().out
        assert table is not None
        assert "dust_law_bc='calzetti'" in banner
        assert "dust_law_diff='smc'" in banner
