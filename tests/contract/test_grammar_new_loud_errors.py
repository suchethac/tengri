# SPDX-License-Identifier: BSD-3-Clause
"""New loud errors from the FIXED-removal / Fixed(DEFAULT) grammar redesign.

Four grammar-validation cases introduced alongside the FIXED sentinel's
removal (pre-1.0 clean break), none previously covered by a contract test:

- A bare sentinel/token given directly as a group's value (``sfh=FREE``,
  carried over from the flat-kwargs ``Parameters(...)`` constructor, where
  that spelling is legal) raises ``ParameterError`` naming the correct,
  nested-dict spelling.
- ``foreground`` declares no fitted parameters of its own (a settings-only MW
  screen, see ``_translate_foreground``), so its wildcard has nothing to
  govern -- ``foreground={'all_params': FREE}`` raises ``ParameterError``.
- ``foreground`` is key-validated like every other group: a typo'd settings
  key gets a "did you mean" suggestion rather than silently vanishing.
- A concrete ``Fixed(v)`` cannot be the ``'all_params'``/``'other_params'``
  wildcard value: one literal value cannot apply across different
  parameters, so the dict-grammar (not just the builder factories) raises
  naming ``Fixed(DEFAULT)`` as the only accepted Fixed spelling there.
"""

from __future__ import annotations

import pytest

from tengri import DEFAULT, FREE, Fixed, parse_groups
from tengri.config.exceptions import ParameterError

pytestmark = pytest.mark.contract


class TestBareGroupValueRaises:
    """A bare wildcard sentinel as a group's own value, not inside its dict."""

    def test_sfh_equals_free_raises(self):
        with pytest.raises(ParameterError, match=r"the wildcard goes inside the group dict"):
            parse_groups(sfh=FREE, redshift=Fixed(0.1))

    def test_sfh_equals_free_names_the_fix(self):
        with pytest.raises(ParameterError, match=r"sfh=\{'all_params': FREE\}"):
            parse_groups(sfh=FREE, redshift=Fixed(0.1))

    def test_sfh_equals_fixed_default_raises(self):
        with pytest.raises(ParameterError, match=r"the wildcard goes inside the group dict"):
            parse_groups(sfh=Fixed(DEFAULT), redshift=Fixed(0.1))

    def test_sfh_equals_bare_default_names_fixed_default(self):
        """``sfh=DEFAULT`` is not the wildcard slip above -- DEFAULT is only
        ever legal wrapped in Fixed(...), so the message points there instead."""
        with pytest.raises(ParameterError, match=r"Fixed\(DEFAULT\)"):
            parse_groups(sfh=DEFAULT, redshift=Fixed(0.1))


class TestForegroundDeclaresNoParameters:
    """``foreground`` is a settings-only MW screen with no wildcard to govern."""

    def test_all_params_free_raises(self):
        with pytest.raises(ParameterError, match=r"declares no parameters"):
            parse_groups(foreground={"all_params": FREE}, redshift=Fixed(0.1))

    def test_all_params_fixed_default_raises_too(self):
        """The rejection is about the wildcard existing at all, not FREE specifically."""
        with pytest.raises(ParameterError, match=r"declares no parameters"):
            parse_groups(foreground={"all_params": Fixed(DEFAULT)}, redshift=Fixed(0.1))

    def test_typo_key_gets_a_suggestion(self):
        """``foreground`` is key-validated like every other group: a typo'd
        settings key ('ebmvv_mw' for 'ebmv_mw') is not silently dropped."""
        with pytest.raises(ValueError, match=r"Did you mean: ebmv_mw\?"):
            parse_groups(foreground={"ebmvv_mw": 0.1}, redshift=Fixed(0.1))

    def test_valid_foreground_settings_still_work(self):
        """The validation above must not have broken the legitimate spelling."""
        spec = parse_groups(
            foreground={"ebmv_mw": 0.05, "law": "cardelli", "rv": 3.1}, redshift=Fixed(0.1)
        )
        assert spec.foreground_ebmv_mw == 0.05
        assert spec.foreground_law == "cardelli"
        assert spec.foreground_rv == 3.1


class TestConcreteFixedCannotSmearAcrossTheWildcard:
    """A literal Fixed(v) cannot be 'all_params': one value, many parameters."""

    def test_dict_grammar_rejects_concrete_fixed_wildcard(self):
        with pytest.raises(ValueError, match=r"one literal value cannot apply"):
            parse_groups(
                sfh={"type": "dpl", "all_params": Fixed(0.5)},
                redshift=Fixed(0.1),
            )

    def test_message_names_fixed_default_as_the_accepted_spelling(self):
        with pytest.raises(ValueError, match=r"must be FREE or Fixed\(DEFAULT\)"):
            parse_groups(
                sfh={"type": "dpl", "all_params": Fixed(0.5)},
                redshift=Fixed(0.1),
            )

    def test_other_params_synonym_rejects_it_too(self):
        """The synonym spelling hits the identical check -- not a second path."""
        with pytest.raises(ValueError, match=r"one literal value cannot apply"):
            parse_groups(
                sfh={"type": "dpl", "other_params": Fixed(0.5)},
                redshift=Fixed(0.1),
            )
