# SPDX-License-Identifier: BSD-3-Clause
"""Tests for the short-name resolution scope in tools/check_param_ranges.py.

The guard resolves a short-form key inside an AGN block by prefixing ``agn_``
and reading the registry. It used to do that for *every* dict literal in the
file, so ``sfh={'type': 'dpl', 'alpha': Uniform(0.5, 2)}`` was measured against
``agn_alpha``'s ``[-2, 0]`` rather than ``sfh_dpl_alpha``'s ``[0.1, 5.0]`` and
reported as disjoint. Nine ALLOWLIST entries accumulated for that one
collision before the resolver was the thing that got fixed.

The false positives were the visible half; the reverse error is what these
tests exist for. An sfh ``alpha`` genuinely outside ``[0.1, 5.0]`` used to be
compared against the AGN declaration, where it could **pass**. A resolver that
ignores context is unreliable in both directions.

So two things are pinned here: that a real AGN violation is still caught
through every syntactic shape a config reaches the builder by, and that a
non-AGN dict is not resolved as one.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

tools_dir = Path(__file__).parent.parent.parent / "tools"
sys.path.insert(0, str(tools_dir))

from check_param_ranges import (
    ALLOWLIST,
    _agn_prior_sites,
    _agn_scoped_dicts,
)

pytestmark = pytest.mark.contract


def _resolved(src: str) -> set[str]:
    """Parameter names the guard's short-name path resolves in ``src``."""
    return {param for param, _call in _agn_prior_sites(ast.parse(src))}


class TestAgnBlockShapes:
    """Every shape a config reaches the builder by must still be scoped.

    A narrower resolver that also stops seeing real AGN priors is a regression,
    not a fix -- these are the shapes measured in the tree when it was narrowed.
    """

    def test_keyword_form(self):
        src = "build(agn={'type': 'composable', 'log_lbol': Uniform(9.0, 12.0)})"
        assert "agn_log_lbol" in _resolved(src)

    def test_fragment_form(self):
        """``{'agn': {...}}`` splatted into the call -- never a keyword itself."""
        src = "GROUPS = {'agn': {'type': 'composable', 'log_lbol': Uniform(9.0, 12.0)}}"
        assert "agn_log_lbol" in _resolved(src)

    def test_subscript_assignment_form(self):
        """Config built by mutation: ``cfg['agn'] = {...}`` (notebook idiom)."""
        src = "cfg = deepcopy(base)\ncfg['agn'] = {'log_lbol': Uniform(9.0, 12.0)}"
        assert "agn_log_lbol" in _resolved(src)

    def test_name_binding_form(self):
        """A named constant passed as ``agn=NAME``."""
        src = "AGN_CFG = {'log_lbol': Uniform(9.0, 12.0)}\nbuild(agn=AGN_CFG)"
        assert "agn_log_lbol" in _resolved(src)

    def test_case_table_form(self):
        """The shape that reaches the builder through a subscript and a local.

        ``_AGN_CASES[key]`` -> local -> ``groups['agn']`` -> ``**groups`` is
        three hops in two functions; the binding name carries it instead.
        """
        src = (
            "_AGN_CASES = {\n"
            "    'skirtor': {'type': 'composable', 'log_lbol': Uniform(9.0, 12.0)},\n"
            "}\n"
        )
        assert "agn_log_lbol" in _resolved(src)

    def test_nested_sub_block_inherits_scope(self):
        """``agn={'disc': {...}}`` -- the sub-block is still the AGN group."""
        src = "build(agn={'disc': {'type': 'multicolor', 'log_lbol': Uniform(9.0, 12.0)}})"
        assert "agn_log_lbol" in _resolved(src)


class TestNonAgnDictsAreNotResolved:
    """The collision that produced nine allowlist entries."""

    def test_sfh_alpha_is_not_agn_alpha(self):
        src = "build(sfh={'type': 'dpl', 'alpha': Uniform(0.5, 2.0)})"
        assert "agn_alpha" not in _resolved(src)

    def test_sfh_alpha_beside_a_real_agn_block(self):
        """Both in one call: the sfh key is skipped, the agn key is not.

        The interesting case -- a file-wide walk cannot tell these apart, which
        is exactly how the false positives arose.
        """
        src = (
            "build(\n"
            "    sfh={'type': 'dpl', 'alpha': Uniform(0.5, 2.0)},\n"
            "    agn={'log_lbol': Uniform(9.0, 12.0)},\n"
            ")"
        )
        resolved = _resolved(src)
        assert "agn_log_lbol" in resolved
        assert "agn_alpha" not in resolved

    def test_dust_delta_is_not_agn_delta(self):
        """``delta`` is a dust attenuation slope and an AGN parameter both."""
        src = "build(dust_attenuation={'law': 'calzetti', 'delta': Uniform(-0.5, 0.5)})"
        assert "agn_delta" not in _resolved(src)

    def test_plain_data_dict_is_not_a_config(self):
        """A filename parser returning ``{'tau': ...}`` is not an AGN block."""
        src = "def parse(m):\n    return {'tau': int(m.group(1)), 'p': float(m.group(2))}"
        assert _resolved(src) == set()


class TestScopeSetItself:
    def test_unrelated_dicts_are_not_scoped(self):
        src = "x = {'a': 1}\ny = {'b': {'c': 2}}"
        assert _agn_scoped_dicts(ast.parse(src)) == set()

    def test_scope_survives_assignment_order(self):
        """``agn=AGN`` before ``AGN = {...}`` still resolves.

        Both passes walk the whole tree, so a forward reference -- a module
        constant defined below its use inside a function -- is not a blind spot.
        """
        src = "def f():\n    return build(agn=AGN)\nAGN = {'log_lbol': Uniform(9.0, 12.0)}"
        assert "agn_log_lbol" in _resolved(src)


class TestAllowlistStaysForRealExceptions:
    """A ratchet, not a style rule.

    An ALLOWLIST entry asserts the *declaration* is wrong for one caller. Nine
    entries once asserted something else: that the tool resolved a name
    incorrectly. That is a bug report filed as a suppression, and it hid the
    reverse error for as long as it stood. If this fails, fix the resolver.
    """

    def test_no_entry_excuses_a_resolution_error(self):
        offenders = {
            key: reason
            for key, reason in ALLOWLIST.items()
            if "false positive" in reason.lower() or "resolves to" in reason.lower()
        }
        assert not offenders, (
            "ALLOWLIST entries describing a misresolved name:\n"
            + "\n".join(f"  {k}: {v}" for k, v in offenders.items())
            + "\n\nAn entry here says the declared range is wrong for one caller. "
            "A name resolved to the wrong parameter is a resolver bug -- scope it "
            "in _agn_scoped_dicts instead, which is what removed the nine entries "
            "this test replaces."
        )
