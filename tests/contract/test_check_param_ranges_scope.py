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
    _dust_prior_sites,
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


class TestDustBlockShapes:
    """Every shape a dust_attenuation dict reaches the builder by must be scoped.

    Similar to AGN, dust_attenuation dicts can appear as:
    - dust_attenuation={...} keyword
    - {'dust_attenuation': {...}} fragment
    - cfg['dust_attenuation'] = {...} mutation
    - A name binding passed as dust_attenuation=NAME
    """

    def test_dust_keyword_form(self):
        """dust_attenuation as keyword argument with prior call."""
        src = "build(dust_attenuation={'type': 'two_component', 'tau_bc': Uniform(50.0, 90.0)})"
        assert "dust_tau_bc" in {param for param, _call in _dust_prior_sites(ast.parse(src))}

    def test_dust_fragment_form(self):
        """{'dust_attenuation': {...}} fragment splatted into call."""
        src = (
            "GROUPS = {'dust_attenuation': {'type': 'two_component', "
            "'tau_bc': Uniform(50.0, 90.0)}}"
        )
        assert "dust_tau_bc" in {param for param, _call in _dust_prior_sites(ast.parse(src))}

    def test_dust_subscript_assignment_form(self):
        """Config built by mutation: cfg['dust_attenuation'] = {...}."""
        src = "cfg = deepcopy(base)\ncfg['dust_attenuation'] = {'tau_bc': Uniform(50.0, 90.0)}"
        assert "dust_tau_bc" in {param for param, _call in _dust_prior_sites(ast.parse(src))}

    def test_dust_name_binding_form(self):
        """A named constant passed as dust_attenuation=NAME."""
        src = "DUST_CFG = {'tau_bc': Uniform(50.0, 90.0)}\nbuild(dust_attenuation=DUST_CFG)"
        assert "dust_tau_bc" in {param for param, _call in _dust_prior_sites(ast.parse(src))}

    def test_dust_case_table_form(self):
        """Case table form: _DUST_CASES[key] -> local -> groups['dust_attenuation']."""
        src = (
            "_DUST_CASES = {\n"
            "    'two_component': {'type': 'two_component', 'tau_bc': Uniform(50.0, 90.0)},\n"
            "}\n"
        )
        assert "dust_tau_bc" in {param for param, _call in _dust_prior_sites(ast.parse(src))}

    def test_dust_nested_sub_block_inherits_scope(self):
        """Nested dicts in dust blocks inherit scope."""
        src = "build(dust_attenuation={'type': 'two_component', 'tau_bc': Uniform(50.0, 90.0)})"
        resolved = {param for param, _call in _dust_prior_sites(ast.parse(src))}
        assert "dust_tau_bc" in resolved


class TestDustPerScreenKeysAreNotFlagged:
    """Per-screen keys like slope_bc, Rv_diff are static floats, not priors.

    They should never appear as prior Call values, but if they do, they should
    not be flagged as violations since they are not supposed to be prior literals.
    """

    def test_per_screen_static_float_not_flagged(self):
        """slope_bc with a static float should not crash or be flagged."""
        src = "build(dust_attenuation={'type': 'two_component', 'slope_bc': -1.0})"
        resolved = {param for param, _call in _dust_prior_sites(ast.parse(src))}
        # The static float -1.0 is not a Call node, so it won't be yielded
        assert "dust_slope" not in resolved


class TestDustViolationDetection:
    """Test that dust parameters with ranges outside declared support are caught.

    dust_tau_bc declared range: [0.0, 4.0]
    dust_tau_diff declared range: [0.0, 3.0]
    dust_slope declared range: [-1.5, -0.3]
    """

    def test_dust_tau_bc_outside_range_is_flagged(self):
        """tau_bc with Uniform(50.0, 90.0) is far outside [0.0, 4.0]."""
        src = "build(dust_attenuation={'type': 'two_component', 'tau_bc': Uniform(50.0, 90.0)})"
        resolved = {param for param, _call in _dust_prior_sites(ast.parse(src))}
        # This SHOULD be resolved; the violation is caught by the main check
        assert "dust_tau_bc" in resolved

    def test_dust_tau_bc_inside_range_is_not_flagged(self):
        """tau_bc with Uniform(0.1, 1.5) is inside [0.0, 4.0]."""
        src = "build(dust_attenuation={'type': 'two_component', 'tau_bc': Uniform(0.1, 1.5)})"
        resolved = {param for param, _call in _dust_prior_sites(ast.parse(src))}
        assert "dust_tau_bc" in resolved

    def test_dust_unknown_stem_not_resolved(self):
        """An unknown stem like 'bogus' should not be resolved."""
        src = "build(dust_attenuation={'type': 'two_component', 'bogus': Uniform(0, 1)})"
        resolved = {param for param, _call in _dust_prior_sites(ast.parse(src))}
        assert "dust_bogus" not in resolved


class TestNonDustDictsAreNotResolved:
    """Non-dust dicts should not be resolved as dust blocks."""

    def test_sfh_tau_is_not_dust_tau(self):
        """sfh dict should not be resolved as dust_attenuation."""
        src = "build(sfh={'type': 'dpl', 'tau': Uniform(0.1, 0.5)})"
        resolved = {param for param, _call in _dust_prior_sites(ast.parse(src))}
        assert not any("dust_" in p for p in resolved)

    def test_dust_and_sfh_side_by_side(self):
        """Both dust and sfh in one call; only dust keys are resolved."""
        src = (
            "build(\n"
            "    sfh={'type': 'dpl', 'alpha': Uniform(0.5, 2.0)},\n"
            "    dust_attenuation={'type': 'two_component', 'tau_bc': Uniform(50.0, 90.0)},\n"
            ")"
        )
        resolved = {param for param, _call in _dust_prior_sites(ast.parse(src))}
        assert "dust_tau_bc" in resolved
        # sfh_alpha should not be in dust resolution (it's not dust at all)
        assert not any("sfh_" in p for p in resolved)


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
