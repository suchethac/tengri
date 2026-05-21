# SPDX-License-Identifier: BSD-3-Clause

import pytest

pytestmark = pytest.mark.contract

"""Tests for provenance tagging in Parameters.summary() output.

When a Parameters is built via parse_groups (or Parameters.from_groups),
each parameter carries a provenance tag (user override, wildcard, registry
default). summary() renders those tags so the user can see why each value
ended up where it did.

These tests verify provenance attribution and the rendered tags. They do
NOT compare against frozen golden strings — fragile under formatting
changes — but instead assert that specific tags appear next to specific
params.
"""

import pytest

from tengri import FIXED, FREE, Fixed, Parameters, Uniform


@pytest.fixture
def grouped_spec():
    """Reference spec exercising every provenance tag."""
    return Parameters.from_groups(
        sfh={
            "type": "dpl",
            "*": FREE,
            "beta": Uniform(1, 3),  # user_prior
            "alpha": Fixed(2.0),  # user_fixed
        },
        dust={
            "type": "two_component",
            "law_bc": "calzetti",
            "*": FIXED,
            "tau_bc": 0.5,  # user_fixed (bare value)
        },
        redshift=Fixed(0.05),  # user_fixed (top-level)
    )


class TestProvenanceAttribution:
    """Each parameter gets the correct provenance tag."""

    def test_provenance_attached(self, grouped_spec):
        """Specs built via from_groups carry a _group_provenance dict."""
        assert hasattr(grouped_spec, "_group_provenance")
        assert isinstance(grouped_spec._group_provenance, dict)

    def test_user_prior_tagged_correctly(self, grouped_spec):
        """A param with a user-supplied free prior gets 'user_prior'."""
        assert grouped_spec._group_provenance["sfh_dpl_beta"] == "user_prior"

    def test_user_fixed_tagged_correctly(self, grouped_spec):
        """A param the user pinned (Fixed or bare value) gets 'user_fixed'."""
        assert grouped_spec._group_provenance["sfh_dpl_alpha"] == "user_fixed"
        assert grouped_spec._group_provenance["dust_tau_bc"] == "user_fixed"
        assert grouped_spec._group_provenance["redshift"] == "user_fixed"

    def test_wildcard_free_tagged_correctly(self, grouped_spec):
        """Wildcard '*': FREE expansions get 'wildcard_free'."""
        # sfh has '*': FREE; sfh_dpl_log_peak_sfr/tau_gyr weren't overridden
        assert grouped_spec._group_provenance["sfh_dpl_log_peak_sfr"] == "wildcard_free"
        assert grouped_spec._group_provenance["sfh_dpl_tau_gyr"] == "wildcard_free"

    def test_wildcard_fixed_tagged_correctly(self, grouped_spec):
        """Wildcard '*': FIXED expansions get 'wildcard_fixed'."""
        # dust has '*': FIXED; dust_tau_diff wasn't overridden
        assert grouped_spec._group_provenance["dust_tau_diff"] == "wildcard_fixed"
        assert grouped_spec._group_provenance["dust_slope"] == "wildcard_fixed"

    def test_flat_construction_has_no_provenance(self):
        """Specs built via the flat-kwarg form have no _group_provenance attribute."""
        spec = Parameters(mean_sfh_type="dpl", redshift=Fixed(0.1))
        assert not hasattr(spec, "_group_provenance")


class TestSummaryRendering:
    """summary_str() renders provenance tags when present, omits them otherwise."""

    def test_grouped_summary_includes_source_column(self, grouped_spec):
        """When provenance is set, the summary has a 'Source' column."""
        out = grouped_spec.summary_str()
        assert "Source" in out

    def test_grouped_summary_includes_user_tag(self, grouped_spec):
        """User-set params appear with [user] or [user FREE] tag."""
        out = grouped_spec.summary_str()
        # sfh_dpl_beta is user-set with a custom prior
        lines = [ln for ln in out.splitlines() if "sfh_dpl_beta" in ln]
        assert len(lines) == 1
        assert "[user]" in lines[0]

    def test_grouped_summary_includes_wildcard_free_tag(self, grouped_spec):
        """Wildcard-FREE params show [* FREE]."""
        out = grouped_spec.summary_str()
        lines = [ln for ln in out.splitlines() if "sfh_dpl_log_peak_sfr" in ln]
        assert len(lines) == 1
        assert "[* FREE]" in lines[0]

    def test_grouped_summary_includes_wildcard_fixed_tag(self, grouped_spec):
        """Wildcard-FIXED params show [* FIXED]."""
        out = grouped_spec.summary_str()
        lines = [ln for ln in out.splitlines() if "dust_tau_diff" in ln]
        assert len(lines) == 1
        assert "[* FIXED]" in lines[0]

    def test_flat_summary_omits_source_column(self):
        """Flat-kwarg specs render the existing summary without a Source column."""
        spec = Parameters(
            mean_sfh_type="dpl",
            sfh_dpl_log_peak_sfr=Uniform(-1, 2),
            sfh_dpl_alpha=Uniform(0.5, 3),
            redshift=Fixed(0.1),
        )
        out = spec.summary_str()
        assert "Source" not in out
        # Tags should not appear
        assert "[user]" not in out
        assert "[* FREE]" not in out
