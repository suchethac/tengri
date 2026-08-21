# SPDX-License-Identifier: BSD-3-Clause
"""Regression: AGN sub-block parameter acceptance checks partition owner, not consumes.

Issue #1998: A parameter written inside an AGN sub-block that CONSUMES it but
does not OWN it (per the partition table) was silently dropped with no error.

Example bug:
  agn_polar_ebv belongs to agn.atten (the owner) but is consumed by SKIRTOR torus.
  Writing it under agn={'torus': {'polar_ebv': Fixed(0.35)}} was silently ignored.
  It should raise with guidance: "'agn_polar_ebv' is a 'agn.atten' parameter,
  not a 'agn.torus' one. Nest it: agn={'atten': {...}}".

The consumes map is a physics-input census, not an acceptance predicate.
Parameters must be nested under their OWNER block, not a consuming block.

Test coverage:
1. (a) Torus-nested polar_ebv raises with guidance naming 'agn.atten'
2. (b) Atten-nested polar_ebv=0.35 still works and lands in get_fixed_values()
3. (c) All three trap params (agn_polar_ebv, agn_polar_beta, agn_cos_inc) raise under wrong block
4. (d) Owned params under their owner still work (e.g., torus-owned tau_skirtor under torus)
"""

from __future__ import annotations

import pytest

import tengri

pytestmark = pytest.mark.regression_bug


class TestAGNSubblockOwnerAcceptance:
    """Parameters must be nested under their owning block, not consuming blocks."""

    def test_polar_ebv_under_torus_raises_with_guidance(self):
        """Torus-nested polar_ebv should raise, naming the correct owner (agn.atten)."""
        agn_bad = {
            "type": "composable",
            "torus": {"type": "skirtor", "polar_ebv": tengri.Fixed(0.35)},
            "atten": {"type": "none"},
        }
        with pytest.raises(
            ValueError,
            match=(
                r"'polar_ebv' is a 'agn\.atten' parameter, "
                r"not a 'agn\.torus' one"
            ),
        ):
            tengri.parse_groups(redshift=tengri.Fixed(0.1), agn=agn_bad)

    def test_polar_ebv_under_atten_works(self):
        """Atten-nested polar_ebv should work and land in get_fixed_values()."""
        agn_good = {
            "type": "composable",
            "torus": {"type": "skirtor"},
            "atten": {"type": "polar_dust", "polar_ebv": tengri.Fixed(0.35)},
        }
        params = tengri.parse_groups(redshift=tengri.Fixed(0.1), agn=agn_good)
        assert params.get_fixed_values()["agn_polar_ebv"] == 0.35

    def test_polar_beta_under_torus_raises(self):
        """Torus-nested polar_beta should raise."""
        agn_bad = {
            "type": "composable",
            "torus": {"type": "skirtor", "polar_beta": tengri.Fixed(1.5)},
            "atten": {"type": "none"},
        }
        with pytest.raises(ValueError, match=r"'polar_beta' is a 'agn\.atten' parameter"):
            tengri.parse_groups(redshift=tengri.Fixed(0.1), agn=agn_bad)

    def test_cos_inc_under_torus_works(self):
        """Torus-nested cos_inc should work (it's shared, can go anywhere)."""
        agn_good = {
            "type": "composable",
            "torus": {"type": "skirtor", "cos_inc": tengri.Fixed(0.5)},
            "atten": {"type": "none"},
        }
        # cos_inc is shared (agn level), can be placed in any sub-block
        params = tengri.parse_groups(redshift=tengri.Fixed(0.1), agn=agn_good)
        assert params.get_fixed_values()["agn_cos_inc"] == 0.5

    def test_torus_owned_param_under_torus_works(self):
        """A torus-owned param under torus should work (e.g., tau_skirtor)."""
        agn_good = {
            "type": "composable",
            "torus": {"type": "skirtor", "tau_skirtor": tengri.Fixed(50.0)},
            "atten": {"type": "none"},
        }
        params = tengri.parse_groups(redshift=tengri.Fixed(0.1), agn=agn_good)
        assert params.get_fixed_values()["agn_tau_skirtor"] == 50.0
