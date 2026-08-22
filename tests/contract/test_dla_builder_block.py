# SPDX-License-Identifier: BSD-3-Clause
"""DLA absorber wired as a nested-dict builder block — closes #507.

The previous API only accepted ``igm={'dla': True}`` and forced users to
set ``dla_log_n_hi`` etc. via top-level overrides. This contract test
pins the new nested form ``igm={'dla': {'log_n_hi': ..., 'all_params': ...}}``
while keeping the boolean form working for back-compat.
"""

import pytest

from tengri import FIXED, FREE, parse_groups
from tengri.parameters.priors import Fixed, Uniform

pytestmark = pytest.mark.contract


class TestDLABuilderBlock:
    def test_dict_form_activates_and_routes_overrides(self):
        spec = parse_groups(
            sfh={"type": "dpl", "all_params": FIXED},
            igm={
                "type": "inoue14",
                # FIXED, not FREE: both DLA params carry Fixed registry
                # defaults, so FREE frees nothing and is now refused. This test
                # asserts activation and per-param override routing.
                "dla": {
                    "log_n_hi": Uniform(19, 22),
                    "b_turb": Fixed(10.0),
                    "all_params": FIXED,
                },
            },
            redshift=Fixed(2.0),
        )
        assert spec.dla is True
        # Per-param prior override took effect.
        assert spec._distributions["dla_log_n_hi"] == Uniform(19, 22)
        assert spec._distributions["dla_b_turb"] == Fixed(10.0)

    def test_boolean_form_still_works(self):
        spec = parse_groups(
            sfh={"type": "dpl", "all_params": FIXED},
            igm={"type": "inoue14", "dla": True},
            redshift=Fixed(2.0),
        )
        assert spec.dla is True

    def test_no_dla_block_means_no_absorber(self):
        spec = parse_groups(
            sfh={"type": "dpl", "all_params": FIXED},
            igm={"type": "inoue14"},
            redshift=Fixed(2.0),
        )
        assert spec.dla is False

    def test_unknown_dla_key_raises(self):
        with pytest.raises(ValueError, match=r"dla|igm"):
            parse_groups(
                sfh={"type": "dpl", "all_params": FIXED},
                igm={
                    "type": "inoue14",
                    "dla": {"not_a_dla_param": Uniform(0, 1), "all_params": FREE},
                },
                redshift=Fixed(2.0),
            )
