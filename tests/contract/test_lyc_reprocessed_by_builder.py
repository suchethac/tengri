# SPDX-License-Identifier: BSD-3-Clause
"""``dust_attenuation={'lyc_reprocessed_by': 'all'}`` threads end-to-end through the builder.

The stellar-LyC absorption mode (young-only default vs absorb-all) must land on
the ``Parameters`` spec, round-trip through ``to_groups``, and enter the kernel
``compile_signature`` — it changes the baked below-912 chain output, so two
models differing only here must not share a compiled kernel (color-leak guard).
"""

from __future__ import annotations

import pytest

import tengri

pytestmark = [pytest.mark.contract]


def _build(ssp, obs, **dust_extra):
    return tengri.SEDModel.build(
        ssp,
        observation=obs,
        sfh={"type": "delayed", "all_params": tengri.Fixed(tengri.DEFAULT)},
        dust_attenuation={
            "law": "power_law",
            "type": "two_component",
            "all_params": tengri.Fixed(tengri.DEFAULT),
            **dust_extra,
        },
        dust_emission={"type": "none"},
        neb={"type": "none"},
        redshift=tengri.Fixed(0.05),
    )


def test_lyc_reprocessed_by_lands_on_spec(synthetic_ssp_wide, synthetic_tophat_obs):
    assert _build(synthetic_ssp_wide, synthetic_tophat_obs).spec.dust_lyc_reprocessed_by == "young"
    m = _build(synthetic_ssp_wide, synthetic_tophat_obs, lyc_reprocessed_by="all")
    assert m.spec.dust_lyc_reprocessed_by == "all"


def test_lyc_reprocessed_by_round_trips(synthetic_ssp_wide, synthetic_tophat_obs):
    m = _build(synthetic_ssp_wide, synthetic_tophat_obs, lyc_reprocessed_by="all")
    groups = m.spec.to_groups()
    assert groups["dust_attenuation"]["lyc_reprocessed_by"] == "all"
    m2 = tengri.SEDModel.build(synthetic_ssp_wide, observation=synthetic_tophat_obs, **groups)
    assert m2.spec.dust_lyc_reprocessed_by == "all"
    # Default doesn't emit the key.
    base = _build(synthetic_ssp_wide, synthetic_tophat_obs)
    assert "lyc_reprocessed_by" not in base.spec.to_groups().get("dust_attenuation", {})


def test_lyc_reprocessed_by_changes_compile_signature(synthetic_ssp_wide, synthetic_tophat_obs):
    base = _build(synthetic_ssp_wide, synthetic_tophat_obs).compile_signature()
    allabs = _build(
        synthetic_ssp_wide, synthetic_tophat_obs, lyc_reprocessed_by="all"
    ).compile_signature()
    assert base != allabs
