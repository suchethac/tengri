# SPDX-License-Identifier: BSD-3-Clause
"""``dust={'law_neb': ...}`` threads end-to-end through the builder.

The nebular birth-cloud law (and its ``*_neb`` per-parameter overrides) must:

* land on the ``Parameters`` spec (``dust_law_neb`` / ``dust_law_overrides['neb']``),
* round-trip through ``model.spec.to_groups()``, and
* enter the kernel-cache ``compile_signature`` — otherwise two models that
  differ only in their nebular reddening would share a compiled kernel and the
  nebular attenuation would leak between them (the color-leak footgun that
  bit the per-component stellar slopes in #669).
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
        dust_emission=None,
        neb={"type": "none"},
        redshift=tengri.Fixed(0.05),
    )


def test_law_neb_lands_on_spec(synthetic_ssp_wide, synthetic_tophat_obs):
    m = _build(synthetic_ssp_wide, synthetic_tophat_obs, law_neb="calzetti")
    assert m.spec.dust_law_neb == "calzetti"
    # Default (unset) inherits the birth cloud -> None on the spec.
    base = _build(synthetic_ssp_wide, synthetic_tophat_obs)
    assert getattr(base.spec, "dust_law_neb", None) is None


def test_law_neb_round_trips_through_to_groups(synthetic_ssp_wide, synthetic_tophat_obs):
    # ``conroy2010`` rather than ``smc``: the nebular law has to READ the
    # ``*_neb`` override paired with it, and smc reads nothing beyond
    # wavelength, so ``slope_neb`` there is a value the curve discards (#2185).
    m = _build(synthetic_ssp_wide, synthetic_tophat_obs, law_neb="conroy2010", slope_neb=-1.3)
    groups = m.spec.to_groups()
    assert groups["dust_attenuation"]["law_neb"] == "conroy2010"
    assert groups["dust_attenuation"]["slope_neb"] == pytest.approx(-1.3)
    # Re-build from the round-tripped groups: the nebular law survives.
    m2 = tengri.SEDModel.build(synthetic_ssp_wide, observation=synthetic_tophat_obs, **groups)
    assert m2.spec.dust_law_neb == "conroy2010"
    assert m2.spec.dust_law_overrides.get("neb", {}).get("n_slope") == pytest.approx(-1.3)


def test_law_neb_changes_compile_signature(synthetic_ssp_wide, synthetic_tophat_obs):
    """Distinct nebular law -> distinct signature (no kernel-cache color-leak)."""
    base = _build(synthetic_ssp_wide, synthetic_tophat_obs).compile_signature()
    neb = _build(synthetic_ssp_wide, synthetic_tophat_obs, law_neb="calzetti").compile_signature()
    ovr = _build(synthetic_ssp_wide, synthetic_tophat_obs, slope_neb=-1.3).compile_signature()
    assert base != neb
    assert base != ovr
    assert neb != ovr
