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


def _value(v):
    """Unwrap a to_groups() per-screen entry to its plain number.

    #2428: an explicitly-set per-screen shape parameter round-trips through
    ``to_groups()`` as the ``Distribution`` the spec actually holds (e.g.
    ``Fixed(-1.3)`` for a plain-number spelling), not the bare float --
    matching ``test_dust_per_component_law.py``'s own ``_value()`` helper for
    the same contract.
    """
    return float(v.bounds[0]) if hasattr(v, "bounds") else float(v)


def test_law_neb_round_trips_through_to_groups(synthetic_ssp_wide, synthetic_tophat_obs):
    # ``conroy2010`` rather than ``smc``: the nebular law has to READ the
    # ``*_neb`` override paired with it, and smc reads nothing beyond
    # wavelength, so ``slope_neb`` there is a value the curve discards (#2185).
    m = _build(synthetic_ssp_wide, synthetic_tophat_obs, law_neb="conroy2010", slope_neb=-1.3)
    groups = m.spec.to_groups()
    assert groups["dust_attenuation"]["law_neb"] == "conroy2010"
    assert _value(groups["dust_attenuation"]["slope_neb"]) == pytest.approx(-1.3)
    # Re-build from the round-tripped groups: the nebular law survives.
    m2 = tengri.SEDModel.build(synthetic_ssp_wide, observation=synthetic_tophat_obs, **groups)
    assert m2.spec.dust_law_neb == "conroy2010"
    # A nested-dict-grammar-built spec's per-screen override lands on the
    # declared parameter (``dust_slope_neb``), not ``dust_law_overrides`` --
    # that static dict is populated only from the flat surface's own
    # plain-number branch (#2428; see ``_emit_declared_structural`` in
    # ``parameters/groups.py``).
    assert m2.spec.get_fixed_values().get("dust_slope_neb") == pytest.approx(-1.3)


def test_law_neb_changes_compile_signature(synthetic_ssp_wide, synthetic_tophat_obs):
    """Distinct nebular law -> distinct signature (no kernel-cache color-leak)."""
    base = _build(synthetic_ssp_wide, synthetic_tophat_obs).compile_signature()
    neb = _build(synthetic_ssp_wide, synthetic_tophat_obs, law_neb="calzetti").compile_signature()
    ovr = _build(synthetic_ssp_wide, synthetic_tophat_obs, slope_neb=-1.3).compile_signature()
    assert base != neb
    assert base != ovr
    assert neb != ovr
