# SPDX-License-Identifier: BSD-3-Clause
"""Contract tests: ``other_params`` is an exact synonym of ``all_params``.

``all_params`` is the canonical wildcard key in the nested-dict grammar
(``'*'`` internally, retired as user input). ``other_params`` is a second,
exact-synonym spelling -- see ``tengri.parameters.sentinels.WILDCARD_ALIAS_OTHER``
for the style rationale (``all_params`` reads best as a group's only
directive; ``other_params`` reads best written last, after explicit
per-parameter entries, as "the others"). Both spellings normalize to the
same internal ``'*'`` key at the ``parse_groups`` boundary, so every group
that accepts ``all_params`` -- top-level or nested sub-block -- must accept
``other_params`` identically, and a dict carrying both must raise.

Mirrors the style of ``test_all_params_wildcard.py``. Covered here:

- Every top-level group that takes a wildcard (a).
- Nested sub-blocks: sfh field, igm.dla, radio.sf/.agn, agn.torus (b).
- Both keys in one dict raises, at the top level and nested (c).
- The builder-factory ``other_params=`` kwarg, across all four factory
  surfaces (``_factory.make_factory``, ``sfh``, ``dust``, ``agn.composable``) (d).
- A user-written ``'*'`` still raises, synonym present or not (e).
"""

from __future__ import annotations

import functools
import warnings

import pytest

from tengri import FIXED, FREE, Fixed, builders
from tengri.parameters.groups import parse_groups

pytestmark = pytest.mark.contract


def _parse(**groups):
    """``parse_groups``, ignoring warnings orthogonal to the synonym property.

    A partial-free wildcard (``WildcardPartialFreeWarning``) or an unstated
    disposition (``DefaultFixedParametersWarning``) is expected for several of
    the minimal recipes below; the synonym property holds regardless of which
    parameters end up free, so those warnings are noise here, not signal.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return parse_groups(**groups)


def _equivalent(all_kwargs: dict, other_kwargs: dict):
    """Assert the ``all_params=`` and ``other_params=`` builds are identical.

    Compares free/fixed parameter names AND the full round-trip
    (``to_groups()``), so a divergence in *which* parameters end up free, or
    in the resolved priors/values, fails this just as loudly as a raw
    ``KeyError`` on the unrecognized spelling would. Returns the
    ``all_params`` build for callers that want to assert something further
    (e.g. that the wildcard actually freed something).
    """
    all_result = _parse(**all_kwargs)
    other_result = _parse(**other_kwargs)
    assert other_result.free_params == all_result.free_params
    assert other_result.fixed_params == all_result.fixed_params
    assert other_result.to_groups() == all_result.to_groups()
    return all_result


# ── (a) Every top-level group that takes a wildcard ────────────────────────

# One minimal, real recipe per group, chosen so 'all_params': FREE frees at
# least one declared parameter (grounded in the per-group free/fixed counts
# in test_wildcard_frees_something.py and the referenced _params.py files):
# sfh 'dpl', met 'delta' (met_logzsol), dust_attenuation 'two_component',
# dust_emission 'astrodust' (the one backend that frees its own declared
# param silently, per test_dust_emission_wildcard.py), neb 'cue' (neb_logU),
# shock/xray/radio/agn at their defaults (shock_*, xray_gamma_agn, the bare
# 'radio' params radio_q_ir/alpha_sf/T_e, agn_* shared params), igm 'inoue14'
# (igm_z_mid/dz/log_nhi, all three declare free_prior).
TOP_LEVEL_GROUPS = [
    ("sfh", {"type": "dpl"}),
    ("met", {"type": "delta"}),
    ("dust_attenuation", {"type": "two_component", "law": "calzetti"}),
    ("dust_emission", {"type": "astrodust"}),
    ("neb", {"type": "cue"}),
    ("shock", {}),
    ("igm", {"type": "inoue14"}),
    ("radio", {}),
    ("xray", {"type": "simple"}),
    ("agn", {"type": "composable"}),
]


def _top_level_kwargs(group: str, spec: dict, key: str, wildcard) -> dict:
    """Build full ``parse_groups`` kwargs with the wildcard written under ``key``."""
    target = {**spec, key: wildcard}
    kwargs: dict = {"redshift": Fixed(0.1)}
    if group == "sfh":
        kwargs["sfh"] = target
    else:
        kwargs["sfh"] = {"type": "dpl", "all_params": FIXED}
        kwargs[group] = target
    return kwargs


@pytest.mark.parametrize("group,spec", TOP_LEVEL_GROUPS, ids=[g for g, _ in TOP_LEVEL_GROUPS])
@pytest.mark.parametrize("wildcard", [FREE, FIXED], ids=["free", "fixed"])
def test_top_level_group_other_params_matches_all_params(group, spec, wildcard):
    """``other_params`` sets the identical wildcard policy as ``all_params``, every group."""
    all_kwargs = _top_level_kwargs(group, spec, "all_params", wildcard)
    other_kwargs = _top_level_kwargs(group, spec, "other_params", wildcard)
    result = _equivalent(all_kwargs, other_kwargs)
    if wildcard is FREE:
        assert result.free_params, (
            f"{group}: the FREE wildcard froze everything -- this recipe no longer "
            f"exercises a live parameter, pick a different minimal spec"
        )


# ── (b) Nested sub-blocks ───────────────────────────────────────────────────


def test_sfh_field_subblock_other_params_matches_all_params():
    """``other_params`` inside sfh's nested ``field`` sub-block is an exact synonym.

    ``sfh_field_psd_sigma`` / ``sfh_field_psd_tau_myr`` default to a free
    ``Uniform`` (not ``Fixed``), so the field wildcard fully frees both.
    """
    all_kwargs = {
        "redshift": Fixed(0.1),
        "sfh": {"type": "dpl", "all_params": FIXED, "field": {"all_params": FREE}},
    }
    other_kwargs = {
        "redshift": Fixed(0.1),
        "sfh": {"type": "dpl", "all_params": FIXED, "field": {"other_params": FREE}},
    }
    result = _equivalent(all_kwargs, other_kwargs)
    assert "sfh_field_psd_sigma" in result.free_params
    assert "sfh_field_psd_tau_myr" in result.free_params


def test_igm_dla_subblock_other_params_matches_all_params():
    """``other_params`` inside the nested ``igm={'dla': {...}}`` sub-block."""
    all_kwargs = {
        "redshift": Fixed(2.0),
        "sfh": {"type": "dpl", "all_params": FIXED},
        "igm": {"type": "inoue14", "all_params": FIXED, "dla": {"all_params": FREE}},
    }
    other_kwargs = {
        "redshift": Fixed(2.0),
        "sfh": {"type": "dpl", "all_params": FIXED},
        "igm": {"type": "inoue14", "all_params": FIXED, "dla": {"other_params": FREE}},
    }
    result = _equivalent(all_kwargs, other_kwargs)
    assert "dla_log_n_hi" in result.free_params


def test_radio_sf_subblock_other_params_matches_all_params():
    """``other_params`` inside the nested ``radio={'sf': {...}}`` sub-block."""
    all_kwargs = {
        "redshift": Fixed(0.1),
        "sfh": {"type": "dpl", "all_params": FIXED},
        "radio": {"sf": {"type": "delvecchio2021", "all_params": FREE}},
    }
    other_kwargs = {
        "redshift": Fixed(0.1),
        "sfh": {"type": "dpl", "all_params": FIXED},
        "radio": {"sf": {"type": "delvecchio2021", "other_params": FREE}},
    }
    result = _equivalent(all_kwargs, other_kwargs)
    assert "radio_delv_q0" in result.free_params


def test_radio_agn_subblock_other_params_matches_all_params():
    """``other_params`` inside the nested ``radio={'agn': {...}}`` sub-block."""
    all_kwargs = {
        "redshift": Fixed(0.1),
        "sfh": {"type": "dpl", "all_params": FIXED},
        "radio": {"agn": {"type": "dpl", "all_params": FREE}},
    }
    other_kwargs = {
        "redshift": Fixed(0.1),
        "sfh": {"type": "dpl", "all_params": FIXED},
        "radio": {"agn": {"type": "dpl", "other_params": FREE}},
    }
    result = _equivalent(all_kwargs, other_kwargs)
    assert "radio_log_nu_t" in result.free_params


def test_agn_torus_subblock_other_params_matches_all_params():
    """``other_params`` inside the nested ``agn={'torus': {...}}`` sub-block."""
    all_kwargs = {
        "redshift": Fixed(0.1),
        "sfh": {"type": "dpl", "all_params": FIXED},
        "agn": {
            "type": "composable",
            "all_params": FIXED,
            "torus": {"type": "skirtor", "all_params": FREE},
        },
    }
    other_kwargs = {
        "redshift": Fixed(0.1),
        "sfh": {"type": "dpl", "all_params": FIXED},
        "agn": {
            "type": "composable",
            "all_params": FIXED,
            "torus": {"type": "skirtor", "other_params": FREE},
        },
    }
    result = _equivalent(all_kwargs, other_kwargs)
    assert any(p.startswith("agn_") for p in result.free_params)


# ── (c) Both keys in one dict raises ────────────────────────────────────────


def test_both_wildcard_keys_in_one_dict_raises_top_level():
    """``all_params`` and ``other_params`` together in a top-level group dict raise."""
    with pytest.raises(ValueError, match=r"synonyms"):
        parse_groups(
            redshift=Fixed(0.1),
            sfh={"type": "dpl", "all_params": FREE, "other_params": FIXED},
        )


def test_both_wildcard_keys_in_one_dict_raises_nested_subblock():
    """``all_params`` and ``other_params`` together in a nested sub-block dict raise."""
    with pytest.raises(ValueError, match=r"synonyms"):
        parse_groups(
            redshift=Fixed(2.0),
            sfh={"type": "dpl"},
            igm={"type": "inoue14", "dla": {"all_params": FREE, "other_params": FIXED}},
        )


# ── (d) Builder kwarg: other_params= across all four factory surfaces ──────

# One factory per wrapper implementation (mirrors
# tests/contract/builders/test_wildcard_rename.py): generic (_factory.make_factory,
# shared by igm/xray/radio/neb/dust.emission/agn sub-blocks), sfh's own inline
# wrapper, dust's own inline wrapper, and agn.composable's own inline wrapper.
_BUILDER_FACTORIES = {
    "generic (neb.cue)": builders.neb.cue,
    "sfh.dpl": builders.sfh.dpl,
    "dust.two_component": functools.partial(builders.dust.two_component, law="calzetti"),
    "agn.composable": builders.agn.composable,
}


@pytest.mark.parametrize(("label", "factory"), _BUILDER_FACTORIES.items())
def test_builder_other_params_matches_all_params(label, factory):
    """``builders.*(other_params=FREE)`` produces the identical dict as ``all_params=FREE``."""
    all_dict = factory(all_params=FREE)
    other_dict = factory(other_params=FREE)
    assert other_dict == all_dict, label
    assert other_dict["all_params"] is FREE, label


def test_sfh_dpl_other_params_matches_all_params_fixed_too():
    """The explicit case the plan calls out: ``builders.sfh.dpl``, both wildcard values."""
    assert builders.sfh.dpl(other_params=FREE) == builders.sfh.dpl(all_params=FREE)
    assert builders.sfh.dpl(other_params=FIXED) == builders.sfh.dpl(all_params=FIXED)


@pytest.mark.parametrize(("label", "factory"), _BUILDER_FACTORIES.items())
def test_builder_both_wildcard_kwargs_raises(label, factory):
    """Passing both ``all_params=`` and ``other_params=`` to a builder raises."""
    with pytest.raises(ValueError, match=r"synonyms"):
        factory(all_params=FREE, other_params=FIXED)


# ── (e) A user-written '*' still raises ─────────────────────────────────────


def test_star_still_raises():
    """The internal ``'*'`` key stays retired; unaffected by the new synonym."""
    with pytest.raises(ValueError, match=r"retired"):
        parse_groups(redshift=Fixed(0.1), sfh={"type": "dpl", "*": FREE})


def test_star_still_raises_even_with_other_params_present():
    """``'*'`` is retired outright -- even alongside the ``other_params`` synonym."""
    with pytest.raises(ValueError, match=r"retired"):
        parse_groups(
            redshift=Fixed(0.1),
            sfh={"type": "dpl", "*": FREE, "other_params": FIXED},
        )
