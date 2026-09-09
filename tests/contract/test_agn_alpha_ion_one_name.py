# SPDX-License-Identifier: BSD-3-Clause
"""The Feltre NLR ionizing power-law slope has ONE name: ``agn_nlr_alpha_pl`` (R50, #2214).

The re-review of R41 (the Feltre dust-to-metal axis rename, ``neb_xid`` ->
``agn_nlr_xi_d``) found the same axis carried a second, unrelated-looking
duplicate: the Feltre+2016 ionizing power-law slope :math:`\\alpha_{\\rm pl}`
was reachable under two names.

``agn_nlr_alpha_pl``
    Declared in ``components/agn/_params.py`` as ``Uniform(-2.0, -1.2,
    default=-1.7)`` and passed by ``blocks/nlr.py`` into
    ``nlr_cloudy.compute_nlr_sed_feltre`` (and the ``cue`` NLR block) as
    ``alpha_pl=`` -- the name the block actually reads.
``agn_alpha_ion``
    Declared separately, same prior, same default, described as the "AGN EUV
    power-law slope ... for Feltre NLR backend", partitioned to ``agn.nlr``,
    and read by nothing on any live path (only the dead ``feltre_nlr``
    precompute adapter's ``AXIS_PARAMS``, which nothing schedules).

Measured before this fix: ``nlr={'type': 'feltre', 'agn_alpha_ion': FREE}``
(and the short form ``alpha_ion``) parsed, freed the parameter, and built a
model where that dimension moved nothing -- a silently inert free parameter,
the same disease R41 fixed for the dust-to-metal axis one ruling earlier.

One quantity, one name. ``agn_alpha_ion`` is retired, and writing it (or its
short form ``alpha_ion``) anywhere raises a single loud message naming the
replacement and its placement.
"""

from __future__ import annotations

import warnings

import pytest

pytestmark = pytest.mark.contract

from tengri.parameters import DEFAULT, FREE, Fixed, parse_groups

#: The retired spelling and the short form the sub-block grammar would give it.
_RETIRED_SPELLINGS = ("agn_alpha_ion", "alpha_ion")

#: The seven parameters ``AGN_BLOCK_CONSUMES[("nlr", "feltre")]`` declares live
#: (blocks/_consumes.py) -- what an ``all_params: FREE`` wildcard under
#: ``nlr={'type': 'feltre'}`` must free, no more and no less.
_FELTRE_NLR_CONSUMES = frozenset(
    {
        "agn_nlr_cf",
        "agn_nlr_fwhm_kms",
        "agn_nlr_alpha_pl",
        "agn_nlr_logU",
        "agn_nlr_logn",
        "agn_nlr_logZ",
        "agn_nlr_xi_d",
    }
)


def _parse(**groups):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return parse_groups(
            sfh={"type": "delayed", "all_params": Fixed(DEFAULT)},
            redshift=Fixed(0.5),
            **groups,
        )


def _composable(nlr_type: str = "feltre", **extra):
    agn = {
        "type": "composable",
        "norm": "independent",
        "all_params": Fixed(DEFAULT),
        "disc": {"type": "multicolor", "all_params": Fixed(DEFAULT)},
        "nlr": {"type": nlr_type, "all_params": Fixed(DEFAULT)},
    }
    agn.update(extra)
    return agn


# ── One name in the registry and the partition ────────────────────────────


def test_the_retired_name_is_gone_from_the_registry():
    """Two live names for one grid axis is the defect; one is the fix."""
    from tengri.parameters.registry import registry

    reg = registry()
    assert "agn_nlr_alpha_pl" in reg, "the surviving name must stay registered"
    assert "agn_alpha_ion" not in reg, (
        "agn_alpha_ion is still registered. The Feltre ionizing power-law slope has "
        "one name, agn_nlr_alpha_pl, owned by the block that reads it."
    )


def test_the_partition_gives_the_axis_one_owner():
    from tengri.parameters.groups import _AGN_PARTITION

    assert _AGN_PARTITION["agn_nlr_alpha_pl"] == "agn.nlr"
    assert "agn_alpha_ion" not in _AGN_PARTITION, (
        "the partition still carries the retired name -- a dead entry, read by nothing"
    )


def test_the_feltre_precompute_reads_the_owned_axis_name():
    from tengri.components.nebular import feltre_precompute

    assert "agn_nlr_alpha_pl" in feltre_precompute.AXIS_PARAMS
    assert "agn_alpha_ion" not in feltre_precompute.AXIS_PARAMS


# ── The retired key is refused loudly, wherever it is written ─────────────


def _message_for(**groups) -> str:
    with pytest.raises(ValueError) as exc_info:
        _parse(**groups)
    message = str(exc_info.value)
    assert "agn_nlr_alpha_pl" in message, message
    return message


@pytest.mark.parametrize("key", _RETIRED_SPELLINGS)
def test_the_retired_key_at_the_agn_top_level_names_the_replacement(key):
    _message_for(agn=_composable(**{key: Fixed(-1.5)}))


@pytest.mark.parametrize("key", _RETIRED_SPELLINGS)
def test_the_retired_key_under_agn_nlr_names_the_replacement(key):
    agn = _composable()
    agn["nlr"] = {"type": "feltre", key: Fixed(-1.5), "all_params": Fixed(DEFAULT)}
    message = _message_for(agn=agn)
    assert "nlr" in message, message


def test_the_message_shows_the_placement_that_works():
    """A refusal that does not spell the working form is half an answer."""
    message = _message_for(agn=_composable(agn_alpha_ion=Fixed(-1.5)))
    assert "'nlr'" in message, message
    assert "feltre" in message, message


# ── The surviving name still works, at the placement the message names ────


@pytest.mark.parametrize("key", ["agn_nlr_alpha_pl", "nlr_alpha_pl"])
def test_the_surviving_name_is_accepted_under_agn_nlr(key):
    agn = _composable()
    agn["nlr"] = {"type": "feltre", key: Fixed(-1.5), "all_params": Fixed(DEFAULT)}
    spec = _parse(agn=agn)
    assert spec.get_fixed_values()["agn_nlr_alpha_pl"] == pytest.approx(-1.5, abs=0.0, rel=1e-12)


def test_nlr_feltre_wildcard_frees_exactly_the_seven_consumes_names():
    """``all_params: FREE`` under ``nlr={'type': 'feltre'}`` must free exactly
    the CONSUMES set -- not a dead ``agn_alpha_ion`` dimension alongside it."""
    agn = _composable()
    agn["nlr"] = {"type": "feltre", "all_params": FREE}
    spec = _parse(agn=agn)
    free_nlr = {n for n in spec.free_params if n.startswith("agn_nlr_")}
    assert free_nlr == _FELTRE_NLR_CONSUMES
    assert "agn_alpha_ion" not in spec.free_params
