# SPDX-License-Identifier: BSD-3-Clause
"""A two-screen attenuation model must name both members of a per-screen pair.

The law rules (#1989) made a lone ``law_bc`` an error: naming one screen of a
two-screen model leaves it half specified, and inheritance used to paper over it.
The same argument covers every other suffixed pair, and it did not: a lone
``tau_bc`` left ``tau_diff`` pinned at its declared 0.3 while the birth cloud was
fitted. The diffuse screen usually carries most of the total attenuation, so a fit
that frees only the birth cloud is rarely what anyone means, and reading it back
off the config required knowing that the absent key has a default at all.

``#1982`` does name the pinned parameter, but inside a list of six, which is how
this survived the first pass over the dust grammar.

Three spellings stay legal, and each says something different:

    'tau_bc': ..., 'tau_diff': ...        per-screen, stated
    'tau_bc': ..., 'all_params': FREE     free the partner too
    neither                               both take declared defaults

Measured when the rule was added: no in-repo call site named a lone pair member
without a wildcard, so this refuses only genuinely half-specified configs.

Each stem is exercised under a law that READS it (#2185). Completeness is only a
question once the key means something: ``slope_bc`` under ``calzetti`` is not
half a pair, it is a value that curve has no place to put, and the grammar now
refuses it by name. ``tau`` is the exception and keeps ``calzetti``, because the
two optical depths enter through the Charlot & Fall geometry rather than as
curve arguments, so every law consumes them.
"""

from __future__ import annotations

import pytest

from tengri import FREE, Uniform, parse_groups

pytestmark = pytest.mark.contract

#: Per-screen stem -> a law that reads it. ``conroy2010`` reads ``n_slope`` and
#: ``dust_Rv``; ``noll09`` reads ``dust_delta`` and ``dust_bump_strength``.
PAIR_STEM_LAWS = {
    "tau": "calzetti",
    "Rv": "conroy2010",
    "delta": "noll09",
    "slope": "conroy2010",
    "bump_strength": "noll09",
}
PAIR_STEMS = tuple(PAIR_STEM_LAWS)

_BASE = {"type": "two_component", "law": "calzetti"}


def _atten(**extra):
    return dict(_BASE, **extra)


def _atten_for(stem, **extra):
    """Attenuation dict on a law that reads ``stem``."""
    return dict(_BASE, law=PAIR_STEM_LAWS[stem], **extra)


def _build(atten):
    return parse_groups(sfh={"type": "dpl"}, redshift=0.1, dust_attenuation=atten)


@pytest.mark.parametrize("stem", PAIR_STEMS)
@pytest.mark.parametrize("named,missing", [("bc", "diff"), ("diff", "bc")])
def test_lone_pair_member_raises_naming_the_partner(stem, named, missing):
    """Either half alone raises, and the message names the key to add."""
    with pytest.raises(ValueError, match=rf"{stem}_{missing}"):
        _build(_atten_for(stem, **{f"{stem}_{named}": 0.5}))


@pytest.mark.parametrize("stem", PAIR_STEMS)
def test_both_members_together_are_accepted(stem):
    spec = _build(_atten_for(stem, **{f"{stem}_bc": 0.5, f"{stem}_diff": 0.4}))
    assert spec is not None


@pytest.mark.parametrize("stem", PAIR_STEMS)
def test_a_wildcard_is_an_accepted_way_to_free_the_partner(stem):
    """The wildcard states 'free the rest', which covers the partner."""
    spec = _build(_atten_for(stem, **{f"{stem}_bc": 0.5, "all_params": FREE}))
    assert spec is not None


@pytest.mark.parametrize("stem", ("Rv", "delta", "slope", "bump_strength"))
def test_the_partner_is_not_required_when_its_law_cannot_read_it(stem):
    """A lone half is complete when the other screen's law has no such knob.

    ``law_bc='conroy2010', law_diff='calzetti'`` reads ``slope_bc`` and has no
    ``slope_diff`` to give. Requiring the partner and then refusing it as unread
    (#2185) would leave no spelling that parses.
    """
    spec = _build(
        {
            "type": "two_component",
            "law_bc": PAIR_STEM_LAWS[stem],
            "law_diff": "smc",
            f"{stem}_bc": 0.5,
        }
    )
    assert spec is not None


def test_naming_neither_member_still_takes_declared_defaults():
    """The rule fires on a HALF-specified pair, not on an unspecified one."""
    spec = _build(_atten())
    assert spec.get_fixed_values()["dust_tau_diff"] == 0.3


def test_single_component_has_one_screen_so_nothing_is_paired():
    """tau_v is the single screen's depth; there is no partner to require."""
    spec = parse_groups(
        sfh={"type": "dpl"},
        redshift=0.1,
        dust_attenuation={"type": "single_component", "law": "calzetti", "tau_v": Uniform(0, 2)},
    )
    assert "dust_tau_v" in spec.free_params or "dust_tau_bc" in spec.free_params


def test_the_neb_channel_is_not_paired():
    """`_neb` is a third optional channel, not half of a pair."""
    spec = _build(_atten(law_neb="smc"))
    assert spec is not None


def test_the_error_shows_all_three_accepted_spellings():
    """A refusal has to say what to write instead, not only what is wrong."""
    with pytest.raises(ValueError) as exc:
        _build(_atten(tau_bc=Uniform(0, 2)))
    msg = str(exc.value)
    assert "tau_diff" in msg
    assert "all_params" in msg
    assert "neither" in msg


def test_the_fitted_case_that_motivated_the_rule():
    """The exact config that silently pinned the dominant screen."""
    with pytest.raises(ValueError, match="silently keeps its declared default"):
        _build(_atten(tau_bc=Uniform(0, 2)))
