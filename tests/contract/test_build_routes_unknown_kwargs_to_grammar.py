# SPDX-License-Identifier: BSD-3-Clause
"""Contract: an unrecognized ``build()`` keyword is judged by the grammar, not ``__init__``.

``SEDModel.build`` names each parameter group in its signature and collects
everything else in ``**model_kwargs``, which it splats into ``__init__``. So a
keyword the grammar *would* have diagnosed never reaches the grammar: it reaches
a constructor that cannot accept it and raises

    TypeError: SEDModel.__init__() got an unexpected keyword argument 'stellar'

naming no replacement. Two error channels are lost that way, both of which exist
and are correct:

* the removed-group translations in ``_translate_structural`` — ``stellar=``
  raises a ``ValueError`` spelling out the ``met=`` migration in full;
* the ``difflib`` suggestion on an unknown group key — ``dsut=`` should offer
  ``dust``.

This is the general form of the bug PR #518 fixed for one case. #518's diagnosis
was exactly right — "``build()`` forwarded unknown keywords straight to
``__init__``" — but the remedy was a four-key allowlist (``_TOP_LEVEL_SETTINGS``,
see ``test_build_n_grid_forwarding.py``) rather than a rule, so every keyword
outside those four kept the old behavior. That is how ``stellar=`` came to die
this way after #1720 removed it, in five reproduction notebooks at once
(#1776–#1781), and why ``CLAUDE.md``'s "Passing ``stellar=`` raises with the
translation" documented something no caller of ``build()`` could observe.

The rule: ``model_kwargs`` that is not an ``__init__`` parameter is grammar
input. ``__init__`` keeps exactly the keywords it declares.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.contract

from tengri.parameters import DEFAULT, Fixed


def _build(ssp, obs, **kw):
    from tengri import SEDModel

    return SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        sfh={"type": "const", "all_params": Fixed(DEFAULT)},
        redshift=Fixed(0.1),
        **kw,
    )


def test_removed_stellar_group_raises_with_its_translation(synthetic_ssp, simple_observation):
    """``stellar=`` must reach the removed-group translation, not ``__init__``.

    The measured pre-fix behavior was ``TypeError: SEDModel.__init__() got an
    unexpected keyword argument 'stellar'`` — after, in the prospector notebook,
    185 s of FSPS work had already run.
    """
    with pytest.raises(ValueError, match=r"the 'stellar' group is gone"):
        _build(synthetic_ssp, simple_observation, stellar={"met_mode": "table"})


def test_removed_stellar_group_names_its_replacement(synthetic_ssp, simple_observation):
    """The point of routing it: the message has to carry the migration."""
    with pytest.raises(ValueError) as exc:
        _build(synthetic_ssp, simple_observation, stellar={"logzsol": Fixed(-0.2)})
    assert "met" in str(exc.value), "the translation must name the replacement group"


def test_unknown_group_key_is_offered_a_suggestion(synthetic_ssp, simple_observation):
    """A typo'd group gets difflib's suggestion instead of a bare TypeError.

    Not a ``stellar``-specific concern: the same lost channel makes *every*
    misspelled group name unrecoverable at the primary entry point.
    """
    with pytest.raises(ValueError, match=r"Unknown group key 'dsut'"):
        _build(synthetic_ssp, simple_observation, dsut={"type": "calzetti00"})

    with pytest.raises(ValueError, match=r"Did you mean.*dust"):
        _build(synthetic_ssp, simple_observation, dsut={"type": "calzetti00"})


def test_no_unknown_keyword_reaches_init_as_typeerror(synthetic_ssp, simple_observation):
    """The failure mode itself: ``__init__`` must never be the one to complain."""
    for bad in ("stellar", "dsut", "not_a_group_at_all"):
        with pytest.raises(Exception) as exc:
            _build(synthetic_ssp, simple_observation, **{bad: {"type": "x"}})
        assert not (
            isinstance(exc.value, TypeError) and "unexpected keyword argument" in str(exc.value)
        ), f"build({bad}=...) leaked to __init__: {exc.value}"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"precompute": False},
        {"csp_integration": "trapz"},
        {"wave_chunk_size": None},
        {"agn_config": None},
        {"compile": None},
        {"precompute": False, "csp_integration": "trapz"},
    ],
)
def test_real_init_keywords_are_not_misrouted(synthetic_ssp, simple_observation, kwargs):
    """The regression this fix could cause: genuine ``__init__`` kwargs must still land there.

    Routing leftovers to the grammar is only safe if the ``__init__`` census is
    right. If it drifts, these become "Unknown group key" — a loud failure, but
    on the working path rather than the broken one.
    """
    model = _build(synthetic_ssp, simple_observation, **kwargs)
    assert model is not None
