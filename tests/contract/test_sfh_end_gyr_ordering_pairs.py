# SPDX-License-Identifier: BSD-3-Clause
"""Ordering-constraint codification for dpl_lookback/trunc_exp (#2247 phase 1).

``Parameters._ORDERED_PAIRS`` (``parameters/parameters.py``,
``_validate_orderings``) held exactly one codified pair --
``sfh_const_start_gyr``/``sfh_const_end_gyr`` -- while the identical
"star formation cannot stop before it starts" constraint on the
``dpl_lookback`` (``sfh_dpl_lookback_age_gyr``/``sfh_dpl_lookback_end_gyr``)
and ``trunc_exp`` (``sfh_trunc_exp_age_gyr``/``sfh_trunc_exp_end_gyr``) SFH
types was refused only by ledger prose in
``tools/check_param_free_priors.py`` (ground ``not-continuous``) -- no
validator enforced it, so an inverted pair (the younger truncation lookback
set larger than the older one) built without complaint.

This pins that both new pairs now raise on inversion, exactly like the
``const`` pair, and that the physically-ordered case still builds.
"""

from __future__ import annotations

import pytest

import tengri
from tengri import DEFAULT, Fixed
from tengri.parameters.parameters import Parameters

pytestmark = pytest.mark.contract

_NEW_PAIRS = ("dpl_lookback", "trunc_exp")


def _build(sfh_type: str, age_gyr: float, end_gyr: float) -> Parameters:
    """Parse a bare SFH of ``sfh_type`` with explicit age_gyr/end_gyr, no noise."""
    return tengri.parse_groups(
        sfh={
            "type": sfh_type,
            "age_gyr": Fixed(age_gyr),
            "end_gyr": Fixed(end_gyr),
            "other_params": Fixed(DEFAULT),
        },
        dust_attenuation={"type": "none"},
        neb={"type": "none"},
        redshift=Fixed(0.1),
    )


@pytest.mark.parametrize("sfh_type", _NEW_PAIRS)
def test_pair_is_codified_in_ordered_pairs(sfh_type):
    """The (age_gyr, end_gyr) pair for sfh_type is a live _ORDERED_PAIRS entry."""
    greater, lesser = f"sfh_{sfh_type}_age_gyr", f"sfh_{sfh_type}_end_gyr"
    pairs = {(g, l) for g, l, _reason in Parameters._ORDERED_PAIRS}
    assert (greater, lesser) in pairs


@pytest.mark.parametrize("sfh_type", _NEW_PAIRS)
def test_inverted_pair_raises(sfh_type):
    """age_gyr <= end_gyr means SF stops before it starts: must raise.

    Only ``Parameters._validate_orderings`` (driven by ``_ORDERED_PAIRS``) can
    catch this -- the per-parameter ``bound_check`` callables are
    ``_lo_positive`` / ``_lo_nonneg`` and cannot see the other parameter, so
    this fails if the pair is removed from ``_ORDERED_PAIRS``.
    """
    with pytest.raises(ValueError, match="must be greater than"):
        _build(sfh_type, age_gyr=2.0, end_gyr=5.0)


@pytest.mark.parametrize("sfh_type", _NEW_PAIRS)
def test_ordered_pair_builds(sfh_type):
    """age_gyr > end_gyr (the physical ordering) builds without error."""
    spec = _build(sfh_type, age_gyr=5.0, end_gyr=2.0)
    age_lo = spec.get_distribution(f"sfh_{sfh_type}_age_gyr").bounds[0]
    end_lo = spec.get_distribution(f"sfh_{sfh_type}_end_gyr").bounds[0]
    assert age_lo == pytest.approx(5.0)
    assert end_lo == pytest.approx(2.0)
    assert age_lo > end_lo
