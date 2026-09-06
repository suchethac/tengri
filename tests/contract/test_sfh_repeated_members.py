# SPDX-License-Identifier: BSD-3-Clause
"""One SFH type may appear twice in a composition list.

``sfh={'type': ['norm', 'norm']}`` -- two Gaussian bursts, the shape a
Synthesizer ``CombinedSFH`` reaches for and one no single spec expresses --
used to be unbuildable::

    ValueError: Parameter name collision: 'sfh_norm_log_total_mass'
    appears in multiple models

raised out of ``resolve_sfh`` while merging the two identical specs, and again
from ``SEDModel.build`` one frame later through ``_build_param_registry``.

The k-th occurrence of a type (k >= 2) now takes the type's public prefix with
the ordinal appended: ``sfh_norm_log_total_mass`` for the first instance,
``sfh_norm_2_log_total_mass`` for the second, with the same priors, defaults
and internal kwargs, and its own dispatch inside the composed closure. The
grammar's short key keeps the ordinal too (``norm_2_log_total_mass``), which is
what the round trip emits. Abbreviated families are numbered on the prefix they
actually declare, so ``continuity`` repeats as ``sfh_cont_2_ratio_0``.
"""

from __future__ import annotations

import numpy as np
import pytest

from tengri import DEFAULT, FREE, Fixed, SEDModel
from tengri.components.stellar.sfh.registry import (
    SFH_REGISTRY,
    _spec_public_prefix,
    resolve_sfh,
)

pytestmark = pytest.mark.contract

_W = Fixed(DEFAULT)
_Z = Fixed(0.05)

# Widths chosen above the SSP grid spacing at each peak (0.122 Gyr at 1 Gyr,
# 0.612 Gyr at 5 Gyr) so neither member trips SFHBurstAliasingWarning (#299).
_FIRST = {
    "sfh_norm_log_total_mass": Fixed(10.0),
    "sfh_norm_peak_lbt_gyr": Fixed(1.0),
    "sfh_norm_width_gyr": Fixed(0.2),
}
_SECOND = {
    "sfh_norm_2_log_total_mass": Fixed(9.7),
    "sfh_norm_2_peak_lbt_gyr": Fixed(5.0),
    "sfh_norm_2_width_gyr": Fixed(0.8),
}
_SECOND_ALONE = {
    "sfh_norm_log_total_mass": Fixed(9.7),
    "sfh_norm_peak_lbt_gyr": Fixed(5.0),
    "sfh_norm_width_gyr": Fixed(0.8),
}
_CONST = {
    "sfh_const_log_total_mass": Fixed(10.2),
    "sfh_const_start_gyr": Fixed(8.0),
    "sfh_const_end_gyr": Fixed(0.0),
}


def _model(ssp, sfh):
    return SEDModel.build(ssp_data=ssp, sfh={"all_params": _W, **sfh}, redshift=_Z)


def _state(ssp, sfh):
    return _model(ssp, sfh).predict_state({})


def test_resolve_sfh_numbers_the_second_instance():
    """['norm', 'norm'] declares both sfh_norm_* and sfh_norm_2_*, mapped to one internal name."""
    _fn, params, param_map, _settings = resolve_sfh(["norm", "norm"])

    assert set(params) == {
        "sfh_norm_log_total_mass",
        "sfh_norm_peak_lbt_gyr",
        "sfh_norm_width_gyr",
        "sfh_norm_2_log_total_mass",
        "sfh_norm_2_peak_lbt_gyr",
        "sfh_norm_2_width_gyr",
    }
    # The repeat carries the base entry's prior and its internal kwarg.
    assert params["sfh_norm_2_width_gyr"] == params["sfh_norm_width_gyr"]
    assert param_map["sfh_norm_2_log_total_mass"] == param_map["sfh_norm_log_total_mass"]

    _fn3, params3, _map3, _s3 = resolve_sfh(["norm", "norm", "norm"])
    assert "sfh_norm_3_width_gyr" in params3


def test_every_registered_type_can_be_numbered_without_colliding():
    """Each spec exposes a public prefix, and its numbered names are all new.

    Nine families abbreviate their prefix (``continuity`` declares
    ``sfh_cont_*``, ``dense_basis`` ``sfh_db_*``, ...), so the ordinal cannot
    simply be inserted after ``sfh_<type>_``. Measured over the 35 registered
    types: every one yields a prefix and no numbered name lands on a name some
    other type already declares.
    """
    declared = {name for spec in SFH_REGISTRY.values() for name in spec.params}
    for type_name, spec in SFH_REGISTRY.items():
        if not spec.params:
            continue
        prefix = _spec_public_prefix(spec)
        assert prefix, f"{type_name} declares no shared public prefix"
        for full_name in spec.params:
            assert full_name.startswith(prefix)
            numbered = f"{prefix}2_{full_name[len(prefix) :]}"
            assert numbered not in declared, f"{numbered} collides with a declared name"


def test_two_norms_are_the_sum_of_the_two_single_norm_histories(ssp_data_fsps):
    """['norm', 'norm'] with different peaks equals norm(1 Gyr) + norm(5 Gyr)."""
    s_first = _state(ssp_data_fsps, {"type": "norm", **_FIRST})
    s_second = _state(ssp_data_fsps, {"type": "norm", **_SECOND_ALONE})
    s_both = _state(ssp_data_fsps, {"type": ["norm", "norm"], **_FIRST, **_SECOND})

    sfr_sum = np.asarray(s_first.derived["sfr_history"]) + np.asarray(
        s_second.derived["sfr_history"]
    )
    np.testing.assert_allclose(
        np.asarray(s_both.derived["sfr_history"]), sfr_sum, rtol=1e-10, atol=0.0
    )

    expected = np.log10(10.0**10.0 + 10.0**9.7)
    assert abs(float(s_both.derived["log_mstar_formed"]) - expected) < 1e-6


def test_the_two_instances_keep_their_own_values(ssp_data_fsps):
    """Swapping which instance gets the 5 Gyr peak changes the history."""
    swapped_first = {
        "sfh_norm_log_total_mass": Fixed(9.7),
        "sfh_norm_peak_lbt_gyr": Fixed(5.0),
        "sfh_norm_width_gyr": Fixed(0.8),
    }
    swapped_second = {
        "sfh_norm_2_log_total_mass": Fixed(10.0),
        "sfh_norm_2_peak_lbt_gyr": Fixed(1.0),
        "sfh_norm_2_width_gyr": Fixed(0.2),
    }
    s_a = _state(ssp_data_fsps, {"type": ["norm", "norm"], **_FIRST, **_SECOND})
    s_b = _state(ssp_data_fsps, {"type": ["norm", "norm"], **swapped_first, **swapped_second})

    # Same two components, listed the other way round: identical history.
    np.testing.assert_allclose(
        np.asarray(s_b.derived["sfr_history"]),
        np.asarray(s_a.derived["sfr_history"]),
        rtol=1e-10,
        atol=0.0,
    )

    # But giving the second instance the FIRST instance's values collapses the
    # composition onto one peak, which it must not silently do.
    s_dup = _state(
        ssp_data_fsps,
        {
            "type": ["norm", "norm"],
            **_FIRST,
            "sfh_norm_2_log_total_mass": Fixed(10.0),
            "sfh_norm_2_peak_lbt_gyr": Fixed(1.0),
            "sfh_norm_2_width_gyr": Fixed(0.2),
        },
    )
    assert not np.allclose(
        np.asarray(s_dup.derived["sfr_history"]), np.asarray(s_a.derived["sfr_history"])
    )


def test_a_repeat_composes_with_a_different_type(ssp_data_fsps):
    """['const', 'norm', 'norm'] is const + norm + norm."""
    s_const = _state(ssp_data_fsps, {"type": "const", **_CONST})
    s_first = _state(ssp_data_fsps, {"type": "norm", **_FIRST})
    s_second = _state(ssp_data_fsps, {"type": "norm", **_SECOND_ALONE})
    s_all = _state(
        ssp_data_fsps, {"type": ["const", "norm", "norm"], **_CONST, **_FIRST, **_SECOND}
    )

    sfr_sum = (
        np.asarray(s_const.derived["sfr_history"])
        + np.asarray(s_first.derived["sfr_history"])
        + np.asarray(s_second.derived["sfr_history"])
    )
    np.testing.assert_allclose(
        np.asarray(s_all.derived["sfr_history"]), sfr_sum, rtol=1e-10, atol=0.0
    )
    expected = np.log10(10.0**10.2 + 10.0**10.0 + 10.0**9.7)
    assert abs(float(s_all.derived["log_mstar_formed"]) - expected) < 1e-6


def test_short_form_keys_reach_the_repeat(ssp_data_fsps):
    """'norm_2_peak_lbt_gyr' is the repeat's short key; the base keeps the bare one."""
    s_full = _state(ssp_data_fsps, {"type": ["norm", "norm"], **_FIRST, **_SECOND})
    s_short = _state(
        ssp_data_fsps,
        {
            "type": ["norm", "norm"],
            "log_total_mass": Fixed(10.0),
            "peak_lbt_gyr": Fixed(1.0),
            "width_gyr": Fixed(0.2),
            "norm_2_log_total_mass": Fixed(9.7),
            "norm_2_peak_lbt_gyr": Fixed(5.0),
            "norm_2_width_gyr": Fixed(0.8),
        },
    )
    np.testing.assert_allclose(
        np.asarray(s_short.derived["sfr_history"]),
        np.asarray(s_full.derived["sfr_history"]),
        rtol=1e-12,
        atol=0.0,
    )


def test_a_typo_in_the_second_members_name_is_refused(ssp_data_fsps):
    """A misspelled repeat key raises and names the parameter it nearly is."""
    with pytest.raises(ValueError, match="Unknown key 'sfh_norm_2_peak_gyr'") as excinfo:
        _model(
            ssp_data_fsps,
            {"type": ["norm", "norm"], **_FIRST, "sfh_norm_2_peak_gyr": Fixed(5.0)},
        )
    assert "sfh_norm_2_peak_lbt_gyr" in str(excinfo.value)


def test_the_wildcard_covers_the_repeat(ssp_data_fsps):
    """sfh={'all_params': FREE} frees the numbered parameters too."""
    model = SEDModel.build(
        ssp_data=ssp_data_fsps,
        sfh={"type": ["norm", "norm"], "all_params": FREE},
        redshift=_Z,
    )
    free = set(model.spec.free_params)
    for name in ("sfh_norm_width_gyr", "sfh_norm_2_width_gyr", "sfh_norm_2_log_total_mass"):
        assert name in free, f"{name} was not freed by the sfh wildcard"


def test_the_repeated_names_round_trip_through_to_groups(ssp_data_fsps):
    """to_groups emits the numbered short keys and rebuilds the same model."""
    model = _model(ssp_data_fsps, {"type": ["norm", "norm"], **_FIRST, **_SECOND})
    groups = model.spec.to_groups()

    assert groups["sfh"]["type"] == ["norm", "norm"]
    assert groups["sfh"]["norm_2_peak_lbt_gyr"] == Fixed(5.0)
    assert groups["sfh"]["peak_lbt_gyr"] == Fixed(1.0)

    rebuilt = SEDModel.build(ssp_data=ssp_data_fsps, **groups)
    assert rebuilt.spec.free_params == model.spec.free_params
    np.testing.assert_allclose(
        np.asarray(rebuilt.predict_state({}).derived["sfr_history"]),
        np.asarray(model.predict_state({}).derived["sfr_history"]),
        rtol=1e-12,
        atol=0.0,
    )


def test_the_burst_aliasing_warning_reaches_the_repeat(ssp_data_fsps):
    """A too-narrow width on the second norm warns, naming the numbered parameter.

    ``_BURST_WIDTH_TO_PEAK`` is a static map of exact parameter names, so the
    numbered repeat would otherwise slip past the one guard that exists for
    narrow Gaussian bursts -- in the very configuration repeats make possible.
    """
    from tengri.components.stellar.sfh._aliasing_warning import SFHBurstAliasingWarning

    with pytest.warns(SFHBurstAliasingWarning, match="sfh_norm_2_width_gyr"):
        _model(
            ssp_data_fsps,
            {
                "type": ["norm", "norm"],
                **_FIRST,
                "sfh_norm_2_log_total_mass": Fixed(9.7),
                "sfh_norm_2_peak_lbt_gyr": Fixed(5.0),
                "sfh_norm_2_width_gyr": Fixed(0.05),
            },
        )


@pytest.mark.parametrize(
    ("types", "message"),
    [
        (["burst", "burst"], "At most one mixture component"),
        (["const", "burst", "burst"], "At most one mixture component"),
        (["const", "field", "field"], "At most one modulator component"),
    ],
    ids=["burst_twice", "const_burst_burst", "field_twice"],
)
def test_a_repeated_compositor_is_still_refused(types, message):
    """Numbering additive repeats must not let a second burst or field through."""
    with pytest.raises(ValueError, match=message):
        resolve_sfh(types)
