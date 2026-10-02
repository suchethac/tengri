# SPDX-License-Identifier: BSD-3-Clause
"""The z-narrowed onset-parameter set is derived from the registry, not hand-listed.

``_narrow_free_priors_to_z`` (``parameters/groups.py``) used to iterate a
3-entry hand-written tuple (``_Z_CAPPED_ONSET_PARAMS``): only
``sfh_exp_start_gyr`` / ``sfh_dexp_start_gyr`` / ``sfh_const_start_gyr``. Every
other family's onset/age/peak-time parameter -- the exact same physical role,
a star-formation onset/age/peak lookback time -- kept a static ceiling of
today's cosmic age (13.81 Gyr) even when the build's redshift prior floor
made that ceiling unphysical (#2521). 21 such parameters exist across the SFH
registry; this test asserts every one of them
is now marked in the registry itself (``ParamDef.z_capped_onset``) and that
the marking is what actually drives the narrowing -- not a second hand list
that could drift from the first.
"""

from __future__ import annotations

import pytest

import tengri
from tengri import FREE
from tengri.components.stellar.sfh.registry import SFH_REGISTRY
from tengri.utils.cosmology import age_at_z

pytestmark = pytest.mark.contract

# The definitive list: every onset/age/peak-time
# scalar parameter across the SFH registry, family -> public param name.
_ONSET_PARAMS_BY_FAMILY = {
    "const": "sfh_const_start_gyr",
    "dexp": "sfh_dexp_start_gyr",
    "exp": "sfh_exp_start_gyr",
    "declining_exp": "sfh_declining_exp_age_gyr",
    "delayed": "sfh_delayed_age_gyr",
    "delayed_bq": "sfh_delayed_bq_age_main_gyr",
    "dpl": "sfh_dpl_age_gyr",
    "dpl_lookback": "sfh_dpl_lookback_age_gyr",
    "lnorm": "sfh_lnorm_age_gyr",
    "norm": "sfh_norm_peak_lbt_gyr",
    "periodic": "sfh_periodic_age_gyr",
    "psb": "sfh_psb_age_gyr",
    "sfh2exp": "sfh_sfh2exp_age_gyr",
    "snorm": "sfh_snorm_peak_lbt_gyr",
    "snorm_burst": "sfh_snorm_burst_peak_lbt_gyr",
    "trunc_exp": "sfh_trunc_exp_age_gyr",
    "tsnorm": "sfh_tsnorm_peak_lbt_gyr",
    "tsnorm_burst": "sfh_tsnorm_burst_peak_lbt_gyr",
    "const_exp": "sfh_cexp_age_gyr",
    "psb_flex": "sfh_psb_flex_tflex_gyr",
    "psb_suess2022": "sfh_psb2022_tflex_gyr",
}


def test_exactly_21_onset_params_named():
    """Sanity check on the fixture itself: one entry per onset-bearing family."""
    assert len(_ONSET_PARAMS_BY_FAMILY) == 21


@pytest.mark.parametrize(
    ("family", "public_name"),
    sorted(_ONSET_PARAMS_BY_FAMILY.items()),
)
def test_registry_marks_every_onset_param(family, public_name):
    """Each of the 21 params carries ``ParamDef.z_capped_onset=True`` in the registry."""
    spec = SFH_REGISTRY[family]
    assert public_name in spec.params, f"{public_name!r} not declared on family {family!r}"
    pdef = spec.params[public_name]
    assert getattr(pdef, "z_capped_onset", False) is True, (
        f"{public_name!r} (family {family!r}) is not marked z_capped_onset=True"
    )


@pytest.mark.parametrize(
    ("family", "public_name"),
    sorted(_ONSET_PARAMS_BY_FAMILY.items()),
)
def test_every_marked_param_narrows_to_age_at_z_floor(family, public_name):
    """Under redshift=Uniform(2, 6), every marked param's free-prior ceiling == age_at_z(2)."""
    kwargs = {
        "sfh": {"type": family, "all_params": FREE},
        "dust_attenuation": {"type": "none"},
        "neb": {"type": "none"},
        "redshift": tengri.Uniform(2.0, 6.0),
    }
    spec = tengri.parse_groups(**kwargs)
    assert public_name in spec.free_params, (
        f"{public_name!r} not freed by 'all_params': FREE for family {family!r}"
    )
    _lo, hi = spec.get_distribution(public_name).bounds
    assert hi == pytest.approx(float(age_at_z(2.0)), rel=1e-9), (
        f"{public_name!r} (family {family!r}): ceiling {hi} != age_at_z(2)={float(age_at_z(2.0))}"
    )


def test_no_hand_written_zcap_tuple_left_to_drift():
    """``_narrow_free_priors_to_z`` no longer iterates a hand-written tuple.

    ``_Z_CAPPED_ONSET_PARAMS`` (the old 3-entry tuple) is deleted: the set is
    derived from the registry's own ``z_capped_onset`` flags so a new SFH
    family cannot reintroduce this gap by omission.
    """
    from tengri.parameters import groups

    assert not hasattr(groups, "_Z_CAPPED_ONSET_PARAMS")
