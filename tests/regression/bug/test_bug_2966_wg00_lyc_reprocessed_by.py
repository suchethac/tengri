# SPDX-License-Identifier: BSD-3-Clause
"""``dust_attenuation={'type': 'wg00', 'lyc_reprocessed_by': ...}`` is parsed, stored and reported.

``wg00`` has no young/old split, so ``lyc_reprocessed_by`` does not change the
SED. It must still be read and stored like it is for the other dust types, so
an explicit value survives ``to_groups()``. ``summary()`` marks the field as
not applicable for the types that have no split, so the user is not told it
has an effect. The age-split and hole-geometry keys stay refused for ``wg00``.
"""

from __future__ import annotations

import pytest

import tengri

pytestmark = pytest.mark.regression_bug

_NA_LINE = "LyC reprocessed by: n/a (no young/old split)"


def _build_wg00(ssp, obs, *, neb_type="none", **dust_extra):
    neb = (
        {"type": "cue", "all_params": tengri.Fixed(tengri.DEFAULT)}
        if neb_type == "cue"
        else {"type": "none"}
    )
    return tengri.SEDModel.build(
        ssp,
        observation=obs,
        sfh={"type": "delayed", "all_params": tengri.Fixed(tengri.DEFAULT)},
        dust_attenuation={
            "type": "wg00",
            "all_params": tengri.Fixed(tengri.DEFAULT),
            **dust_extra,
        },
        dust_emission={"type": "none"},
        neb=neb,
        redshift=tengri.Fixed(0.05),
    )


def _build_single(ssp, obs):
    return tengri.SEDModel.build(
        ssp,
        observation=obs,
        sfh={"type": "delayed", "all_params": tengri.Fixed(tengri.DEFAULT)},
        dust_attenuation={
            "type": "single_component",
            "law": "power_law",
            "all_params": tengri.Fixed(tengri.DEFAULT),
        },
        dust_emission={"type": "none"},
        neb={"type": "none"},
        redshift=tengri.Fixed(0.05),
    )


@pytest.mark.parametrize("value", ["all", "young"])
def test_wg00_explicit_lyc_reprocessed_by_lands_on_spec(
    synthetic_ssp_wide, synthetic_tophat_obs, value
):
    m = _build_wg00(synthetic_ssp_wide, synthetic_tophat_obs, lyc_reprocessed_by=value)
    assert m.spec.dust_lyc_reprocessed_by == value
    assert m.spec.dust_lyc_reprocessed_by_explicit is True


@pytest.mark.parametrize("value", ["all", "young"])
def test_wg00_explicit_lyc_reprocessed_by_survives_cue_default(
    synthetic_ssp_wide, synthetic_tophat_obs, value
):
    """Under Cue the default is 'all', so an explicit 'young' is the one that differs."""
    m = _build_wg00(
        synthetic_ssp_wide, synthetic_tophat_obs, neb_type="cue", lyc_reprocessed_by=value
    )
    assert m.spec.dust_lyc_reprocessed_by == value
    assert m.spec.dust_lyc_reprocessed_by_explicit is True


@pytest.mark.parametrize(
    ("neb_type", "value"),
    [("none", "all"), ("cue", "young")],
    ids=["all-without-cue", "young-under-cue"],
)
def test_wg00_explicit_lyc_reprocessed_by_round_trips(
    synthetic_ssp_wide, synthetic_tophat_obs, neb_type, value
):
    """A non-default value is emitted; the default is omitted, as for two_component."""
    m = _build_wg00(
        synthetic_ssp_wide, synthetic_tophat_obs, neb_type=neb_type, lyc_reprocessed_by=value
    )
    groups = m.spec.to_groups()
    assert groups["dust_attenuation"]["lyc_reprocessed_by"] == value
    m2 = tengri.SEDModel.build(
        synthetic_ssp_wide,
        observation=synthetic_tophat_obs,
        **groups,
    )
    assert m2.spec.dust_lyc_reprocessed_by == value
    assert m2.spec.dust_lyc_reprocessed_by_explicit is True


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(lambda s, o: _build_wg00(s, o, lyc_reprocessed_by="all"), id="wg00"),
        pytest.param(lambda s, o: _build_single(s, o), id="single_component"),
    ],
)
def test_summary_marks_lyc_reprocessed_by_not_applicable(
    synthetic_ssp_wide, synthetic_tophat_obs, build
):
    assert _NA_LINE in build(synthetic_ssp_wide, synthetic_tophat_obs).spec.summary_str()


@pytest.mark.parametrize(
    "extra",
    [
        {"t_birth_yr": 1e7},
        {"transition_width_dex": 0.2},
        {"lyc_escape_geometry": "birth_cloud_holes"},
    ],
    ids=["t_birth_yr", "transition_width_dex", "lyc_escape_geometry"],
)
def test_wg00_still_refuses_split_and_geometry_keys(
    synthetic_ssp_wide, synthetic_tophat_obs, extra
):
    with pytest.raises(ValueError):
        _build_wg00(synthetic_ssp_wide, synthetic_tophat_obs, **extra)
