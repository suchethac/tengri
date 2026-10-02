# SPDX-License-Identifier: BSD-3-Clause
"""Contract: one exact young/old split, from per-node formed-mass fractions.

The stellar component publishes ``age_boundary_younger_fraction`` (n_boundary,
n_age): the share of each SSP node's formed mass younger than the boundary,
computed through the same age kernel as the node weights. Every attenuator
reads it, so the split does not depend on where the SSP nodes happen to sit.

Checks, on the synthetic wide SSP and a delayed-tau SFH:

* the publication has the declared shape and range, for both age kernels;
* the CIC and DSPS kernels agree on it to the kernels' own resolution;
* ``transition_width_dex -> 0`` approaches the hard step; moving
  ``t_birth_yr`` later only adds young mass;
* the grammar keys round-trip through ``spec.to_groups`` and are refused on a
  type that has no age split.
"""

import numpy as np
import pytest

from tengri import DEFAULT, Fixed, SEDModel

pytestmark = pytest.mark.contract

SFH = {
    "type": "delayed",
    "tau_gyr": Fixed(1.0),
    "age_gyr": Fixed(5.0),
    "log_total_mass": Fixed(10.0),
    "all_params": Fixed(DEFAULT),
}


def _build(ssp, *, age_kernel="cic", **dust_extra):
    dust = {
        "law": "power_law",
        "type": "two_component",
        "tau_bc": Fixed(0.5),
        "tau_diff": Fixed(0.3),
        "all_params": Fixed(DEFAULT),
    }
    dust.update(dust_extra)
    return SEDModel.build(
        ssp_data=ssp,
        met={"logzsol": Fixed(0.0), "all_params": Fixed(DEFAULT)},
        sfh=dict(SFH, age_kernel=age_kernel),
        dust_attenuation=dust,
        neb={"type": "none"},
        redshift=Fixed(0.05),
    )


def _fraction(model):
    return np.asarray(model.predict_state({}).derived["age_boundary_younger_fraction"])


@pytest.mark.parametrize("kernel", ["cic", "dsps"])
def test_publication_has_declared_shape_and_range(synthetic_ssp_wide, kernel):
    model = _build(synthetic_ssp_wide, age_kernel=kernel)
    frac = _fraction(model)
    assert frac.shape == (1, np.asarray(synthetic_ssp_wide.ssp_lg_age_gyr).shape[0])
    assert np.all(np.isfinite(frac))
    assert frac.min() >= 0.0 and frac.max() <= 1.0
    # Mass formed over 5 Gyr is overwhelmingly old: the oldest nodes hold none young.
    assert frac[0, -1] == 0.0


def test_cic_and_dsps_kernels_agree_on_the_young_fraction(synthetic_ssp_wide):
    """Both kernels describe one SFH; node fractions differ only by kernel resolution."""
    cic = _fraction(_build(synthetic_ssp_wide, age_kernel="cic"))
    dsps = _fraction(_build(synthetic_ssp_wide, age_kernel="dsps"))
    young_nodes = np.argwhere((cic[0] > 1e-3) | (dsps[0] > 1e-3)).ravel()
    assert young_nodes.size > 0
    np.testing.assert_allclose(cic[0, young_nodes], dsps[0, young_nodes], atol=0.25)


def test_later_birth_cloud_lifetime_only_adds_young_mass(synthetic_ssp_wide):
    early = _fraction(_build(synthetic_ssp_wide, t_birth_yr=3.0e6))
    late = _fraction(_build(synthetic_ssp_wide, t_birth_yr=3.0e7))
    assert np.all(late >= early - 1e-12)
    assert late.sum() > early.sum()


def test_narrow_transition_width_approaches_the_hard_step(synthetic_ssp_wide):
    step = _fraction(_build(synthetic_ssp_wide))
    errs = [
        float(np.max(np.abs(_fraction(_build(synthetic_ssp_wide, transition_width_dex=w)) - step)))
        for w in (0.3, 0.03, 0.003)
    ]
    assert errs[0] > errs[2]
    assert errs[2] < 0.05


def test_grammar_keys_round_trip(synthetic_ssp_wide):
    model = _build(synthetic_ssp_wide, t_birth_yr=2.0e7, transition_width_dex=0.1)
    dust = model.spec.to_groups()["dust_attenuation"]
    assert dust["t_birth_yr"] == pytest.approx(2.0e7)
    assert dust["transition_width_dex"] == pytest.approx(0.1)
    rebuilt = _build(
        synthetic_ssp_wide, **{k: dust[k] for k in ("t_birth_yr", "transition_width_dex")}
    )
    np.testing.assert_array_equal(_fraction(rebuilt), _fraction(model))


def test_age_split_keys_are_refused_without_an_age_split(synthetic_ssp_wide):
    with pytest.raises(ValueError, match="t_birth_yr"):
        SEDModel.build(
            ssp_data=synthetic_ssp_wide,
            sfh=SFH,
            dust_attenuation={
                "type": "single_component",
                "law": "power_law",
                "tau_v": Fixed(0.3),
                "t_birth_yr": 1e7,
                "all_params": Fixed(DEFAULT),
            },
            redshift=Fixed(0.05),
        )
