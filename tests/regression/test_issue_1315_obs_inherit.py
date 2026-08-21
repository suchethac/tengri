# SPDX-License-Identifier: BSD-3-Clause
"""Test ForwardModel.build observation inheritance and LUT-mismatch guard (#1315).

Regression test for:
- ForwardModel.build can omit observation and inherit from sed.observation
- Different observation with baked LUT raises ValueError
- populations= branch still requires observation
"""

import pytest

from tengri import Fixed, ForwardModel, Observation, Photometry, SEDModel, WavePrecomp

pytestmark = pytest.mark.regression_bug


def _fingerprint(obs):
    """Compute content-hash of filter transmission curves."""
    import numpy as np

    if obs.photometry is None:
        return None
    return hash(tuple(np.asarray(t).tobytes() for t in obs.photometry.filter_trans))


@pytest.fixture
def obs_a(synthetic_ssp):
    """Two-band observation: SDSS g + r."""
    return Observation(photometry=Photometry.from_names(["sdss_g", "sdss_r"]))


@pytest.fixture
def obs_b(synthetic_ssp):
    """Two-band observation: SDSS i + z (different from obs_a)."""
    return Observation(photometry=Photometry.from_names(["sdss_i", "sdss_z"]))


def test_omitted_observation_inherits(synthetic_ssp, obs_a):
    """ForwardModel.build with no observation kwarg inherits from sed."""
    sed = SEDModel.build(
        ssp_data=synthetic_ssp, observation=obs_a, sfh={"type": "dpl"}, redshift=Fixed(0.1)
    )
    fwd = ForwardModel.build(sed=sed)  # no observation kwarg
    assert _fingerprint(fwd.observation) == _fingerprint(obs_a)


def test_different_filters_no_lut_allowed(synthetic_ssp, obs_a, obs_b):
    """ForwardModel.build can accept different observation if sed has no LUT."""
    sed = SEDModel.build(
        ssp_data=synthetic_ssp, observation=obs_a, sfh={"type": "dpl"}, redshift=Fixed(0.1)
    )
    fwd = ForwardModel.build(sed=sed, observation=obs_b)  # explicit obs_b: fine
    assert _fingerprint(fwd.observation) == _fingerprint(obs_b)


def test_different_filters_with_lut_raises(synthetic_ssp, obs_a, obs_b):
    """ForwardModel.build raises when filters differ and sed has baked LUT."""
    sed = SEDModel.build(
        ssp_data=synthetic_ssp,
        observation=obs_a,
        sfh={"type": "dpl"},
        approx=WavePrecomp(),
        redshift=Fixed(0.1),
    )
    with pytest.raises(ValueError, match="LUT"):
        ForwardModel.build(sed=sed, observation=obs_b)


def test_population_branch_still_requires_observation(synthetic_ssp, obs_a):
    """ForwardModel.build(populations=...) still requires observation explicitly."""
    with pytest.raises(TypeError):
        ForwardModel.build(populations=[])  # no obs: must not silently pass
