# SPDX-License-Identifier: BSD-3-Clause
"""``dust={'type': 'none'}`` disables the dust block, like every other group.

Fresh-user audit (2026-07): every composable group (neb, agn, radio, xray,
igm, shock) accepts ``type='none'`` to disable it, and the generic grammar
error even promises ``{group}={'type': 'none'} / omit to disable`` — but dust
raised ``ValueError: Unknown dust type 'none'``. The forward model already reads
``dust_model == 'off'`` as ``use_dust=False``; the grammar and Parameters just
never produced that sentinel. These tests pin the disable path end to end.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

from tengri import FIXED, Fixed, Parameters, SEDModel

pytestmark = [pytest.mark.contract, pytest.mark.regression_bug]


def _phot(ssp, obs, **build):
    model = SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        redshift=Fixed(0.1),
        sfh={"type": "dpl", "*": FIXED},
        **build,
    )
    phot = np.asarray(model.predict_photometry(model.spec.sample(jax.random.PRNGKey(0))))
    return model, phot


def test_dust_none_disables_dust(synthetic_ssp_wide, synthetic_tophat_obs):
    model, phot = _phot(synthetic_ssp_wide, synthetic_tophat_obs, dust={"type": "none"})
    assert model._dust_model == "off"  # internal disable sentinel
    assert np.all(np.isfinite(phot))


def test_dust_none_matches_omitting_dust(synthetic_ssp_wide, synthetic_tophat_obs):
    """A dust-free model built with 'none' is not attenuated."""
    _, none = _phot(synthetic_ssp_wide, synthetic_tophat_obs, dust={"type": "none"})
    _, dusty = _phot(
        synthetic_ssp_wide,
        synthetic_tophat_obs,
        dust={
            "type": "two_component",
            "law_bc": "calzetti",
            "law_diff": "calzetti",
            "*": FIXED,
            "tau_bc": Fixed(3.0),
            "tau_diff": Fixed(3.0),
        },
    )
    # Heavy attenuation must pull the bluest band well below the dust-free flux.
    assert dusty[0] < none[0] * 0.5


def test_dust_none_with_wildcard_builds(synthetic_ssp_wide, synthetic_tophat_obs):
    model, phot = _phot(
        synthetic_ssp_wide, synthetic_tophat_obs, dust={"type": "none", "*": FIXED}
    )
    assert model._dust_model == "off"
    assert np.all(np.isfinite(phot))


def test_dust_emission_none_builds(synthetic_ssp_wide, synthetic_tophat_obs):
    _, phot = _phot(
        synthetic_ssp_wide,
        synthetic_tophat_obs,
        dust={
            "law": "power_law",
            "type": "two_component",
            "*": FIXED,
            "emission": {"type": "none"},
        },
    )
    assert np.all(np.isfinite(phot))


@pytest.mark.parametrize("value", ["off", "none"])
def test_parameters_accepts_off_and_none(value):
    """The flat Parameters escape hatch accepts the disable sentinel too,
    normalizing 'none' onto the internal 'off'."""
    spec = Parameters(mean_sfh_type="dpl", dust_model=value)
    assert spec.dust_model == "off"


def test_parameters_rejects_unknown_dust_model():
    with pytest.raises(ValueError, match="dust_model must be"):
        Parameters(mean_sfh_type="dpl", dust_model="not_a_real_model")
