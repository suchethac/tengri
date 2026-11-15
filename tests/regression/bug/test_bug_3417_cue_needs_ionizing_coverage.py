# SPDX-License-Identifier: BSD-3-Clause
"""Regression test for #3417: Cue refuses an SSP grid with no ionizing coverage, by name.

An SSP wavelength axis that starts above the Lyman limit (912 A) has no ionizing
spectrum. The table builder used to reduce an empty slice and die with a numpy
zero-size error. The backend now refuses such a grid by name before any table work.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from tengri import DEFAULT, Fixed, SEDModel
from tengri.components.stellar.sps.dsps_wrapper import SSPData
from tengri.config.exceptions import ConfigError
from tests._data_skip import requires_cue_weights

pytestmark = [pytest.mark.regression_bug, requires_cue_weights]


def _ssp_on(wave) -> SSPData:
    n_met, n_age = 8, 15
    n_wave = wave.shape[0]
    rng = np.random.default_rng(0)
    return SSPData(
        ssp_wave=wave,
        ssp_flux=jnp.asarray(rng.uniform(0.5, 1.5, (n_met, n_age, n_wave)), dtype=jnp.float64),
        ssp_lg_age_gyr=jnp.linspace(-3.0, 1.1, n_age),
        ssp_lgmet=jnp.linspace(-2.0, 0.3, n_met),
    )


def _cue_model(ssp: SSPData) -> SEDModel:
    return SEDModel.build(
        ssp,
        neb={"type": "cue", "all_params": Fixed(DEFAULT)},
        redshift=Fixed(0.1),
    )


class TestBug3417CueIonizingCoverage:
    """cue.py::CueBackend._precompute_ionizing_params refuses a truncated SSP axis."""

    def test_axis_above_lyman_limit_is_refused_by_name(self):
        ssp = _ssp_on(jnp.logspace(3.0, 4.5, 200))  # starts at 1000 A
        with pytest.raises(ConfigError) as info:
            _cue_model(ssp)
        message = str(info.value)
        assert "912" in message
        assert "cue" in message

    def test_axis_extended_below_lyman_limit_builds(self):
        ssp = _ssp_on(jnp.logspace(2.0, 4.5, 200))  # reaches 100 A
        model = _cue_model(ssp)
        assert model is not None
