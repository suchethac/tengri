# SPDX-License-Identifier: BSD-3-Clause
"""Regression: an SSP age axis in log10(yr) is refused, not read as log10(Gyr) (#3337 b).

``SSPData.ssp_lg_age_gyr`` is log10(age / Gyr). ``SEDModel`` adds 9.0 to get
log10(yr). A grid whose ages were passed in log10(yr) (for example
``linspace(6, 10.1)``) was accepted silently: its youngest template became
1e15 yr, older than the universe, and every history younger than 1 Gyr fell
into no age bin. The model now refuses an age axis whose oldest template is
older than the age of the universe (max log10 age > 1.2, i.e. > 15.8 Gyr).

Issue: https://github.com/suchethac/tengri/issues/3337
"""

import jax.numpy as jnp
import numpy as np
import pytest

from tengri import Parameters, SEDModel, Uniform
from tengri.components.stellar.sps.dsps_wrapper import SSPData, load_ssp
from tengri.forward.sed_model import _validate_ssp_age_axis_units
from tengri.observation.observation import Observation
from tengri.observation.photometry_config import Photometry

pytestmark = pytest.mark.regression_bug


def _ssp_with_ages(lg_age: np.ndarray) -> SSPData:
    n_met, n_wave = 3, 50
    n_age = lg_age.shape[0]
    rng = np.random.default_rng(1)
    return SSPData(
        ssp_wave=jnp.logspace(3, 4.5, n_wave),
        ssp_flux=jnp.asarray(rng.uniform(0.5, 1.5, (n_met, n_age, n_wave)), dtype=jnp.float64),
        ssp_lg_age_gyr=jnp.asarray(lg_age),
        ssp_lgmet=jnp.linspace(-2.0, 0.0, n_met),
    )


def _model(ssp: SSPData) -> SEDModel:
    spec = Parameters(
        redshift=0.1,
        sfh_dpl_alpha=Uniform(0.5, 4.0),
        sfh_dpl_beta=Uniform(0.3, 3.0),
    )
    obs = Observation(photometry=Photometry.from_names(["sdss_g", "sdss_r", "sdss_i"]))
    return SEDModel(spec, ssp, observation=obs)


def test_age_axis_in_log_yr_is_refused():
    """Ages passed as log10(yr) (6 .. 10.1) are an age-units bug, not a grid."""
    ssp = _ssp_with_ages(np.linspace(6.0, 10.1, 15))
    with pytest.raises(ValueError, match=r"log10\(yr\)"):
        _model(ssp)


def test_shipped_default_grid_is_accepted():
    """The default SSP grid is in log10(Gyr) with a youngest age near 0.3 Myr."""
    ssp = load_ssp()
    assert float(np.max(np.asarray(ssp.ssp_lg_age_gyr))) < 1.2
    model = _model(ssp)
    assert model is not None


def test_bpass_style_axis_with_100_gyr_template_is_accepted():
    """A log10(Gyr) axis reaching 100 Gyr (shipped BPASS grid) is not log10(yr)."""
    _validate_ssp_age_axis_units(np.linspace(-3.0, 2.0, 51))
