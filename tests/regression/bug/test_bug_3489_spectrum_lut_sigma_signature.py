# SPDX-License-Identifier: BSD-3-Clause
"""Two SpectrumPrecomp models with different fixed sigma_v share no compile signature (#3489).

``approx=SpectrumPrecomp()`` on a fixed redshift pre-broadens the SSP spectrum table by the
fixed ``sigma_v_kms`` at build time. Two such models in one process, with ``sigma_v_kms``
Fixed(0.0) and Fixed(150.0), used to compare equal in ``compile_signature()``; the shared
kernel then closed over the first model's table, and the second model's eager
``predict_observables`` refused with "the spectrum LUT was built with sigma_v_kms=0.0 but the
draw has 150.0". Contract: the two signatures differ, and every prediction surface of the
second model equals its result when it was evaluated before the first model existed.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from tengri import DEFAULT, Fixed, Observation, SEDModel, Spectroscopy
from tengri.components.stellar.sps.dsps_wrapper import SSPData
from tengri.forward.sed_model import SpectrumPrecomp
from tengri.observation.photometry_config import Photometry

pytestmark = pytest.mark.regression_bug

N_MET, N_AGE, N_WAVE = 3, 12, 400


def _ssp3d() -> SSPData:
    """A small 3-D (n_met, n_age, n_wave) grid with a Mg-b-like trough."""
    wave = jnp.linspace(3000.0, 10000.0, N_WAVE)
    ages = jnp.linspace(-1.0, 1.14, N_AGE)
    rng = np.random.default_rng(3489)
    base = rng.uniform(1e-3, 2e-3, (N_MET, N_AGE, N_WAVE))
    feature = np.exp(-0.5 * ((np.asarray(wave) - 5175.0) / 40.0) ** 2)
    flux = base * (1.0 - 0.3 * feature)[None, None, :]
    return SSPData(
        ssp_wave=wave,
        ssp_flux=jnp.asarray(flux),
        ssp_lg_age_gyr=ages,
        ssp_lgmet=jnp.array([-2.5, -2.0, -1.5]),
    )


def _observation() -> Observation:
    wave_obs = jnp.linspace(4000.0, 8000.0, 400)
    return Observation(
        photometry=Photometry.from_names(["sdss_g", "sdss_r"]),
        spectroscopy=Spectroscopy(wave_obs=wave_obs, resolution=jnp.full(400, 2000.0)),
    )


def _model(ssp: SSPData, sigma_v_kms: float) -> SEDModel:
    """Fixed-redshift SpectrumPrecomp model with a Fixed sigma_v_kms merged into its spec."""
    kwargs = dict(
        ssp_data=ssp,
        observation=_observation(),
        sfh={
            "type": "delayed",
            "all_params": Fixed(DEFAULT),
            "tau_gyr": Fixed(2.0),
            "age_gyr": Fixed(3.0),
            "log_total_mass": Fixed(10.0),
        },
        met={"logzsol": Fixed(-0.3)},
        redshift=Fixed(0.05),
        neb={"type": "ssp"},
    )
    base = SEDModel.build(**kwargs, approx=None)
    spec = base.spec.merge_observation_params(sigma_v_kms=Fixed(sigma_v_kms))
    return SEDModel(spec, ssp, observation=_observation(), approx=SpectrumPrecomp())


def _surfaces(model: SEDModel) -> dict[str, np.ndarray]:
    """Every spectrum-producing surface a fitter or user reaches, as arrays."""
    params = {}
    return {
        "predict_spectrum": np.asarray(model.predict_spectrum(params)),
        "predict().spectrum()": np.asarray(model.predict(params).spectrum()),
        "predict_observables": np.asarray(model.predict_observables(params).spec_fnu),
    }


def test_sigma_v_kms_is_part_of_the_compile_signature():
    ssp = _ssp3d()
    m1 = _model(ssp, 0.0)
    m2 = _model(ssp, 150.0)
    assert m1.compile_signature() != m2.compile_signature()


def test_second_model_matches_itself_alone_on_every_surface():
    ssp = _ssp3d()
    m2_alone = _surfaces(_model(ssp, 150.0))

    m1 = _model(ssp, 0.0)
    _surfaces(m1)
    m2 = _model(ssp, 150.0)
    m2_after_m1 = _surfaces(m2)

    for name, expected in m2_alone.items():
        np.testing.assert_array_equal(m2_after_m1[name], expected, err_msg=name)
