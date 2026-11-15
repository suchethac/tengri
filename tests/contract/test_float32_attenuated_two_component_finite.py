# SPDX-License-Identifier: BSD-3-Clause
"""Float32 two-component dust on a 4-D [alpha/Fe] grid keeps the attenuated SED finite.

The grouped age mixture reads the stellar component's per-interval spectra
``lnu_interval``. On a 4-D grid they were scaled by ``total_mass * L_sun``, an erg/s-scale
linear factor: for a dwarf (1e8 Msun, L_sun = 3.8e33) it is 3.8e41, above the float32
maximum 3.4e38, so every interval spectrum, the intrinsic stellar SED and
``sed_dust_attenuated`` came out non-finite in float32 while float64 was fine.
Contract: float32 equals float64 to 1e-4 relative at a dwarf's mass.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import DEFAULT, SEDModel, Uniform
from tengri.components.stellar.sps.dsps_wrapper import SSPData
from tengri.parameters import Fixed
from tengri.utils.physics_constants import LOG10_ZSUN

pytestmark = pytest.mark.contract

ALPHA_NODES = np.array([-0.2, 0.0, 0.2, 0.4, 0.6])
LOG_MASS_DWARF = 8.0
RTOL = 1e-4
#: Puts the grid at a real library's level, about 1e-16 L_sun/Hz/Msun in the optical.
FLUX_SCALE = 1e-13


def _ssp4d() -> SSPData:
    n_met, n_age, n_wave = 3, 20, 300
    wave = np.linspace(3000.0, 10000.0, n_wave)
    ages = np.linspace(-1.0, 1.14, n_age)
    noise = np.abs(np.random.default_rng(7).standard_normal((n_met, n_age, n_wave)))
    base = (noise * 1e-3 + 1e-3) * FLUX_SCALE
    feature = np.exp(-0.5 * ((wave - 5175.0) / 40.0) ** 2)
    cubes = [base * (1.0 - 0.25 * (1.0 + a) * feature)[None, None, :] for a in ALPHA_NODES]
    return SSPData(
        ssp_wave=jnp.asarray(wave),
        ssp_flux=jnp.asarray(np.stack(cubes, axis=1)),
        ssp_lg_age_gyr=jnp.asarray(ages),
        ssp_lgmet=jnp.asarray(np.array([-1.5, -0.5, 0.0]) + LOG10_ZSUN),
        ssp_alpha_fe=jnp.asarray(ALPHA_NODES),
    )


def _attenuated(x64: bool) -> np.ndarray:
    with jax.enable_x64(x64):
        model = SEDModel.build(
            ssp_data=_ssp4d(),
            sfh={
                "type": "delayed",
                "all_params": Fixed(DEFAULT),
                "tau_gyr": Fixed(2.0),
                "age_gyr": Fixed(3.0),
                "log_total_mass": Fixed(LOG_MASS_DWARF),
            },
            met={"logzsol": Fixed(-0.3), "alpha_fe": Uniform(-0.2, 0.6)},
            dust_attenuation={
                "law": "power_law",
                "type": "two_component",
                "age_mixture": "grouped",
                "transition_width_dex": 0.3,
                "tau_bc": Fixed(1.25),
                "tau_diff": Fixed(0.17),
                "all_params": Fixed(DEFAULT),
            },
            neb={"type": "none"},
            redshift=Fixed(0.05),
        )
        dtype = jnp.float64 if x64 else jnp.float32
        state = model.predict_state({"met_alpha_fe": jnp.asarray(0.2, dtype)})
        return np.asarray(state.derived["sed_dust_attenuated"], np.float64)


def test_float32_grouped_attenuated_sed_is_finite_and_matches_float64():
    ref = _attenuated(True)
    got = _attenuated(False)
    assert np.all(np.isfinite(ref)) and np.max(ref) > 0.0
    assert np.all(np.isfinite(got)), f"{int((~np.isfinite(got)).sum())} of {got.size} non-finite"
    np.testing.assert_allclose(got, ref, rtol=RTOL, atol=0.0)
