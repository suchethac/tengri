# SPDX-License-Identifier: BSD-3-Clause
"""#3541: the free-z pre-broadened spectrum cube reads metallicity from native slopes.

With a free redshift and a fixed velocity dispersion, the spectrum route stores the SSP
cube broadened once at build (``spec_ssp_broadened``). Before the native-slope fix that
cube was read across metallicity by monotone PCHIP over its own stored values, which is
not the PCHIP of the native cube, so the table route differed from the per-draw sum route
over the native cube (5.3e-5 at z = 0.08 on the wNE grid, 4.8e-5 at z = 0.12).

The fix (owner ruling 2026-11-15, "Exact: native slopes") broadens the native PCHIP
tangents with the same linear build and reads the cube by cubic Hermite of (values,
slopes). The reference here is the per-draw sum over the native cube (a sigma_v that is
free, same model), which is the stellar route the table replaces: the table must equal it
to round-off at an off-node metallicity. The redshift does not enter the cube, so the
bound holds at every z the pixels sample.

The metallicity is solar-relative ``met_logzsol = -0.052``, absolute log Z -1.900, between
the wNE nodes -2.0 and -1.85387.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from tengri import DEFAULT, Fixed, Observation, SEDModel, SpectrumPrecomp, Uniform
from tengri.observation.spectroscopy import Spectroscopy

pytestmark = pytest.mark.regression_bug

#: Relative tolerance of the table route against the per-draw sum route (round-off).
RTOL = 1e-12
_MET_LOGZSOL_OFF_NODE = -0.052  # absolute log Z -1.900, between the wNE nodes -2.0 and -1.85387
_WAVE_OBS = np.linspace(4000.0, 9000.0, 2000)
_Z_GRID = (0.08, 0.12)


def _build(ssp, sigma_v):
    obs = Observation(
        spectroscopy=Spectroscopy(resample="point", wave_obs=_WAVE_OBS, resolution=2000.0)
    )
    base = SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        approx=SpectrumPrecomp(n_z=16, z_min=0.04, z_max=0.16),
        redshift=Uniform(0.05, 0.15),
        igm={"type": "none"},
        sfh={"type": "tsnorm", "all_params": Fixed(DEFAULT), "log_total_mass": Fixed(10.0)},
        met={
            "type": "delta",
            "met_logzsol": Fixed(_MET_LOGZSOL_OFF_NODE),
            "all_params": Fixed(DEFAULT),
        },
        dust_attenuation={
            "type": "two_component",
            "law": "calzetti",
            "all_params": Fixed(DEFAULT),
        },
        neb={"type": "none"},
    )
    spec = base.spec.merge_observation_params(sigma_v_kms=sigma_v)
    return SEDModel(
        spec, ssp, observation=obs, approx=SpectrumPrecomp(n_z=16, z_min=0.04, z_max=0.16)
    )


def _max_rel_to_median(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


def _stellar_spectrum(model, params):
    """The stellar component's pixel-sampled spectrum and its per-age split (#3541)."""
    derived = model.predict_state(params).derived
    return (
        np.asarray(derived["stellar_spec_lnu_precomp"]),
        np.asarray(derived["stellar_spec_lnu_per_age_precomp"]),
    )


@pytest.mark.parametrize("z", _Z_GRID)
def test_free_z_prebroadened_cube_matches_summed_route_off_node_z(ssp_data_wne, z):
    """The stellar piece of the table route equals the per-draw sum route to round-off.

    Compared at the stellar component's own outputs, which are the table read and nothing
    else: the stellar sum over the cube and over the native templates agrees to 1e-15. The
    full predicted spectrum differs by a further 7e-8 that does not come from this table
    (see the notes on the spectrum-level residual in the commit that added this test).
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cube = _build(ssp_data_wne, Fixed(300.0))
        summed = _build(ssp_data_wne, Uniform(0.0, 2000.0))
    chain = cube._cached_component_chain or cube._build_component_chain()
    assert chain[0]._state.spec_ssp_broadened is not None
    tot_c, age_c = _stellar_spectrum(cube, {"redshift": z})
    tot_s, age_s = _stellar_spectrum(summed, {"redshift": z, "sigma_v_kms": 300.0})
    assert _max_rel_to_median(tot_c, tot_s) <= RTOL
    assert _max_rel_to_median(age_c, age_s) <= RTOL
