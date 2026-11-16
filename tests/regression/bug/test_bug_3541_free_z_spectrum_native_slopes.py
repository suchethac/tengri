# SPDX-License-Identifier: BSD-3-Clause
"""#3541: the spectrum table broadened at build reads metallicity from native slopes.

With a fixed velocity dispersion the spectrum table stores the SSP templates broadened
once at build (on the ln-lambda lattice since #3500; before it, for a free redshift, as a
cube on the native grid). Before the native-slope fix the stored templates were read
across metallicity by monotone PCHIP over their own values, which is not the PCHIP of the
native cube, so the table differed from the per-draw route (5.3e-5 at z = 0.08 on the wNE
grid, 4.8e-5 at z = 0.12).

The fix (owner ruling 2026-11-15, "Exact: native slopes") broadens the native PCHIP
tangents with the same linear build and reads the table by cubic Hermite of (values,
slopes). The reference here is the per-draw route (a sigma_v that is free, same model),
which reads the unbroadened table by the same Hermite and then convolves: the two must
agree to round-off at an off-node metallicity. The redshift does not enter the table, so
the bound holds at every z the pixels sample.

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
def test_free_z_table_broadened_at_build_matches_the_per_draw_route_off_node_z(ssp_data_wne, z):
    """The stellar piece of the table broadened at build equals the per-draw route.

    Compared at the stellar component's own outputs on the lattice cells, which are the
    table read and nothing else.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cube = _build(ssp_data_wne, Fixed(300.0))
        # Floored at 20 km/s: the table declines point-sampled pixels whose kernel can
        # reach zero (#3500), and the reference would then be the exact path.
        summed = _build(ssp_data_wne, Uniform(20.0, 2000.0))
    chain = cube._cached_component_chain or cube._build_component_chain()
    assert chain[0]._state.ssp_spec_lut is not None
    assert cube._approx.get("spectrum_precomp") and summed._approx.get("spectrum_precomp")
    tot_c, age_c = _stellar_spectrum(cube, {"redshift": z})
    tot_s, age_s = _stellar_spectrum(summed, {"redshift": z, "sigma_v_kms": 300.0})
    assert _max_rel_to_median(tot_c, tot_s) <= RTOL
    assert _max_rel_to_median(age_c, age_s) <= RTOL
