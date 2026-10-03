# SPDX-License-Identifier: BSD-3-Clause
"""#1134: free-z ztable photometry must track the exact path to <1%.

Default n_z=100 gave -4% (des_i, z=0.3) and +40% (GALEX FUV, z=1):
linear z-interpolation across the Lyman-break sweep. This test is the
permanent accuracy gate for the ztable defaults.

GALEX FUV carries its own budget. Over z = 0.5-1.5 it samples rest 700-900 Å,
below the Lyman limit, where this dusty wNE model has an IGM-free LUT floor of
-0.91 % at fixed z that more sub-bands do not remove (K = 3/5/8: -0.95/-0.91/
-0.90 %; the #2447 Lyman-limit residual). Under the exact IGM fold (the default
since #2445) the free-z worst is 1.10 % (z = 0.956), i.e. that floor plus the
z interpolation. The 0.69 % the node fold read was the node fold's own IGM error
(-1.35 % at fixed z) partly canceling the floor across the grid, not accuracy.
"""

import time

import jax
import numpy as np
import pytest

pytestmark = pytest.mark.regression_bug

BANDS = ["galex_fuv", "galex_nuv", "sdss_u", "des_i"]
Z_GRID = np.linspace(0.05, 1.5, 25)
RTOL = 0.01
#: Per-band override: FUV's IGM-free Lyman-limit floor (0.91 %) plus z interpolation.
BAND_RTOL = {"galex_fuv": 0.015}


@pytest.fixture(scope="session")
def ssp_data_for_accuracy(ssp_data_wne):
    """Use real SSP data with Lyman break."""
    return ssp_data_wne


def test_ztable_matches_exact_below_1pct(ssp_data_for_accuracy):
    """Test that WavePrecomp LUT photometry agrees with exact path to < 1%."""
    from tengri import DEFAULT, Fixed, SEDModel, Uniform, WavePrecomp
    from tengri.observation import Observation, Photometry

    obs = Observation(photometry=Photometry.from_names(BANDS))
    common = dict(
        ssp_data=ssp_data_for_accuracy,
        observation=obs,
        sfh={"type": "dpl"},
        dust_attenuation={
            "type": "two_component",
            "law": "power_law",
            "tau_bc": Uniform(0.0, 4.0),
            "tau_diff": Uniform(0.0, 3.0),
            "all_params": Fixed(DEFAULT),
        },
        redshift=Uniform(0.01, 2.0),
        igm={"type": "inoue"},
    )

    # Exact path (no approximation)
    exact = SEDModel.build(approx=None, **common)

    # Fast path with WavePrecomp LUT (library default)
    fast = SEDModel.build(approx=WavePrecomp(), **common)

    worst = 0.0
    for z in Z_GRID:
        # Sample a random parameter dict and then override redshift
        p_sampled = dict(exact.spec.sample(jax.random.PRNGKey(0)))
        p = p_sampled | {"redshift": float(z)}

        fe = np.asarray(exact.predict_photometry(p))
        ff = np.asarray(fast.predict_photometry(p))
        # Relative error
        rel = np.abs(ff - fe) / np.abs(fe) / np.array([BAND_RTOL.get(b, RTOL) for b in BANDS])
        worst = max(worst, float(np.max(rel)))

    assert worst < 1.0, f"worst ztable error is {worst:.2f}x its band budget"


def measure_ztable_error_and_cost(n_z_value, ssp_data_for_accuracy):
    """Measure worst error and build time for a given n_z."""
    from tengri import DEFAULT, Fixed, SEDModel, Uniform, WavePrecomp
    from tengri.observation import Observation, Photometry

    obs = Observation(photometry=Photometry.from_names(BANDS))
    common = dict(
        ssp_data=ssp_data_for_accuracy,
        observation=obs,
        sfh={"type": "dpl"},
        dust_attenuation={
            "type": "two_component",
            "law": "power_law",
            "tau_bc": Uniform(0.0, 4.0),
            "tau_diff": Uniform(0.0, 3.0),
            "all_params": Fixed(DEFAULT),
        },
        redshift=Uniform(0.01, 2.0),
        igm={"type": "inoue"},
    )

    # Exact path
    exact = SEDModel.build(approx=None, **common)

    # Fast path with custom n_z
    start = time.perf_counter()
    fast = SEDModel.build(approx=WavePrecomp(n_z=n_z_value), **common)
    build_time = time.perf_counter() - start

    worst = 0.0
    for z in Z_GRID:
        p_sampled = dict(exact.spec.sample(jax.random.PRNGKey(0)))
        p = p_sampled | {"redshift": float(z)}

        fe = np.asarray(exact.predict_photometry(p))
        ff = np.asarray(fast.predict_photometry(p))
        rel = np.max(np.abs(ff - fe) / np.abs(fe))
        worst = max(worst, rel)

    return worst, build_time


@pytest.mark.parametrize("n_z_value", [100, 200, 400, 800])
def test_ztable_accuracy_sweep(n_z_value, ssp_data_for_accuracy):
    """Sweep n_z values to find best accuracy/cost tradeoff."""
    worst, build_time = measure_ztable_error_and_cost(n_z_value, ssp_data_for_accuracy)
    print(f"n_z={n_z_value:3d}: worst_error={worst:7.2%}, build_time={build_time:6.2f}s")
