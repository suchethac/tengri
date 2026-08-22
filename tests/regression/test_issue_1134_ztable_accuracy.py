# SPDX-License-Identifier: BSD-3-Clause
"""#1134: free-z ztable photometry must track the exact path to <1%.

Default n_z=100 gave -4% (des_i, z=0.3) and +40% (GALEX FUV, z=1):
linear z-interpolation across the Lyman-break sweep. This test is the
permanent accuracy gate for the ztable defaults.
"""

import time

import jax
import numpy as np
import pytest

pytestmark = pytest.mark.regression_bug

BANDS = ["galex_fuv", "galex_nuv", "sdss_u", "des_i"]
Z_GRID = np.linspace(0.05, 1.5, 25)
RTOL = 0.01


@pytest.fixture(scope="session")
def ssp_data_for_accuracy(ssp_data_wne):
    """Use real SSP data with Lyman break."""
    return ssp_data_wne


def test_ztable_matches_exact_below_1pct(ssp_data_for_accuracy):
    """Test that WavePrecomp LUT photometry agrees with exact path to < 1%."""
    from tengri import FIXED, SEDModel, Uniform, WavePrecomp
    from tengri.observation import Observation, Photometry

    obs = Observation(photometry=Photometry.from_names(BANDS))
    common = dict(
        ssp_data=ssp_data_for_accuracy,
        observation=obs,
        sfh={"type": "dpl"},
        dust_attenuation={"type": "two_component", "law": "power_law", "all_params": FIXED},
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
        rel = np.max(np.abs(ff - fe) / np.abs(fe))
        worst = max(worst, rel)

    assert worst < RTOL, f"worst ztable error {worst:.1%} exceeds {RTOL:.0%}"


def measure_ztable_error_and_cost(n_z_value, ssp_data_for_accuracy):
    """Measure worst error and build time for a given n_z."""
    from tengri import FIXED, SEDModel, Uniform, WavePrecomp
    from tengri.observation import Observation, Photometry

    obs = Observation(photometry=Photometry.from_names(BANDS))
    common = dict(
        ssp_data=ssp_data_for_accuracy,
        observation=obs,
        sfh={"type": "dpl", "all_params": FIXED},
        dust_attenuation={"type": "two_component", "law": "power_law", "all_params": FIXED},
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
