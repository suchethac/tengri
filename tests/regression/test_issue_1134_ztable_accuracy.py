# SPDX-License-Identifier: BSD-3-Clause
"""#1134: free-z ztable photometry must track the exact path to <1%.

Default n_z=100 gave -4% (des_i, z=0.3) and +40% (GALEX FUV, z=1):
linear z-interpolation across the Lyman-break sweep. This test is the
permanent accuracy gate for the ztable defaults.

GALEX FUV carries its own budget. Over z = 0.5-1.5 it samples rest 700-900 Å,
below the Lyman limit, where this dusty wNE model has an IGM-free LUT floor of
-0.91 % at fixed z that more sub-bands do not remove (K = 3/5/8: -0.95/-0.91/
-0.90 %; the #2447 Lyman-limit residual). Under the exact IGM fold (the default
since #2445) the free-z worst was 1.10 % (z = 0.956), i.e. that floor plus the
z interpolation. The 0.69 % the node fold read was the node fold's own IGM error
(-1.35 % at fixed z) partly canceling the floor across the grid, not accuracy.

Re-measured after #2724 and #2749 (fixed redshift, same model, ``WavePrecomp``
against the exact path, FUV): -0.04 / -0.09 / -0.38 / -0.59 / -0.09 % at
z = 0.6 / 0.8 / 0.95 / 0.975 / 1.1. The fixed-z floor is now -0.59 % at its worst
(z = 0.975), down from the -0.91 % quoted above, and the free-z worst, 0.59 % at
z = 0.97, is that floor: the table's own interpolation adds nothing visible at
n_z = 250. The 1.5 % FUV budget is unchanged.

The redshifts where the rest-frame Lyman limit (911.76 A) crosses a band's
support limit or half-peak flank are sampled here (#2749): the band flux
collapses there, and a grid of 25 points that skips z = 0.94-0.99 for FUV never
sees the worst of it (8.5 % at z = 0.983 with the triweight window and uniform
nodes). The z-table now reads by local cubic Hermite interpolation and carries
nodes on those crossings.
"""

import time

import jax
import numpy as np
import pytest

pytestmark = pytest.mark.regression_bug

BANDS = ["galex_fuv", "galex_nuv", "sdss_u", "des_i"]
Z_LO, Z_HI = 0.05, 1.5
LYMAN_LIMIT_AA = 911.76
#: Offsets [redshift] around each crossing at which the table is probed.
CROSSING_OFFSETS = (-0.03, -0.01, 0.0, 0.01, 0.03)


def _band_crossing_redshifts(bands, z_lo=Z_LO, z_hi=Z_HI):
    """z at which 911.76 (1+z) A hits each band's support limits and half-peak flanks."""
    from tengri.observation import Photometry

    phot = Photometry.from_names(bands)
    out = []
    for w, t in zip(phot.filter_waves, phot.filter_trans, strict=True):
        w, t = np.asarray(w, float), np.asarray(t, float)
        half = np.flatnonzero(t >= 0.5 * t.max())
        sup = np.flatnonzero(t > 0.0)
        for idx in (sup[0], sup[-1], half[0], half[-1]):
            z = w[idx] / LYMAN_LIMIT_AA - 1.0
            if z_lo < z < z_hi:
                out.append(z)
    return np.asarray(out)


def _probe_redshifts(bands=BANDS, z_lo=Z_LO, z_hi=Z_HI, n_uniform=25):
    """Uniform grid plus the crossings of every band, each at the probe offsets."""
    around = [
        zc + dz for zc in _band_crossing_redshifts(bands, z_lo, z_hi) for dz in CROSSING_OFFSETS
    ]
    z = np.concatenate([np.linspace(z_lo, z_hi, n_uniform), np.asarray(around)])
    return np.unique(np.round(z[(z >= z_lo) & (z <= z_hi)], 6))


Z_GRID = _probe_redshifts()
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
