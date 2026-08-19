# SPDX-License-Identifier: BSD-3-Clause
"""Regression test for issue #438: WavePrecomp ztable z-interpolation.

The free-z LUT path previously used linear interpolation in z. Linear
interp on a uniform grid is O(h^2) and **non-monotonic in n_z** at
fixed test redshifts — doubling n_z can shift a test point into a
less-favourable cell and raise the error.

The fix is the triweight kernel (Hearin et al. 2023), which is the
canonical smooth-grid interpolant used elsewhere in tengri (SSP, CLOUDY,
SKIRTOR grids). It is C²-continuous and converges monotonically as
the grid is refined.

The remaining residual at off-grid z (~1-2%) is the intrinsic
Charlot-and-Fall dust × per-age factorization approximation in
``Observation.predict_via_precomp`` and is independent of n_z.
"""

import pytest

pytestmark = pytest.mark.regression_bug


def test_waveprecomp_error_bounded_at_high_n_z():
    """At n_z=400 over z∈[0,3], the LUT error stays under the documented 3% ceiling.

    With the triweight z-kernel the original ~3% spike at n_z=200 / z=1.0
    no longer dominates; what remains (~1-2% off-grid) is the intrinsic
    Charlot-and-Fall dust × per-age factorization approximation in
    ``Observation.predict_via_precomp`` and is independent of n_z.
    """
    import jax
    import numpy as np

    jax.config.update("jax_enable_x64", True)
    import tengri

    try:
        ssp = tengri.load_ssp("fsps_prsc_miles_chabrier")
    except (FileNotFoundError, OSError):
        pytest.skip("fsps_prsc_miles_chabrier SSP not available")

    obs = tengri.Observation(
        photometry=tengri.Photometry.from_names(["sdss_u", "sdss_g", "sdss_r", "sdss_i"])
    )
    spec = dict(
        sfh={"type": "tsnorm", "*": tengri.FIXED},
        dust={"law": "power_law", "type": "two_component", "*": tengri.FIXED, "tau_diff": 0.3, "tau_bc": 0.2},
        redshift=tengri.Uniform(0.0, 3.0),
    )
    model_exact = tengri.SEDModel.build(ssp, observation=obs, approx=None, **spec)
    model_lut = tengri.SEDModel.build(
        ssp,
        observation=obs,
        approx=tengri.WavePrecomp(n_z=400, z_min=0.0, z_max=3.0),
        **spec,
    )
    baseline = dict(model_exact.spec.sample(jax.random.PRNGKey(0)))
    # Interior z values — boundary z=0/z=3 hit kernel-support truncation
    # and aren't representative of the in-range LUT accuracy claim.
    for z in (0.5, 1.0, 1.5, 2.0, 2.5):
        params = {**baseline, "redshift": float(z)}
        ex = np.asarray(model_exact.predict_photometry(params))
        lu = np.asarray(model_lut.predict_photometry(params))
        rel = np.abs(lu - ex) / np.maximum(np.abs(ex), 1e-30)
        assert rel.max() < 0.03, f"WavePrecomp(n_z=400) at z={z}: max rel err {rel.max():.3e} > 3%"


def test_waveprecomp_on_grid_is_near_exact():
    """When z falls exactly on a ztable grid point, LUT ≈ exact (dust-free)."""
    import jax
    import numpy as np

    jax.config.update("jax_enable_x64", True)
    import tengri

    try:
        ssp = tengri.load_ssp("fsps_prsc_miles_chabrier")
    except (FileNotFoundError, OSError):
        pytest.skip("fsps_prsc_miles_chabrier SSP not available")

    obs = tengri.Observation(photometry=tengri.Photometry.from_names(["sdss_g", "sdss_r"]))
    spec = dict(
        sfh={"type": "tsnorm", "*": tengri.FIXED},
        redshift=tengri.Uniform(0.0, 3.0),
    )
    model_exact = tengri.SEDModel.build(ssp, observation=obs, approx=None, **spec)
    # n_z=301 over [0,3] puts z=1.0 exactly on grid index 100.
    model_lut = tengri.SEDModel.build(
        ssp,
        observation=obs,
        approx=tengri.WavePrecomp(n_z=301, z_min=0.0, z_max=3.0),
        **spec,
    )
    baseline = dict(model_exact.spec.sample(jax.random.PRNGKey(0)))
    params = {**baseline, "redshift": 1.0}
    ex = np.asarray(model_exact.predict_photometry(params))
    lu = np.asarray(model_lut.predict_photometry(params))
    rel = np.abs(lu - ex) / np.maximum(np.abs(ex), 1e-30)
    # On-grid: triweight collapses to the exact table value at the node,
    # so LUT matches exact to ~1% (filter-convolution / dust-free LUT noise).
    assert rel.max() < 0.02, f"on-grid z=1.0: max rel err {rel.max():.3e} > 2%"
