# SPDX-License-Identifier: BSD-3-Clause
"""Regression: dense SFH integrand for non-parametric SFHs (#758).

Non-parametric SFHs (continuity / dirichlet / post-starburst) were evaluated
only at the ~107 coarse SSP template ages before being handed to DSPS, which
linearly smeared each bin-edge transition and produced a 2-4.5 % optical
residual vs Prospector. The fix evaluates the SFH on a dense integrand grid
(``_refine_sfh_table_ages``) for the non-field delta-metallicity path; DSPS
still returns age weights on ``ssp_lg_age_gyr`` so mass is conserved and every
downstream consumer is unchanged.

The full <1 % residual-vs-Prospector check lives in
``reproduction/prospector/01_prospector.py`` (needs python-fsps + SPS_HOME);
here we guard the mass-conserving, JIT-safe properties that hold without FSPS.

Issue: https://github.com/suchethac/tengri/issues/758
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import FIXED, Fixed, SEDModel
from tengri.components.stellar.component import (
    _build_dsps_sfh_table,
    _refine_sfh_table_ages,
)

pytestmark = pytest.mark.conservation


def test_refine_sfh_table_ages_is_dense_monotonic_and_spans_range():
    """The refined grid is strictly increasing, denser, and spans the SSP ages."""
    ssp_ages_yr = 10.0 ** jnp.linspace(5.0, 10.13, 107)  # ~1e5 yr – 13.5 Gyr
    fine = _refine_sfh_table_ages(ssp_ages_yr, factor=16)
    fine_np = np.asarray(fine)
    assert fine_np.shape[0] == (107 - 1) * 16 + 1
    assert np.all(np.diff(fine_np) > 0.0)  # strictly increasing
    assert np.isclose(fine_np[0], float(ssp_ages_yr[0]))
    assert np.isclose(fine_np[-1], float(ssp_ages_yr[-1]))


def test_build_dsps_sfh_table_conserves_trapezoidal_mass():
    """total_mass equals ∫SFR dt over the ascending cosmic-time table."""
    age_yr = 10.0 ** jnp.linspace(5.0, 10.13, 200)
    sfr = jnp.full_like(age_yr, 3.0)  # constant 3 Msun/yr
    t_obs_gyr = 13.7
    t_asc, sfr_asc, total = _build_dsps_sfh_table(age_yr, sfr, t_obs_gyr)
    assert np.all(np.diff(np.asarray(t_asc)) > 0.0)  # strictly increasing
    expected = float(np.trapezoid(np.asarray(sfr_asc), np.asarray(t_asc) * 1e9))
    assert np.isclose(float(total), expected, rtol=1e-6)
    assert float(total) > 0.0


@pytest.mark.parametrize(
    "sfh",
    [
        {"type": "continuity", "log_total_mass": Fixed(10.0), "*": FIXED},
        {"type": "dirichlet", "log_total_mass": Fixed(10.0), "*": FIXED},
    ],
)
def test_nonparametric_sfh_conserves_formed_mass(synthetic_ssp_wide, sfh):
    """∫SFR dt = M_formed: the dense integrand must not break mass conservation."""
    model = SEDModel.build(
        ssp_data=synthetic_ssp_wide,
        sfh=sfh,
        dust={"law": "power_law", "type": "two_component", "tau_bc": Fixed(0.0), "tau_diff": Fixed(0.0), "*": FIXED},
        redshift=Fixed(0.0),
    )
    state = model.predict_state({})
    formed = float(np.sum(np.asarray(state.derived["age_weights"])))
    # Tolerance accommodates the coarse 25-age synthetic SSP grid (real grids
    # conserve to ~0.04%); a true mass-conservation break would be ≫10%.
    assert np.isclose(np.log10(formed), 10.0, atol=1.5e-2), (
        f"formed mass log10={np.log10(formed):.5f} (expected 10.0) for {sfh['type']}"
    )
    sed = np.asarray(state.sed_intrinsic)
    assert np.all(np.isfinite(sed)) and np.all(sed >= 0.0)


def test_nonparametric_sfh_is_jit_safe(synthetic_ssp_wide):
    """The dense-integrand path stays jittable (no Python branch on traced values)."""
    model = SEDModel.build(
        ssp_data=synthetic_ssp_wide,
        sfh={"type": "continuity", "log_total_mass": Fixed(10.0), "*": FIXED},
        dust={"law": "power_law", "type": "two_component", "tau_bc": Fixed(0.0), "tau_diff": Fixed(0.0), "*": FIXED},
        redshift=Fixed(0.0),
    )
    sed = jax.jit(lambda p: model.predict_state(p).sed_intrinsic)({})
    assert np.all(np.isfinite(np.asarray(sed)))
