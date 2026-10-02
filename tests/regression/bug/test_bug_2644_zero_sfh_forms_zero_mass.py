# SPDX-License-Identifier: BSD-3-Clause
"""Regression tests for #2644: an SFH identically zero inside its support
must form zero mass on BOTH age kernels, not the declared mass.

``delayed_bq`` with ``age_bq_gyr >= age_main_gyr`` clips the main-sequence
SFR negative (hence zero) everywhere; before this fix the "dsps" kernel
published the DECLARED mass anyway (DSPS's own SFR_MIN floor keeps its
histogram weights nonzero for an all-zero input), while "cic" already
published the representable-floor mass. Both kernels now agree.
"""

from __future__ import annotations

import warnings

import jax
import numpy as np
import pytest

import tengri
from tengri import DEFAULT, Fixed
from tengri.components.stellar.component import ZeroSFHWarning
from tengri.cosmology import age_at_z

pytestmark = pytest.mark.regression_bug


@pytest.fixture(scope="module")
def ssp():
    return tengri.load_ssp("fsps_prsc_miles_chabrier", download=False)


@pytest.mark.parametrize("kernel", ["cic", "dsps"])
def test_zero_history_forms_zero_mass_both_kernels(ssp, kernel):
    z = 6.0
    age_main = 0.5 * float(age_at_z(z))  # < default age_bq_gyr bound; forces negative SFR
    cfg = {
        "type": "delayed_bq",
        "all_params": Fixed(DEFAULT),
        "log_total_mass": Fixed(9.0),
        "age_main_gyr": Fixed(age_main),
        "age_bq_gyr": Fixed(age_main + 0.05),  # explicit override past the new registry ceiling
        "age_kernel": kernel,
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = tengri.SEDModel.build(ssp, sfh=cfg, redshift=Fixed(z), neb={"type": "ssp"})
    p = dict(model.spec.sample(jax.random.PRNGKey(0)))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        st = model.predict_state(p)
        assert any(issubclass(w.category, ZeroSFHWarning) for w in caught), (
            f"ZeroSFHWarning not raised on {kernel} kernel for a degenerate SFH"
        )

    sfr = np.asarray(st.derived["sfr_history"])
    lbt = np.asarray(st.derived["sfh_grid_lbt_yr"])
    formed_mass = 10 ** float(st.derived["log_mstar_formed"])

    assert sfr.max() == 0.0, f"{kernel}: SFR should be identically zero, got max {sfr.max()}"
    np.testing.assert_allclose(np.trapezoid(sfr, lbt), 0.0, atol=1e-6)
    # Representable floor, not the declared 1e9.
    assert formed_mass < 1e-20, (
        f"{kernel}: formed mass {formed_mass:.3e} should be ~0, not declared"
    )


def test_nondegenerate_delayed_bq_bit_identical_to_declared(ssp):
    """age_bq < age_main (the common case): unaffected by the #2644 guard."""
    z = 0.5
    age_main = float(age_at_z(z))
    cfg = {
        "type": "delayed_bq",
        "all_params": Fixed(DEFAULT),
        "log_total_mass": Fixed(10.0),
        "age_main_gyr": Fixed(age_main),
        "age_bq_gyr": Fixed(0.3),
        "tau_main_gyr": Fixed(2.0),
        "r_sfr": Fixed(1.0),
    }
    model = tengri.SEDModel.build(ssp, sfh=cfg, redshift=Fixed(z), neb={"type": "ssp"})
    st = model.predict_state({})
    formed_mass = 10 ** float(st.derived["log_mstar_formed"])
    np.testing.assert_allclose(formed_mass, 1e10, rtol=1e-6)


def test_registry_bound_makes_default_priors_unreachable(ssp):
    """The default free age_bq_gyr/age_main_gyr priors cannot produce age_bq > age_main."""
    from tengri.components.stellar.sfh.registry import SFH_REGISTRY

    spec = SFH_REGISTRY["delayed_bq"]
    age_bq_def = spec.params["sfh_delayed_bq_age_bq_gyr"]
    age_main_def = spec.params["sfh_delayed_bq_age_main_gyr"]
    age_bq_prior = age_bq_def.free_prior or age_bq_def.default
    age_main_prior = age_main_def.free_prior or age_main_def.default
    assert age_bq_prior.bounds[1] <= age_main_prior.bounds[0], (
        f"age_bq_gyr ceiling ({age_bq_prior.bounds[1]}) must not exceed "
        f"age_main_gyr floor ({age_main_prior.bounds[0]})"
    )
